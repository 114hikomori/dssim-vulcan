/*
 * © 2011-2022 Kornel Lesiński. All rights reserved.
 *
 * This file is part of DSSIM.
 *
 * DSSIM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * DSSIM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the license along with DSSIM.
 * If not, see <http://www.gnu.org/licenses/agpl.txt>.
 */
#![allow(clippy::manual_range_contains)]
use getopts::Options;
#[cfg(feature = "threads")]
use rayon::prelude::*;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

#[cfg(feature = "gpu")]
use dssim_vulkan as gpu_backend;

fn usage(argv0: &str) {
    eprintln!("\
       Usage: {argv0} original.png modified.png [modified.png...]\
     \n   or: {argv0} -o difference.png original.png modified.png\n\n\
       Compares first image against subsequent images, and outputs\n\
       1/SSIM-1 difference for each of them in order (0 = identical).\n\n\
       Images must have identical size, but may have different gamma & depth.\n\
       \n--gpu uses the experimental Vulkan backend (falls back to CPU).\n\
       --gpu-prep=device opts into GPU-side pyramid downsample (bitwise-equal,\n\
       experimental; default is --gpu-prep=cpu).\n\
       \nVersion {} https://kornel.ski/dssim\n", env!("CARGO_PKG_VERSION"));
}

#[inline(always)]
fn to_byte(i: f32) -> u8 {
    if i <= 0.0 {0}
    else if i >= 255.0/256.0 {255}
    else {(i * 256.0) as u8}
}

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        if let Some(s) = e.source() {
            eprintln!("  {s}");
        }
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args();
    let program = args.next().unwrap_or_default();

    let mut opts = Options::new();
    opts.optopt("o", "", "set output file name", "NAME");
    opts.optflag("h", "help", "print this help menu");
    opts.optflag("", "gpu", "use the experimental Vulkan compute backend (falls back to CPU if unavailable)");
    opts.optopt("", "gpu-prep", "with --gpu: pyramid prep 'cpu' (default) or 'device' (GPU-side downsample, opt-in)", "cpu|device");
    let matches = opts.parse(args)?;

    if matches.opt_present("h") {
        usage(&program);
        return Ok(());
    }

    let map_output_file_tmp = matches.opt_str("o");
    let map_output_file = map_output_file_tmp.as_ref();
    let gpu_prep = matches.opt_str("gpu-prep");
    // BH14: --gpu is advertised in help even in gpu-less builds; if the flag is
    // given but the feature is compiled out, say so instead of silently using
    // the CPU path (the user thinks they're testing the GPU backend).
    let gpu_requested = matches.opt_present("gpu");
    if gpu_requested && !cfg!(feature = "gpu") {
        eprintln!("warning: --gpu requested but this build has no Vulkan support (gpu feature off); using CPU");
    }
    let use_gpu = gpu_requested && cfg!(feature = "gpu");
    // BH37 (related nit): --gpu-prep only affects the GPU path; if given without
    // an active --gpu it is silently ignored. Say so rather than let the user
    // think device-prep is engaged.
    if gpu_prep.is_some() && !use_gpu {
        eprintln!("warning: --gpu-prep has no effect without --gpu (ignored)");
    }

    let files = matches.free;

    if files.len() < 2 {
        usage(&program);
        return Err("You must specify at least 2 files to compare".into());
    }

    if use_gpu {
        #[cfg(feature = "gpu")]
        return run_gpu(map_output_file, &files, gpu_prep.as_deref());
        #[cfg(not(feature = "gpu"))]
        unreachable!("use_gpu requires the gpu feature");
    }

    let (images_send, mut images_recv) = ordered_channel::bounded(2);
    let (filenames_send, filenames_recv) = crossbeam_channel::unbounded();
    let mut attr = dssim::Dssim::new();
    if map_output_file.is_some() {
        attr.set_save_ssim_maps(8);
    }

    std::thread::scope(|scope| {
        let decode_thread = || {
            let images_send = images_send; // ensure it's moved, and attr isn't
            filenames_recv.into_iter().try_for_each(|(i, file): (usize, PathBuf)| {
                dssim::load_image(&attr, &file)
                    .map_err(|e| format!("Can't load {}, because: {e}", file.display()))
                    .and_then(|image| images_send.send(i, (file, image)).map_err(|_| "Aborted".into()))
            })
        };

        let threads = [
            scope.spawn(decode_thread.clone()),
            scope.spawn(decode_thread),
        ];

        let result = (|| {
            files.into_iter().map(PathBuf::from).enumerate()
                .try_for_each(move |f| filenames_send.send(f))?;

            let (file1, original) = images_recv.next().ok_or("Can't load any images")?;

            for (file2, modified) in images_recv {
                if original.width() != modified.width() || original.height() != modified.height() {
                    return Err(format!("Image {} has a different size ({}x{}) than {} ({}x{})\n",
                        file2.display(), modified.width(), modified.height(),
                        file1.display(), original.width(), original.height()).into());
                }

                let (dssim, ssim_maps) = attr.compare(&original, modified);

                println!("{dssim:.8}\t{}", file2.display());

                if let Some(map_output_file) = map_output_file {
                    write_ssim_maps(&ssim_maps, map_output_file)?;
                }
            }
            Ok(())
        })();

        threads.into_iter().try_for_each(|t| t.join().map_err(|_| "thread panicked; this is a bug")?)?;
        result
    })
}

/// The `--gpu` comparison path. Image decoding, color profiles, sRGB
/// linearization, premultiplication, and the 2×2 downsample stay on the CPU
/// (dssim-core); the Lab conversion, statistics, SSIM combine, and score run
/// through the Vulkan backend. Output uses the same `{dssim:.8}\t{file}`
/// *format* as the CPU path, but the score *values* can differ in the last
/// decimals from GPU floating-point drift (within the 5e-6 parity bound; not
/// byte-identical — see CHECKPOINT M6). Map writing is CPU-only for now: the
/// GPU path returns pooled scores, not SsimMaps (Phase G limitation).
#[cfg(feature = "gpu")]
fn run_gpu(
    map_output_file: Option<&String>,
    files: &[String],
    gpu_prep: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use imgref::ImgVec;
    // Tier 5: opt-in device-side pyramid prep. Default (unset or "cpu") keeps the
    // fully-tested CPU-prep path; "device" opts into the GPU 2x2 downsample
    // (bitwise-equal, Mode A parity). Anything else is a hard error, not a
    // silent fall-through.
    let prep_mode = match gpu_prep {
        None | Some("cpu") => gpu_backend::PrepMode::Cpu,
        Some("device") => gpu_backend::PrepMode::Device,
        Some(other) => {
            return Err(format!("--gpu-prep must be 'cpu' or 'device', got '{other}'").into())
        }
    };
    let context = match gpu_backend::Context::new() {
        Ok(context) => std::sync::Arc::new(context),
        // BH13: the CPU fallback CAN write -o maps, so honor them here. The old
        // code warned "ignoring -o" BEFORE even trying the context, then fell
        // back to a CPU path that never received map_output_file -- exit 0, no
        // maps, and a misleading warning.
        Err(e) => {
            eprintln!("note: --gpu unavailable ({e}); falling back to CPU");
            return run_cpu_simple(files, map_output_file);
        }
    };
    // BH13: only now that the GPU path is actually running is -o genuinely
    // unsupported (the GPU path returns pooled scores, not SsimMaps).
    if map_output_file.is_some() {
        eprintln!("warning: --gpu does not support -o map output yet; ignoring -o");
    }
    // Positive signal that the GPU path is actually running. The gpu_cli test
    // asserts this line is present (not merely that the fallback note is
    // absent) so the check cannot silently rot if the fallback wording changes.
    let device_name = context.device_name().to_owned();
    // BH2: keep a clone of the context so the scope below can call
    // `supports_size` (GpuSsim::new otherwise consumes the Arc).
    let gpu = gpu_backend::GpuSsim::with_prep_mode(context.clone(), prep_mode)?;
    eprintln!("dssim: gpu device: {device_name}");
    // Tier 5: make the active prep mode explicit (like the BH6 upload-path line)
    // so a "--gpu-prep=device" run is verifiable in logs, not assumed.
    eprintln!(
        "dssim: gpu prep: {}",
        match gpu.prep_mode() {
            gpu_backend::PrepMode::Cpu => "cpu",
            gpu_backend::PrepMode::Device => "device",
        }
    );
    type GpuErr = Box<dyn std::error::Error + Send + Sync>;

    // BH2: set inside the scope when the device limits can't handle the size;
    // Cell is fine because only the main thread (the result closure) touches it.
    let gpu_limits_exceeded = std::cell::Cell::new(false);
    let (images_send, mut images_recv) = ordered_channel::bounded(2);
    let (filenames_send, filenames_recv) = crossbeam_channel::unbounded();
    let scope_result: Result<(), GpuErr> = std::thread::scope(|scope| -> Result<(), GpuErr> {
        // Decoding produces raw bitmaps (pre-GpuSsim), so two decode threads
        // are safe: GpuSsim isn't touched here.
        let decode_thread = || {
            let images_send = images_send; // ensure it's moved
            filenames_recv.into_iter().try_for_each(|(i, file): (usize, PathBuf)| {
                let img: ImgVec<dssim::RGBAPLU> = dssim::load_image_rgba(&file)
                    .map_err(|e| -> GpuErr { format!("Can't load {}, because: {e}", file.display()).into() })?;
                images_send.send(i, (file, img)).map_err(|_| -> GpuErr { "Aborted".into() })
            })
        };

        let threads = [
            scope.spawn(decode_thread.clone()),
            scope.spawn(decode_thread),
        ];

        let result = (|| -> Result<(), GpuErr> {
            files.iter().map(PathBuf::from).enumerate()
                .try_for_each(move |f| filenames_send.send(f))?;

            let (file1, original) = images_recv.next().ok_or("Can't load any images")?;
            // BH2: if this device's limits can't handle the image size, stop the
            // GPU path cleanly and let the caller fall back to CPU (matching the
            // "--gpu ... falls back to CPU if unavailable" contract) rather than
            // mis-execute on a minimum-conforming driver.
            if !context.supports_size(original.width(), original.height()) {
                gpu_limits_exceeded.set(true);
                return Ok(());
            }
            let original_gpu = gpu.create_image(&original)?;

            for (file2, modified) in images_recv {
                if original.width() != modified.width() || original.height() != modified.height() {
                    return Err(format!("Image {} has a different size ({}x{}) than {} ({}x{})",
                        file2.display(), modified.width(), modified.height(),
                        file1.display(), original.width(), original.height()).into());
                }

                let modified_gpu = gpu.create_image(&modified)?;
                let dssim = gpu.compare(&original_gpu, &modified_gpu)?;

                println!("{dssim:.8}\t{}", file2.display());
            }
            Ok(())
        })();

        for t in threads {
            match t.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err("thread panicked; this is a bug".into()),
            }
        }
        result
    });
    // BH2: the GPU path bailed out because the device limits can't handle the
    // size -- fall back to CPU (run_cpu_simple reloads the files) rather than
    // exit 1, honoring the "--gpu falls back if unavailable" contract.
    if gpu_limits_exceeded.get() {
        eprintln!("note: --gpu device limits can't handle this image size; falling back to CPU");
        return run_cpu_simple(files, map_output_file);
    }
    scope_result.map_err(|e| -> Box<dyn std::error::Error> { e })
}

/// CPU fallback used when `--gpu` is requested but can't run (no Vulkan device,
/// or BH2 device limits). Same output as the default CPU path. BH13: it also
/// honors `-o` map output, since the CPU path can write maps even though the
/// GPU path cannot.
fn run_cpu_simple(files: &[String], map_output_file: Option<&String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut attr = dssim::Dssim::new();
    if map_output_file.is_some() {
        attr.set_save_ssim_maps(8);
    }
    let mut files = files.iter();
    let file1 = files.next().ok_or("You must specify at least 2 files to compare")?;
    let original = dssim::load_image(&attr, file1)
        .map_err(|e| format!("Can't load {}, because: {e}", file1))?;

    for file2 in files {
        let modified = dssim::load_image(&attr, file2)
            .map_err(|e| format!("Can't load {}, because: {e}", file2))?;
        if original.width() != modified.width() || original.height() != modified.height() {
            return Err(format!("Image {} has a different size ({}x{}) than {} ({}x{})",
                file2, modified.width(), modified.height(), file1, original.width(), original.height()).into());
        }
        let (dssim, ssim_maps) = attr.compare(&original, modified);
        println!("{dssim:.8}\t{}", file2);
        if let Some(map_output_file) = map_output_file {
            write_ssim_maps(&ssim_maps, map_output_file)?;
        }
    }
    Ok(())
}

fn write_ssim_maps(ssim_maps: &[dssim_core::SsimMap], map_output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "threads")]
    let ssim_maps_iter = ssim_maps.par_iter();
    #[cfg(not(feature = "threads"))]
    let ssim_maps_iter = ssim_maps.iter();
    ssim_maps_iter.enumerate().try_for_each(|(n, map_meta)| {
        let avgssim = map_meta.ssim as f32;
        let out: Vec<_> = map_meta.map.pixels().map(|ssim|{
            let max = 1_f32 - ssim;
            let maxsq = max * max;
            rgb::RGBA8 {
                r: to_byte(maxsq * 16.0),
                g: to_byte(max * 3.0),
                b: to_byte(max / ((1_f32 - avgssim) * 4_f32)),
                a: 255,
            }
        }).collect();
        lodepng::encode32_file(format!("{map_output_file}-{n}.png"), &out, map_meta.map.width(), map_meta.map.height())
            .map_err(|e| {
                format!("Can't write {map_output_file}: {e}")
            })
    })?;
    Ok(())
}

#[test]
fn image_gray() {
    let attr = dssim::Dssim::new();

    let g1 = dssim::load_image(&attr, "tests/gray1-rgba.png").unwrap();
    let g2 = dssim::load_image(&attr, "tests/gray1-pal.png").unwrap();
    let g3 = dssim::load_image(&attr, "tests/gray1-gray.png").unwrap();
    let g4 = dssim::load_image(&attr, "tests/gray1.jpg").unwrap();

    let (diff, _) = attr.compare(&g1, g2);
    assert!(diff < 0.00001, "{diff}");

    let (diff, _) = attr.compare(&g1, g3);
    assert!(diff < 0.00001, "{diff}");

    let (diff, _) = attr.compare(&g1, g4);
    assert!(diff < 0.00006, "{diff}");
}

#[test]
fn image_gray_profile() {
    let attr = dssim::Dssim::new();

    let gp1 = dssim::load_image(&attr, "tests/gray-profile.png").unwrap();
    let gp2 = dssim::load_image(&attr, "tests/gray-profile2.png").unwrap();
    let gp3 = dssim::load_image(&attr, "tests/gray-profile.jpg").unwrap();

    let (diff, _) = attr.compare(&gp1, gp2);
    assert!(diff < 0.0003, "{}", diff);

    let (diff, _) = attr.compare(&gp1, gp3);
    assert!(diff < 0.0003, "{}", diff);
}

#[test]
fn image_load1() {
    let attr = dssim::Dssim::new();
    let prof_jpg = dssim::load_image(&attr, "tests/profile.jpg").unwrap();
    let prof_png = dssim::load_image(&attr, "tests/profile.png").unwrap();
    let (diff, _) = attr.compare(&prof_jpg, prof_png);
    assert!(diff <= 0.002);

    let strip_jpg = dssim::load_image(&attr, "tests/profile-stripped.jpg").unwrap();
    let (diff, _) = attr.compare(&strip_jpg, prof_jpg);
    assert!(diff > 0.008, "{}", diff);

    let strip_png = dssim::load_image(&attr, "tests/profile-stripped.png").unwrap();
    let (diff, _) = attr.compare(&strip_jpg, strip_png);
    assert!(diff > 0.009, "{}", diff);
}

#[test]
fn rgblu_input() {
    use dssim::{Dssim, RGBLU};
    use imgref::{Img, ImgRef, ImgVec};

    let ctx = Dssim::new();
    let im: ImgVec<RGBLU> = Img::new(vec![rgb::RGB::new(0., 0., 0.)], 1, 1);
    let imr: ImgRef<'_, RGBLU> = im.as_ref();
    ctx.create_image(&imr);
}
