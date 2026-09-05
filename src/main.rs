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
    let matches = opts.parse(args)?;

    if matches.opt_present("h") {
        usage(&program);
        return Ok(());
    }

    let map_output_file_tmp = matches.opt_str("o");
    let map_output_file = map_output_file_tmp.as_ref();
    let use_gpu = matches.opt_present("gpu") && cfg!(feature = "gpu");

    let files = matches.free;

    if files.len() < 2 {
        usage(&program);
        return Err("You must specify at least 2 files to compare".into());
    }

    if use_gpu {
        #[cfg(feature = "gpu")]
        return run_gpu(map_output_file, &files);
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
/// through the Vulkan backend. Output format is byte-identical to the CPU
/// path (`{dssim:.8}\t{file}`). Map writing is CPU-only for now: the GPU
/// path returns pooled scores, not SsimMaps (Phase G limitation).
#[cfg(feature = "gpu")]
fn run_gpu(map_output_file: Option<&String>, files: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use imgref::ImgVec;
    if map_output_file.is_some() {
        eprintln!("warning: --gpu does not support -o map output yet; ignoring -o");
    }

    let context = match gpu_backend::Context::new() {
        Ok(context) => std::sync::Arc::new(context),
        // Plan §6 Phase G: CPU fallback when no Vulkan device is available.
        Err(e) => {
            eprintln!("note: --gpu unavailable ({e}); falling back to CPU");
            return run_cpu_simple(files);
        }
    };
    let gpu = gpu_backend::GpuSsim::new(context)?;
    type GpuErr = Box<dyn std::error::Error + Send + Sync>;

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
    scope_result.map_err(|e| -> Box<dyn std::error::Error> { e })
}

/// CPU fallback used when `--gpu` is requested but no Vulkan device is
/// available. Same output as the default CPU path.
fn run_cpu_simple(files: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let attr = dssim::Dssim::new();
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
        let (dssim, _) = attr.compare(&original, modified);
        println!("{dssim:.8}\t{}", file2);
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
