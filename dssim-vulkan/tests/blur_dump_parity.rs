//! Phase C real-image parity (VULKAN_PORT_PLAN.md §4 Phase 3 exit criteria):
//! GPU blur vs the CPU pipeline's blur outputs for the golden fixtures.
//!
//! Uses the M0 dump harness (`dssim-dumps` feature) to materialize the CPU
//! pipeline's per-scale Lab planes (`lab_plane`) and their blurred means
//! (`chan_mu` = `blur(lab plane)` on the CPU). Each dump pair becomes one
//! GPU-vs-CPU blur comparison: upload the Lab plane, run `blur_gpu`, diff
//! against the CPU's `mu` dump. Covers all scales and channels of the full
//! test1/test2 pair (both images), both sub-image crops, the gray pair, and
//! the alpha pair (premultiply + dither path).

use std::path::Path;
use std::sync::Arc;

use dssim_core::dumps::{self, Dump};
use dssim_core::{new, ToRGBAPLU};
use dssim_vulkan::{blur_gpu, Context};
use imgref::Img;

const TOL: f64 = 2e-6;

fn decode_rgba(path: &str) -> imgref::ImgVec<dssim_core::RGBAPLU> {
    let file = lodepng::decode32_file(path).unwrap();
    Img::new(file.buffer.to_rgbaplu(), file.width, file.height)
}

#[test]
fn gpu_blur_matches_cpu_dumps() {
    let _ = env_logger::try_init();
    let context = Arc::new(Context::new().expect("Vulkan context"));

    let dir = std::env::temp_dir().join(format!("dssim-blur-dumps-{}", std::process::id()));
    let _guard = dumps::enable(&dir).unwrap();

    // Same scenario set as the M0 golden generator.
    let d = new();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let _ = d.create_image(&img1).unwrap();
    let _ = d.create_image(&img2).unwrap();
    let _ = d.create_image(&img1.sub_image(2, 3, 44, 33)).unwrap();
    let _ = d.create_image(&img2.sub_image(17, 9, 44, 33)).unwrap();
    let _ = d.create_image(&img1.sub_image(22, 8, 61, 40)).unwrap();
    let _ = d.create_image(&img2.sub_image(22, 8, 61, 40)).unwrap();
    let _ = d.create_image(&decode_rgba("../tests/alpha1.png")).unwrap();
    let _ = d.create_image(&decode_rgba("../tests/alpha2.png")).unwrap();
    drop(_guard);

    // Pair each lab_plane dump (input) with the chan_mu dump of the same
    // scale/channel/run. For chroma channels (c > 0) the CPU pipeline first
    // applies the in-place chroma pre-blur (dssim.rs preprocess:
    // `blur_in_place` then `mu = blur(img)`), so `chan_mu` is a double blur
    // and `chan_img` is the pre-blurred plane — the GPU test reproduces both
    // stages, mirroring what Phase D's pipeline must do.
    let mut checked = 0usize;
    let mut chroma_pre_checked = 0usize;
    let mut max_overall = 0.0f64;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if !name.starts_with("lab_plane.") {
            continue;
        }
        let mu_name = name.replacen("lab_plane", "chan_mu", 1);
        let img_name = name.replacen("lab_plane", "chan_img", 1);
        let lab = Dump::read(dir.join(&name)).unwrap();
        let mu = Dump::read(dir.join(&mu_name)).unwrap();
        let chan_img = Dump::read(dir.join(&img_name)).unwrap();

        assert_eq!(
            (lab.header.width, lab.header.height),
            (mu.header.width, mu.header.height),
            "{name}: dims mismatch vs {mu_name}"
        );
        assert_eq!(lab.header.stride, lab.header.width, "{name}: expected tight dump");
        let (w, h) = (lab.header.width as usize, lab.header.height as usize);
        assert_eq!(lab.data.len(), w * h);
        let channel = lab.header.channel;

        // Stage 1: single blur of the raw Lab plane.
        let gpu_pre = blur_gpu(&context, &lab.data, w, h, w)
            .unwrap_or_else(|e| panic!("{name}: GPU blur failed: {e}"));

        if channel == 0 {
            // L channel: mu = blur(lab) directly.
            for (i, (c, g)) in mu.data.iter().zip(gpu_pre.iter()).enumerate() {
                let diff = (f64::from(*c) - f64::from(*g)).abs();
                if diff > max_overall {
                    max_overall = diff;
                }
                assert!(
                    diff <= TOL,
                    "{name}: pixel {i} (x={}, y={}) diverged: cpu={} gpu={} ({diff:.3e} > {TOL:.0e})",
                    i % w,
                    i / w,
                    c,
                    g
                );
            }
        } else {
            // Chroma: stage 1 output must equal the CPU's in-place pre-blur.
            for (i, (c, g)) in chan_img.data.iter().zip(gpu_pre.iter()).enumerate() {
                let diff = (f64::from(*c) - f64::from(*g)).abs();
                if diff > max_overall {
                    max_overall = diff;
                }
                assert!(
                    diff <= TOL,
                    "{name} pre-blur: pixel {i} diverged: cpu={} gpu={} ({diff:.3e} > {TOL:.0e})",
                    c,
                    g
                );
            }
            chroma_pre_checked += 1;

            // Stage 2: mu = blur(pre-blurred plane).
            let gpu_mu = blur_gpu(&context, &gpu_pre, w, h, w)
                .unwrap_or_else(|e| panic!("{name}: GPU re-blur failed: {e}"));
            for (i, (c, g)) in mu.data.iter().zip(gpu_mu.iter()).enumerate() {
                let diff = (f64::from(*c) - f64::from(*g)).abs();
                if diff > max_overall {
                    max_overall = diff;
                }
                assert!(
                    diff <= TOL,
                    "{name} mu: pixel {i} diverged: cpu={} gpu={} ({diff:.3e} > {TOL:.0e})",
                    c,
                    g
                );
            }
        }
        checked += 1;
    }

    assert!(checked >= 40, "expected 40+ lab/mu pairs across scenarios, got {checked}");
    eprintln!(
        "gpu_blur_matches_cpu_dumps: {checked} planes ({chroma_pre_checked} chroma pre-blur+mu chains), max_abs={max_overall:.3e}"
    );
    assert!(
        max_overall <= TOL,
        "overall max abs {max_overall:.3e} exceeds {TOL:.0e}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Sanity: the dump directory is scoped to this test process.
#[test]
fn dump_dir_helper_is_unique() {
    let a = std::env::temp_dir().join(format!("dssim-blur-dumps-{}", std::process::id()));
    let p = Path::new(&a);
    assert!(p.file_name().is_some());
}
