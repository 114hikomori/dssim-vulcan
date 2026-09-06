//! Phase D parity (VULKAN_PORT_PLAN.md §4 Phase 4): GPU SSIM map + CPU-pooled
//! single-scale score vs the CPU pipeline's dumps. The dumps provide the
//! blurred statistics (chan_mu, chan_sq_blur, cross_blur) and the reference
//! SSIM map / per-scale scores for: the full test1-vs-test2 pair, both
//! png_compare sub-image crops, the alpha pair, a gray (1-channel) pair, and
//! the identity case (map must be exactly 1.0 everywhere).
//!
//! Pooling replicates dssim.rs:299-302 exactly (f64, per-scale power term);
//! the plan keeps pooling on the CPU — the GPU only produces the map.
//! Bounds: map ≤ 2e-6 (plan §5), score ≤ 5e-6 (Phase 4 exit).

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use dssim_core::dumps::{self, Dump};
use dssim_core::{new, ToRGBAPLU};
use dssim_vulkan::{ssim_combine_gpu, Context};
use imgref::Img;

const MAP_TOL: f64 = 2e-6;
const SCORE_TOL: f64 = 5e-6;

/// The dump sink is a process-global, so dump-generating tests must run one
/// at a time (same pattern as dssim-core/tests/dump_goldens.rs).
static GEN_LOCK: Mutex<()> = Mutex::new(());

fn gen_lock() -> MutexGuard<'static, ()> {
    GEN_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn decode_rgba(path: &str) -> imgref::ImgVec<dssim_core::RGBAPLU> {
    let file = lodepng::decode32_file(path).unwrap();
    Img::new(file.buffer.to_rgbaplu(), file.width, file.height)
}

fn synth_gray(width: usize, height: usize, seed: u64) -> imgref::ImgVec<f32> {
    let mut s = seed;
    let mut buf = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = (s & 0xFFFF) as f32 / 65535.0;
            let gradient = ((x + y) % 64) as f32 / 64.0;
            buf.push(0.5 * noise + 0.5 * gradient);
        }
    }
    Img::new(buf, width, height)
}

fn read_dump(dir: &Path, kind: &str, scale: u32, chan: u32, run: u64) -> Option<Dump> {
    Dump::read(dir.join(format!("{kind}.s{scale}.c{chan:#06x}.run{run}.bin"))).ok()
}

/// CPU pooling, replicating dssim.rs:299-302 for one scale.
fn cpu_pool(map: &[f32], width: usize, height: usize, scale_n: usize) -> f64 {
    let sum = map.iter().fold(0.0f64, |s, i| s + f64::from(*i));
    let len = (width * height) as f64;
    let avg = (sum / len).max(0.0).powf((0.5f64).powf(scale_n as f64));
    1.0 - map.iter().fold(0.0f64, |s, i| s + (avg - f64::from(*i)).abs()) / len
}

/// All scenarios: (label, ref-run, mod-run, cmp-run, num_channels).
/// Runs are sequential: each create_image or compare bumps the sink's run
/// counter (dumps.rs next_run).
fn generate_dump_scenarios(dir: &Path) -> Vec<(&'static str, u64, u64, u64, usize)> {
    let _guard = dumps::enable(dir).unwrap();
    let d = new();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let mut run = 0u64;
    let mut scenarios = Vec::new();

    let pair = |d: &dssim_core::Dssim,
                    reference: &imgref::ImgVec<dssim_core::RGBAPLU>,
                    modified: &imgref::ImgVec<dssim_core::RGBAPLU>,
                    run: &mut u64,
                    scenarios: &mut Vec<(&'static str, u64, u64, u64, usize)>| {
        *run += 1;
        let r = d.create_image(reference).unwrap();
        *run += 1;
        let m = d.create_image(modified).unwrap();
        *run += 1;
        d.compare(&r, m);
        scenarios.push(("", *run - 2, *run - 1, *run, 3));
    };

    pair(&d, &img1, &img2, &mut run, &mut scenarios); // full
    pair(&d, &img1, &img2, &mut run, &mut scenarios); // identity (same image twice)

    // Gray (1-channel) pair.
    run += 1;
    let g1 = d.create_image(&synth_gray(96, 80, 0x9E37_79B9_7F4A_7C15)).unwrap();
    run += 1;
    let g2 = d.create_image(&synth_gray(96, 80, 0x632B_E5AB_2A2D_4D8F)).unwrap();
    run += 1;
    d.compare(&g1, g2);
    scenarios.push(("gray", run - 2, run - 1, run, 1));

    pair(
        &d,
        &decode_rgba("../tests/alpha1.png"),
        &decode_rgba("../tests/alpha2.png"),
        &mut run,
        &mut scenarios,
    ); // alpha

    // Sub-image crops (strided inputs, 3ch).
    run += 1;
    let s1 = d.create_image(&img1.sub_image(2, 3, 44, 33)).unwrap();
    run += 1;
    let s2 = d.create_image(&img2.sub_image(17, 9, 44, 33)).unwrap();
    run += 1;
    d.compare(&s1, s2);
    scenarios.push(("sub44x33", run - 2, run - 1, run, 3));

    run += 1;
    let s1 = d.create_image(&img1.sub_image(22, 8, 61, 40)).unwrap();
    run += 1;
    let s2 = d.create_image(&img2.sub_image(22, 8, 61, 40)).unwrap();
    run += 1;
    d.compare(&s1, s2);
    scenarios.push(("sub61x40", run - 2, run - 1, run, 3));

    drop(_guard);
    scenarios
}

#[test]
fn gpu_ssim_matches_cpu_dumps() {
    let _ = env_logger::try_init();
    let _serial = gen_lock();

    let dir = std::env::temp_dir().join(format!("dssim-ssim-dumps-{}", std::process::id()));
    let scenarios = generate_dump_scenarios(&dir);

    // Give the scenarios real labels for report lines.
    let labels = ["full", "identity", "gray", "alpha", "sub44x33", "sub61x40"];

    let probe = Context::new().expect("Vulkan context");
    let candidates = probe.device_candidates.clone();
    drop(probe);

    for (dev_idx, dev_name) in candidates.iter().enumerate() {
        let context = Arc::new(Context::new_with_device(dev_idx).expect("pinned context"));
        let mut map_checks = 0usize;
        let mut max_map = 0.0f64;
        let mut max_score = 0.0f64;

        for (label, (name, ref_run, mod_run, cmp_run, channels)) in labels.iter().zip(scenarios.iter()) {
            let name = if name.is_empty() { label } else { name };
            let (ref_run, mod_run, cmp_run, channels) = (*ref_run, *mod_run, *cmp_run, *channels);
            // Scales: probe on a compare-side artifact. create_image yields
            // one more scale than compare() processes (scale_weights zip),
            // so chan_mu alone would over-run into missing cross/ssim dumps.
            let mut scale = 0u32;
            while read_dump(&dir, "ssim_map", scale, 0xFFFF_FFFF, cmp_run).is_some() {
                let (w, h) = {
                    let first = read_dump(&dir, "chan_mu", scale, 0, ref_run).unwrap();
                    (first.header.width as usize, first.header.height as usize)
                };
                let pixels = w * h;

                let mut mu = Vec::with_capacity(2 * channels * pixels);
                let mut sq = Vec::with_capacity(2 * channels * pixels);
                let mut cross = Vec::with_capacity(channels * pixels);
                for c in 0..channels as u32 {
                    mu.extend_from_slice(&read_dump(&dir, "chan_mu", scale, c, ref_run).unwrap().data);
                    sq.extend_from_slice(&read_dump(&dir, "chan_sq_blur", scale, c, ref_run).unwrap().data);
                    cross.extend_from_slice(&read_dump(&dir, "cross_blur", scale, c, cmp_run).unwrap().data);
                }
                for c in 0..channels as u32 {
                    mu.extend_from_slice(&read_dump(&dir, "chan_mu", scale, c, mod_run).unwrap().data);
                    sq.extend_from_slice(&read_dump(&dir, "chan_sq_blur", scale, c, mod_run).unwrap().data);
                }

                let gpu_map = ssim_combine_gpu(&context, &mu, &sq, &cross, w, h, channels)
                    .unwrap_or_else(|e| panic!("{name} s{scale}{dev_name}: combine failed: {e}"));

                let cpu_map = read_dump(&dir, "ssim_map", scale, 0xFFFF_FFFF, cmp_run).unwrap();
                for (i, (c, g)) in cpu_map.data.iter().zip(gpu_map.iter()).enumerate() {
                    let diff = (f64::from(*c) - f64::from(*g)).abs();
                    if diff > max_map {
                        max_map = diff;
                    }
                    assert!(
                        diff <= MAP_TOL,
                        "{name} s{scale} [{dev_name}]: map pixel {i} (x={}, y={}) diverged: cpu={} gpu={} ({diff:.3e} > {MAP_TOL:.0e})",
                        i % w,
                        i / w,
                        c,
                        g
                    );
                }

                let score_dump = read_dump(&dir, "score_ssim", scale, 0xFFFF_FFFF, cmp_run).unwrap();
                let pooled = cpu_pool(&gpu_map, w, h, scale as usize);
                let score_diff = (pooled - f64::from(score_dump.data[0])).abs();
                if score_diff > max_score {
                    max_score = score_diff;
                }
                assert!(
                    score_diff <= SCORE_TOL,
                    "{name} s{scale} [{dev_name}]: pooled score {pooled} vs cpu {} ({score_diff:.3e} > {SCORE_TOL:.0e})",
                    score_dump.data[0]
                );

                map_checks += 1;
                scale += 1;
            }
        }

        eprintln!(
            "device {dev_idx} ({dev_name}): {map_checks} scale-maps checked, max_map={max_map:.3e}, max_score={max_score:.3e}"
        );
        assert!(map_checks >= 20, "expected 20+ scale-maps across scenarios, got {map_checks}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn gpu_ssim_identity_is_exactly_one() {
    let _ = env_logger::try_init();
    let _serial = gen_lock();

    let dir = std::env::temp_dir().join(format!("dssim-ssim-identity-{}", std::process::id()));
    // Identity scenario: same image twice, map must be exactly 1.0.
    let _guard = dumps::enable(&dir).unwrap();
    let d = new();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let a = d.create_image(&img1).unwrap();
    let b = d.create_image(&img1).unwrap();
    d.compare(&a, b);
    drop(_guard);

    let probe = Context::new().expect("Vulkan context");
    let candidates = probe.device_candidates.clone();
    drop(probe);

    for dev_idx in 0..candidates.len() {
        let context = Arc::new(Context::new_with_device(dev_idx).expect("pinned context"));
        let mut scale = 0u32;
        // Probe on ssim_map: compare() processes one scale fewer than
        // create_image generates (scale_weights zip), so scale 5 has no
        // cross/ssim dumps.
        while read_dump(&dir, "ssim_map", scale, 0xFFFF_FFFF, 3).is_some() {
            let mu0 = read_dump(&dir, "chan_mu", scale, 0, 1).unwrap();
            let (w, h) = (mu0.header.width as usize, mu0.header.height as usize);
            let pixels = w * h;
            let mut mu = Vec::with_capacity(2 * 3 * pixels);
            let mut sq = Vec::with_capacity(2 * 3 * pixels);
            let mut cross = Vec::with_capacity(3 * pixels);
            for c in 0..3u32 {
                mu.extend_from_slice(&read_dump(&dir, "chan_mu", scale, c, 1).unwrap().data);
                sq.extend_from_slice(&read_dump(&dir, "chan_sq_blur", scale, c, 1).unwrap().data);
                cross.extend_from_slice(&read_dump(&dir, "cross_blur", scale, c, 3).unwrap().data);
            }
            for c in 0..3u32 {
                mu.extend_from_slice(&read_dump(&dir, "chan_mu", scale, c, 2).unwrap().data);
                sq.extend_from_slice(&read_dump(&dir, "chan_sq_blur", scale, c, 2).unwrap().data);
            }

            let gpu_map = ssim_combine_gpu(&context, &mu, &sq, &cross, w, h, 3).unwrap();
            for (i, v) in gpu_map.iter().enumerate() {
                assert_eq!(
                    *v, 1.0f32,
                    "device {dev_idx} s{scale}: identity map pixel {i} (x={}, y={}) = {v}, must be exactly 1.0",
                    i % w,
                    i / w
                );
            }
            scale += 1;
        }
        eprintln!("device {dev_idx}: identity maps exactly 1.0 on all {scale} scales");
        // BH8: guard against a vacuous pass. The while loop above is gated on
        // read_dump(...).is_some() with hardcoded run numbers; if the dump
        // numbering shifts (an extra next_run, a new hook) zero iterations run
        // and the test passes having checked nothing. The sibling test floors
        // its count (map_checks >= 20); do the same here.
        assert!(
            scale > 0,
            "device {dev_idx}: identity test checked 0 scales -- dump numbering \
             shifted and the test is now vacuous"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}
