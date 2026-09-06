//! Phase E end-to-end parity (plan §6 Phase E / VULKAN_PORT_PLAN §4 Phase 5):
//! the full multi-scale pipeline — CPU Lab + downsample, GPU statistics +
//! SSIM, CPU f64 pooling — against the repository's locked values and the
//! live CPU path.
//!
//! Locked (dssim.rs ssim_locked_values, tolerance 5e-6 per plan §5):
//! test1 vs test2            = 0.0009483923725199794
//! sub [2,3,44x33]/[17,9,..] = 0.10810340934514495
//! sub [22,8,61x40] aligned  = 0.001675780079775091
//! Identity is locked to exactly 0.0 (mathematical fact).

use dssim_core::{new as cpu_new, Downsample, ToLABBitmap, ToRGBAPLU};
use dssim_vulkan::{GpuSsim, Context};
use imgref::{Img, ImgVec};
use std::sync::{Arc, Mutex, MutexGuard};

const TOL: f64 = 5e-6;

/// One GPU test at a time — driver-timeout (TDR) caution, see blur_parity.rs.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn decode_rgba(path: &str) -> ImgVec<dssim_core::RGBAPLU> {
    let file = lodepng::decode32_file(path).unwrap();
    Img::new(file.buffer.to_rgbaplu(), file.width, file.height)
}

fn synth_gray(width: usize, height: usize, seed: u64) -> ImgVec<f32> {
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

/// Synthetic RGBAPLU pair (distinct) for the RGB create path.
fn synth_rgba(width: usize, height: usize, seed: u64) -> ImgVec<dssim_core::RGBAPLU> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s & 0xFF) as f32 / 255.0
    };
    let buf = (0..width * height)
        .map(|_| dssim_core::RGBAPLU::new(next(), next(), next(), 1.0))
        .collect();
    Img::new(buf, width, height)
}

fn all_devices() -> Vec<(usize, Arc<Context>)> {
    let _ = env_logger::try_init();
    let probe = Context::new().expect("Vulkan context");
    let candidates = probe.device_candidates.clone();
    drop(probe);
    (0..candidates.len())
        .map(|i| {
            let ctx = Arc::new(Context::new_with_device(i).expect("pinned context"));
            eprintln!("device {i}: {}", ctx.device_name());
            (i, ctx)
        })
        .collect()
}

/// CPU reference score via the unmodified dssim-core path.
fn cpu_score<B>(reference: &B, modified: &B) -> f64
where
    B: ToLABBitmap + Downsample<Output = B> + Send + Sync + Clone,
{
    let d = cpu_new();
    let r = d.create_image(reference).unwrap();
    let m = d.create_image(modified).unwrap();
    f64::from(d.compare(&r, m).0)
}

fn assert_parity(name: &str, gpu_score: f64, cpu: f64) -> f64 {
    let diff = (gpu_score - cpu).abs();
    assert!(
        diff <= TOL,
        "{name}: GPU dssim {gpu_score} vs CPU {cpu} ({diff:.3e} > {TOL:.0e})"
    );
    eprintln!("{name}: gpu={gpu_score:.16} cpu={cpu:.16} diff={diff:.3e}");
    gpu_score
}

#[test]
fn phase_e_locked_values() {
    let _ = env_logger::try_init();
    let _gpu = gpu_lock();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");

    for (dev_idx, context) in all_devices() {
        let gpu = GpuSsim::new(context).unwrap();

        // 1. Headline locked value.
        let r = gpu.create_image(&img1).unwrap();
        let m = gpu.create_image(&img2).unwrap();
        let got = gpu.compare(&r, &m).unwrap();
        let diff = (got - 0.0009483923725199794).abs();
        assert!(diff <= TOL, "device {dev_idx}: full got {got}, locked 0.0009483923725199794 (diff {diff:.3e})");
        eprintln!("device {dev_idx} full: {got:.16} (diff {diff:.3e})");

        // 2. Identity: exactly 0.0, asserted with == (mathematical fact).
        let r2 = gpu.create_image(&img1).unwrap();
        let m2 = gpu.create_image(&img1).unwrap();
        let got = gpu.compare(&r2, &m2).unwrap();
        assert_eq!(got, 0.0, "device {dev_idx}: identity must be exactly 0.0, got {got}");

        // 3. Sub-image scenarios (strided views materialized tight — pixel
        //    equivalence argued in tests: to_lab/blur read the same pixels).
        let s1 = img1.sub_image(2, 3, 44, 33);
        let s2 = img2.sub_image(17, 9, 44, 33);
        let tight1: ImgVec<dssim_core::RGBAPLU> = Img::new(s1.pixels().collect(), 44, 33);
        let tight2: ImgVec<dssim_core::RGBAPLU> = Img::new(s2.pixels().collect(), 44, 33);
        let r3 = gpu.create_image(&tight1).unwrap();
        let m3 = gpu.create_image(&tight2).unwrap();
        let got = gpu.compare(&r3, &m3).unwrap();
        let diff = (got - 0.10810340934514495).abs();
        assert!(diff <= TOL, "device {dev_idx}: sub44x33 got {got}, locked 0.10810340934514495 (diff {diff:.3e})");

        let s1 = img1.sub_image(22, 8, 61, 40);
        let s2 = img2.sub_image(22, 8, 61, 40);
        let tight1: ImgVec<dssim_core::RGBAPLU> = Img::new(s1.pixels().collect(), 61, 40);
        let tight2: ImgVec<dssim_core::RGBAPLU> = Img::new(s2.pixels().collect(), 61, 40);
        let r4 = gpu.create_image(&tight1).unwrap();
        let m4 = gpu.create_image(&tight2).unwrap();
        let got = gpu.compare(&r4, &m4).unwrap();
        let diff = (got - 0.001675780079775091).abs();
        assert!(diff <= TOL, "device {dev_idx}: sub61x40 got {got}, locked 0.001675780079775091 (diff {diff:.3e})");

        eprintln!("device {dev_idx}: locked values OK");
    }
}

/// M7 close-out: the full pipeline on small and odd-size images —
/// below/around the 8-px scale cutoff, exercising the scale-count edge.
#[test]
fn phase_e_small_and_odd_sizes() {
    let _gpu = gpu_lock();

    for (w, h) in [(16usize, 16usize), (17usize, 9usize), (9usize, 7usize), (1usize, 1usize)] {
        let a = synth_gray(w, h, 0x5EED_0000 + (w * 31 + h) as u64);
        let b = synth_gray(w, h, 0xC0FF_EE00 + (w * 17 + h) as u64);

        for (dev_idx, context) in all_devices() {
            let gpu = GpuSsim::new(context).unwrap();
            let r = gpu.create_image_gray(&a).unwrap();
            let m = gpu.create_image_gray(&b).unwrap();
            let got = gpu.compare(&r, &m).unwrap();
            let cpu = cpu_score(&a, &b);
            let diff = (got - cpu).abs();
            eprintln!("gray {w}x{h} [d{dev_idx}]: gpu={got:.12} cpu={cpu:.12} diff={diff:.3e}");
            assert!(
                diff <= TOL,
                "{w}x{h} [d{dev_idx}]: GPU dssim {got} vs CPU {cpu} ({diff:.3e} > {TOL:.0e})"
            );
        }
    }
}

/// F28: exercise the per-scale flush path (normally only reached at or above
/// SPLIT_SUBMIT_MIN_PIXELS, far too slow for lavapipe CI) by forcing the
/// threshold to 0 on small images, and prove split == batch == CPU for both
/// the RGB and gray create paths plus compare. The passes are identical in
/// both modes -- only submission boundaries differ -- so split and batch must
/// agree bit-for-bit.
#[test]
fn phase_e_split_submit_matches_batch_and_cpu() {
    let _gpu = gpu_lock();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let g1 = synth_gray(96, 80, 0x1234_5678);
    let g2 = synth_gray(96, 80, 0x9ABC_DEF0);
    let cpu_rgb = cpu_score(&img1, &img2);
    let cpu_gray = cpu_score(&g1, &g2);

    for (dev_idx, context) in all_devices() {
        // --- RGB: batch (default) vs split (threshold 0) vs CPU ---
        let gpu_batch = GpuSsim::new(context.clone()).unwrap();
        let rb = gpu_batch.create_image(&img1).unwrap();
        let mb = gpu_batch.create_image(&img2).unwrap();
        let batch = gpu_batch.compare(&rb, &mb).unwrap();

        let mut gpu_split = GpuSsim::new(context.clone()).unwrap();
        gpu_split.set_split_threshold_for_test(0);
        let rs = gpu_split.create_image(&img1).unwrap();
        let ms = gpu_split.create_image(&img2).unwrap();
        let split = gpu_split.compare(&rs, &ms).unwrap();

        assert_parity("rgb split-vs-cpu", split, cpu_rgb);
        assert_parity("rgb batch-vs-cpu", batch, cpu_rgb);
        assert_eq!(batch, split, "device {dev_idx}: rgb split {split} != batch {batch}");

        // --- Gray: split (threshold 0) vs CPU (exercises create_image_gray flush) ---
        let mut gpu_gsplit = GpuSsim::new(context).unwrap();
        gpu_gsplit.set_split_threshold_for_test(0);
        let rg = gpu_gsplit.create_image_gray(&g1).unwrap();
        let mg = gpu_gsplit.create_image_gray(&g2).unwrap();
        assert_parity("gray split-vs-cpu", gpu_gsplit.compare(&rg, &mg).unwrap(), cpu_gray);
        let _ = dev_idx;
    }
}

#[test]
fn phase_e_cpu_reference_parity() {
    let _gpu = gpu_lock();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let alpha1 = decode_rgba("../tests/alpha1.png");
    let alpha2 = decode_rgba("../tests/alpha2.png");
    let g1 = synth_gray(96, 80, 0x9E37_79B9_7F4A_7C15);
    let g2 = synth_gray(96, 80, 0x632B_E5AB_2A2D_4D8F);

    for (dev_idx, context) in all_devices() {
        let gpu = GpuSsim::new(context).unwrap();

        let r = gpu.create_image(&img1).unwrap();
        let m = gpu.create_image(&img2).unwrap();
        assert_parity("full[recheck]", gpu.compare(&r, &m).unwrap(), cpu_score(&img1, &img2));

        let r = gpu.create_image(&alpha1).unwrap();
        let m = gpu.create_image(&alpha2).unwrap();
        assert_parity("alpha", gpu.compare(&r, &m).unwrap(), cpu_score(&alpha1, &alpha2));

        // Gray pipeline: GPU Lab (1ch ×1.16 branch) + 1ch combine.
        let r = gpu.create_image_gray(&g1).unwrap();
        let m = gpu.create_image_gray(&g2).unwrap();
        assert_parity("gray(1ch)", gpu.compare(&r, &m).unwrap(), cpu_score(&g1, &g2));
        let _ = dev_idx;
    }
}


/// F28: MEASURE (not just infer) that the adaptive split-submit path lowers
/// peak allocation bytes vs the batched path, via the Context allocator
/// live/peak counters. Forces each mode with the test seam on a 512^2 image
/// (large enough that the pyramid-transient excess is material, small enough to
/// stay fast on lavapipe CI).
#[test]
fn phase_e_split_submit_lowers_peak_vram() {
    let _gpu = gpu_lock();
    let src = synth_rgba(512, 512, 0xF28F_28F2_8F28_F28F);

    for (dev_idx, context) in all_devices() {
        let mut gpu = GpuSsim::new(context.clone()).unwrap();

        // Batched: threshold above the image => one submit, every scale's
        // transients live simultaneously.
        gpu.set_split_threshold_for_test(usize::MAX);
        context.reset_alloc_peak();
        let batched = gpu.create_image(&src).unwrap();
        let peak_batched = context.peak_alloc_bytes();
        drop(batched);

        // Split: threshold 0 => flush per scale, only one scale's transients
        // live at a time (persistent outputs still accumulate).
        gpu.set_split_threshold_for_test(0);
        context.reset_alloc_peak();
        let split = gpu.create_image(&src).unwrap();
        let peak_split = context.peak_alloc_bytes();
        drop(split);

        let saved = peak_batched.saturating_sub(peak_split);
        eprintln!(
            "device {dev_idx}: peak batched={peak_batched} B, split={peak_split} B, saved={saved} B"
        );
        assert!(
            peak_split < peak_batched,
            "device {dev_idx}: split peak ({peak_split}) should be below batched peak ({peak_batched})"
        );
        assert!(
            saved > 1_000_000,
            "device {dev_idx}: expected >1 MB saved, got {saved} B"
        );
    }
}


/// T11: create_image_pair (both pyramids in one submit) must produce the exact
/// same comparison result as two separate create_image calls -- the passes are
/// identical, only grouped into one submit -- and match the CPU reference.
#[test]
fn phase_e_create_pair_matches_separate_and_cpu() {
    let _gpu = gpu_lock();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let cpu = cpu_score(&img1, &img2);

    for (dev_idx, context) in all_devices() {
        let gpu = GpuSsim::new(context).unwrap();

        let (r1, m1) = gpu.create_image_pair(&img1, &img2).unwrap();
        let paired = gpu.compare(&r1, &m1).unwrap();

        let r2 = gpu.create_image(&img1).unwrap();
        let m2 = gpu.create_image(&img2).unwrap();
        let separate = gpu.compare(&r2, &m2).unwrap();

        assert_eq!(
            paired, separate,
            "device {dev_idx}: pair {paired} != separate {separate}"
        );
        assert_parity("pair-vs-cpu", paired, cpu);
    }
}


/// BH16: the F29 channel-count assert in `compare` (score.rs) is a public-API
/// guard against GPU out-of-bounds (the cross-blur derives src2 stride from
/// src1). The CLI checks sizes upstream, so nothing else drives a mismatch
/// through `compare` -- a refactor deleting the assert would silently
/// reintroduce the hazard. Pin it with a should_panic.
#[test]
#[should_panic(expected = "channel count mismatch")]
fn phase_e_compare_panics_on_channel_mismatch() {
    let _gpu = gpu_lock();
    let context = Arc::new(Context::new().expect("context"));
    let gpu = GpuSsim::new(context).unwrap();
    let rgb = gpu.create_image(&synth_rgba(64, 64, 0x1111_2222)).unwrap();
    let gray = gpu.create_image_gray(&synth_gray(64, 64, 0x3333_4444)).unwrap();
    let _ = gpu.compare(&rgb, &gray);
}

/// BH16: the F29 dimension assert in `compare` -- a ref/mod shape mismatch
/// reads out of bounds on the GPU (no CPU slice panic). Pin it.
#[test]
#[should_panic(expected = "matching dimensions")]
fn phase_e_compare_panics_on_dimension_mismatch() {
    let _gpu = gpu_lock();
    let context = Arc::new(Context::new().expect("context"));
    let gpu = GpuSsim::new(context).unwrap();
    let a = gpu.create_image(&synth_rgba(64, 64, 0x5555_6666)).unwrap();
    let b = gpu.create_image(&synth_rgba(64, 63, 0x7777_8888)).unwrap();
    let _ = gpu.compare(&a, &b);
}

/// BH34: create_image_pair in SPLIT mode (per-scale flush into the SHARED
/// passes Vec across BOTH images) must match batch pair and CPU. This is the
/// T11 x F28 intersection where BH3/BH18 hide -- previously untested.
#[test]
fn phase_e_create_pair_split_matches_batch_and_cpu() {
    let _gpu = gpu_lock();
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let cpu = cpu_score(&img1, &img2);

    for (dev_idx, context) in all_devices() {
        let mut gpu_split = GpuSsim::new(context.clone()).unwrap();
        gpu_split.set_split_threshold_for_test(0);
        let (rs, ms) = gpu_split.create_image_pair(&img1, &img2).unwrap();
        let split_pair = gpu_split.compare(&rs, &ms).unwrap();

        let gpu_batch = GpuSsim::new(context).unwrap();
        let (rb, mb) = gpu_batch.create_image_pair(&img1, &img2).unwrap();
        let batch_pair = gpu_batch.compare(&rb, &mb).unwrap();

        assert_eq!(
            split_pair, batch_pair,
            "device {dev_idx}: pair-split {split_pair} != pair-batch {batch_pair}"
        );
        assert_parity("pair-split-vs-cpu", split_pair, cpu);
    }
}


/// BH3: create_image_pair rejects a mismatched-size pair with Err (not a later
/// panic in compare), so the reference-only threshold can never build the
/// larger pyramid in batch mode.
#[test]
fn phase_e_create_pair_rejects_size_mismatch() {
    let _gpu = gpu_lock();
    let context = Arc::new(Context::new().expect("context"));
    let gpu = GpuSsim::new(context).unwrap();
    let big = synth_rgba(64, 64, 0xAAAA_BBBB);
    let small = synth_rgba(64, 63, 0xCCCC_DDDD);
    let err = gpu
        .create_image_pair(&big, &small)
        .err()
        .expect("expected Err for a mismatched-size pair");
    assert!(
        matches!(err, dssim_vulkan::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}
