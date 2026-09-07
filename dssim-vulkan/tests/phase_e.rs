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
use dssim_vulkan::{GpuSsim, Context, PrepMode};
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


/// BH35: tiny RGB end-to-end (1x1, 7x7, 8x8) through create_image + compare.
/// Only blur-level tiny sweeps existed; the full pipeline at sub-scale-cutoff
/// sizes (pyramid collapses to 1 scale) was untested at the score level.
#[test]
fn phase_e_tiny_rgb_end_to_end() {
    let _gpu = gpu_lock();
    let devices = all_devices();
    for (w, h) in [(1usize, 1usize), (7, 7), (8, 8)] {
        let seed = (w * 131 + h * 17) as u64;
        let a = synth_rgba(w, h, 0x1234_5678 ^ seed);
        let b = synth_rgba(w, h, 0x9ABC_DEF0 ^ seed);
        let cpu = cpu_score(&a, &b);
        for (dev_idx, context) in &devices {
            let gpu = GpuSsim::new(context.clone()).unwrap();
            let r = gpu.create_image(&a).unwrap();
            let m = gpu.create_image(&b).unwrap();
            assert_parity(&format!("tiny {w}x{h} dev {dev_idx}"), gpu.compare(&r, &m).unwrap(), cpu);
        }
    }
}


/// BH5: prove the submit_lock makes concurrent `&self` use correct. Four threads
/// share one GpuSsim (Arc<Context> is Sync) and each runs create_image_pair +
/// compare on its own copy of the pair; every result must match the CPU
/// reference. Without the lock they would race on the shared command pool /
/// descriptor pools / perf query pool and corrupt.
#[test]
fn phase_e_concurrent_submits_are_serialized() {
    let _gpu = gpu_lock();
    let context = Arc::new(Context::new().expect("context"));
    let gpu = Arc::new(GpuSsim::new(context).unwrap());
    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");
    let cpu = cpu_score(&img1, &img2);

    let results: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let gpu = gpu.clone();
                let a = img1.clone();
                let b = img2.clone();
                s.spawn(move || {
                    let (r, m) = gpu.create_image_pair(&a, &b).unwrap();
                    gpu.compare(&r, &m).unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("submit thread")).collect()
    });
    assert_eq!(results.len(), 4);
    for (i, score) in results.iter().enumerate() {
        assert_parity(&format!("concurrent submit {i}"), *score, cpu);
    }
}


/// Tier 4: compare_many (resident reference vs N modifieds) must equal the
/// individual create+compare results and the CPU reference for each.
#[test]
fn phase_e_compare_many_matches_individual_and_cpu() {
    let _gpu = gpu_lock();
    let (w, h) = (96usize, 80usize);
    let orig = synth_rgba(w, h, 0x1111_2222);
    let mods: Vec<_> = (0..3u64).map(|i| synth_rgba(w, h, 0x3333_4444 ^ i)).collect();
    let cpus: Vec<f64> = mods.iter().map(|m| cpu_score(&orig, m)).collect();

    for (dev_idx, context) in all_devices() {
        let gpu = GpuSsim::new(context).unwrap();
        let r = gpu.create_image(&orig).unwrap();
        let many = gpu.compare_many(&r, &mods).unwrap();
        assert_eq!(many.len(), 3);
        for i in 0..3 {
            let one = {
                let m = gpu.create_image(&mods[i]).unwrap();
                gpu.compare(&r, &m).unwrap()
            };
            assert_eq!(
                many[i], one,
                "dev {dev_idx} mod {i}: many {} != individual {one}",
                many[i]
            );
            assert_parity(&format!("compare_many[{i}]"), many[i], cpus[i]);
        }
    }
}


/// Tier 5 Mode A: `PrepMode::Device` (GPU 2x2 downsample) must produce results
/// BITWISE equal to `PrepMode::Cpu` (CPU downsample) -- the opt-in is a drop-in,
/// not an approximation. Includes odd sizes to exercise the floor-drop of the
/// trailing row/col, and small sizes to exercise the w<8||h<8 stop.
#[test]
fn phase_e_device_prep_is_bitwise_equal_to_cpu_prep() {
    let _gpu = gpu_lock();
    let cases: Vec<(usize, usize)> =
        vec![(96, 80), (64, 64), (63, 63), (100, 37), (255, 255), (8, 8), (9, 9), (16, 15)];
    for (dev_idx, context) in all_devices() {
        let gpu_cpu = GpuSsim::new(context.clone()).unwrap();
        let gpu_dev = GpuSsim::with_prep_mode(context.clone(), PrepMode::Device).unwrap();
        assert_eq!(gpu_cpu.prep_mode(), PrepMode::Cpu);
        assert_eq!(gpu_dev.prep_mode(), PrepMode::Device);
        for &(w, h) in &cases {
            let seed = (w * 31 + h * 17) as u64;
            let a = synth_rgba(w, h, 0xABCD_0000 ^ seed);
            let b = synth_rgba(w, h, 0x1234_0000 ^ seed);
            let (ra, ma) = gpu_cpu.create_image_pair(&a, &b).unwrap();
            let cpu_prep = gpu_cpu.compare(&ra, &ma).unwrap();
            let (rd, md) = gpu_dev.create_image_pair(&a, &b).unwrap();
            let dev_prep = gpu_dev.compare(&rd, &md).unwrap();
            assert_eq!(
                dev_prep.to_bits(),
                cpu_prep.to_bits(),
                "dev {dev_idx} {w}x{h}: device prep {dev_prep} != cpu prep {cpu_prep} (not bitwise-equal)"
            );
            assert_parity(&format!("device-prep {w}x{h}"), dev_prep, cpu_score(&a, &b));
        }
    }
}

/// Tier 5: device prep must also honor the split-submit path (large images) and
/// stay bitwise-equal to CPU prep there. Forces split via the test seam.
#[test]
fn phase_e_device_prep_split_matches_cpu_prep() {
    let _gpu = gpu_lock();
    let a = synth_rgba(96, 80, 0x5A5A_0001);
    let b = synth_rgba(96, 80, 0x3C3C_0002);
    for (dev_idx, context) in all_devices() {
        let mut gpu_cpu = GpuSsim::new(context.clone()).unwrap();
        gpu_cpu.set_split_threshold_for_test(0);
        let (ra, ma) = gpu_cpu.create_image_pair(&a, &b).unwrap();
        let cpu_prep = gpu_cpu.compare(&ra, &ma).unwrap();

        let mut gpu_dev = GpuSsim::with_prep_mode(context, PrepMode::Device).unwrap();
        gpu_dev.set_split_threshold_for_test(0);
        let (rd, md) = gpu_dev.create_image_pair(&a, &b).unwrap();
        let dev_prep = gpu_dev.compare(&rd, &md).unwrap();

        assert_eq!(
            dev_prep.to_bits(),
            cpu_prep.to_bits(),
            "dev {dev_idx}: device-prep split {dev_prep} != cpu-prep split {cpu_prep}"
        );
    }
}


/// BH36: compare_many must reject a mismatched modified with Err BEFORE any GPU
/// dispatch -- not panic in compare() after wasting a full pyramid build for the
/// bad modified and discarding the scores already computed.
#[test]
fn phase_e_compare_many_rejects_size_mismatch() {
    let _gpu = gpu_lock();
    let context = Arc::new(Context::new().expect("context"));
    let gpu = GpuSsim::new(context).unwrap();
    let orig = synth_rgba(64, 64, 0x1111_2222);
    let r = gpu.create_image(&orig).unwrap();
    let good = synth_rgba(64, 64, 0x3333_4444);
    let bad = synth_rgba(64, 63, 0x5555_6666); // height differs
    let err = gpu
        .compare_many(&r, &[good, bad])
        .unwrap_err();
    assert!(
        matches!(err, dssim_vulkan::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}
