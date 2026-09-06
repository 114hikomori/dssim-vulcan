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
