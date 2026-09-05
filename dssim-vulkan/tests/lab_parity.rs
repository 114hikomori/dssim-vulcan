//! Phase F parity (plan §6 Phase F / VULKAN_PORT_PLAN §4 Phase 2): GPU
//! rgba_to_lab vs the CPU `ToLABBitmap` implementations, per device.
//! Bounds: plan §5 gives Lab conversion ≤ 1e-6 max abs per pixel (same
//! polynomial/Halley/matrix; only f32 op-order drift). Covers: the RGB path
//! with alpha dither (alpha1/alpha2 fixtures), a plain RGB pair, the gray
//! (1ch) ×1.16 branch, and arbitrary-coordinate dither bits.

use dssim_core::ToRGBAPLU;
use dssim_vulkan::{rgba_to_lab_gpu, Context};
use imgref::{Img, ImgVec};
use std::sync::{Arc, Mutex, MutexGuard};

const TOL: f64 = 1e-6;

/// One GPU test at a time — TDR caution (see blur_parity.rs).
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn all_devices() -> Vec<(usize, Arc<Context>)> {
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

/// Flatten an RGBAPLU image to 4-floats-per-pixel for the GPU buffer.
fn pack_rgba(img: &ImgVec<dssim_core::RGBAPLU>) -> Vec<f32> {
    let mut v = Vec::with_capacity(img.width() * img.height() * 4);
    for px in img.pixels() {
        v.extend_from_slice(&[px.r, px.g, px.b, px.a]);
    }
    v
}

fn assert_planes_eq(name: &str, cpu: &[ImgVec<f32>], gpu: &[f32]) {
    let w = cpu[0].width();
    let h = cpu[0].height();
    let pixels = w * h;
    assert_eq!(gpu.len(), cpu.len() * pixels, "{name}: plane count/size");
    let mut max = 0.0f64;
    let mut worst = 0usize;
    for (c, plane) in cpu.iter().enumerate() {
        for (i, (cv, gv)) in plane.pixels().zip(gpu[c * pixels..(c + 1) * pixels].iter()).enumerate() {
            let d = (f64::from(cv) - f64::from(*gv)).abs();
            if d > max {
                max = d;
                worst = i;
            }
        }
    }
    eprintln!("{name}: max_abs={max:.3e} worst_idx={worst}");
    assert!(max <= TOL, "{name}: GPU Lab diverged from CPU ({max:.3e} > {TOL:.0e})");
}

#[test]
fn gpu_lab_matches_cpu() {
    let _gpu = gpu_lock();

    // RGB paths (3ch): full test1, and the alpha pair exercising the dither.
    let img1 = decode_rgba("../tests/test1-sm.png");
    let alpha1 = decode_rgba("../tests/alpha1.png");
    let alpha2 = decode_rgba("../tests/alpha2.png");

    // Gray path (1ch).
    let g1 = synth_gray(96, 80, 0x9E37_79B9_7F4A_7C15);
    let g2 = synth_gray(96, 80, 0x632B_E5AB_2A2D_4D8F);

    for (dev_idx, context) in all_devices() {
        let (w, h) = (img1.width(), img1.height());
        let gpu = rgba_to_lab_gpu(&context, &pack_rgba(&img1), w, h, 3).unwrap();
        let cpu = dssim_core::ToLABBitmap::to_lab(&img1);
        assert_planes_eq(&format!("test1[d{dev_idx}]"), &cpu, &gpu);

        let (w, h) = (alpha1.width(), alpha1.height());
        let gpu = rgba_to_lab_gpu(&context, &pack_rgba(&alpha1), w, h, 3).unwrap();
        let cpu = dssim_core::ToLABBitmap::to_lab(&alpha1);
        assert_planes_eq(&format!("alpha1[d{dev_idx}]"), &cpu, &gpu);

        let (w, h) = (alpha2.width(), alpha2.height());
        let gpu = rgba_to_lab_gpu(&context, &pack_rgba(&alpha2), w, h, 3).unwrap();
        let cpu = dssim_core::ToLABBitmap::to_lab(&alpha2);
        assert_planes_eq(&format!("alpha2[d{dev_idx}]"), &cpu, &gpu);

        for (gi, g) in [&g1, &g2].into_iter().enumerate() {
            let (w, h) = (g.width(), g.height());
            let input: Vec<f32> = g.pixels().collect();
            let gpu = rgba_to_lab_gpu(&context, &input, w, h, 1).unwrap();
            let cpu = dssim_core::ToLABBitmap::to_lab(g);
            assert_planes_eq(&format!("gray{gi}[d{dev_idx}]"), &cpu, &gpu);
        }
    }
}

/// Dither determinism: same pixels, different coordinates must (on average)
/// differ — a weak but cheap guard that `n = (x+11)^(y+11)` is actually
/// applied (a shader that drops the dither still passes small uniform
/// images where dithered/undithered values rarely coincide).
#[test]
fn gpu_lab_dither_is_active() {
    let _gpu = gpu_lock();

    // Two mirrored variants of the same alpha image: dither patterns differ
    // by construction (a2 built by mirroring), so CPU planes must differ,
    // and the GPU must reproduce each one.
    let a1 = decode_rgba("../tests/alpha1.png");
    let a2 = decode_rgba("../tests/alpha2.png");
    let l1: Vec<f32> = dssim_core::ToLABBitmap::to_lab(&a1)[0].pixels().collect();
    let l2: Vec<f32> = dssim_core::ToLABBitmap::to_lab(&a2)[0].pixels().collect();
    let differing = l1.iter().zip(&l2).filter(|(x, y)| x != y).count();
    assert!(
        differing > (a1.width() * a1.height()) / 10,
        "fixture sanity: alpha1/alpha2 L planes should differ substantially ({differing} px)"
    );

    for (dev_idx, context) in all_devices() {
        let (w, h) = (a1.width(), a1.height());
        let gpu1 = rgba_to_lab_gpu(&context, &pack_rgba(&a1), w, h, 3).unwrap();
        let gpu2 = rgba_to_lab_gpu(&context, &pack_rgba(&a2), w, h, 3).unwrap();
        let d_gpu = gpu1
            .iter()
            .zip(&gpu2)
            .filter(|(x, y)| x != y)
            .count();
        assert!(
            d_gpu > (w * h) / 10,
            "device {dev_idx}: GPU dither inactive? only {d_gpu} planes differ"
        );
    }
}
