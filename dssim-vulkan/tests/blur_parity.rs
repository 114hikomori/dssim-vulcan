//! Phase C parity tests (VULKAN_PORT_PLAN.md §4 Phase 3): the Vulkan blur
//! against `dssim_core::blur` — the same fused 5-tap implementation the
//! CPU `blur/equiv_tests.rs` battery proves equivalent to upstream's
//! double-3×3 — over the same battery of patterns, including the tiny-size
//! sweep 1..=8 and the strided-subimage case. Bound: plan §5 gives blur
//! ≤ 2×10⁻⁶ max abs per pixel (CPU-equiv drift is ≤5.6×10⁻⁷).

use dssim_vulkan::{blur_gpu, blur_mul_gpu, Context};
use std::sync::Arc;

const TOL: f64 = 2e-6;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn random_image(w: usize, h: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..w * h)
        .map(|_| (xorshift32(&mut s) as f32) / (u32::MAX as f32))
        .collect()
}

fn linear_gradient(w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x + y) as f32 / (w + h) as f32))
        .collect()
}

fn step_edge(w: usize, h: usize) -> Vec<f32> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| if x < w / 2 || y < h / 2 { 0.0 } else { 1.0 }))
        .collect()
}

fn impulse(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h];
    buf[(h / 2) * w + (w / 2)] = 1.0;
    buf
}

/// CPU reference via dssim-core (one implementation, two callers).
fn cpu_blur(data: &[f32], w: usize, h: usize) -> Vec<f32> {
    let img = imgref::ImgVec::new(data.to_vec(), w, h);
    let mut tmp: Vec<std::mem::MaybeUninit<f32>> =
        (0..w * h).map(|_| std::mem::MaybeUninit::uninit()).collect();
    dssim_core::blur::blur(img.as_ref(), &mut tmp).buf().to_vec()
}

fn cpu_blur_strided(data: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let img = imgref::ImgVec::new_stride(data.to_vec(), w, h, stride);
    let mut tmp: Vec<std::mem::MaybeUninit<f32>> =
        (0..w * h).map(|_| std::mem::MaybeUninit::uninit()).collect();
    dssim_core::blur::blur(img.as_ref(), &mut tmp).buf().to_vec()
}

fn cpu_blur_mul(a: &[f32], b: &[f32], w: usize, h: usize) -> Vec<f32> {
    let img1 = imgref::ImgVec::new(a.to_vec(), w, h);
    let img2 = imgref::ImgVec::new(b.to_vec(), w, h);
    let mut tmp: Vec<std::mem::MaybeUninit<f32>> =
        (0..w * h).map(|_| std::mem::MaybeUninit::uninit()).collect();
    dssim_core::blur::blur_mul(img1.as_ref(), img2.as_ref(), &mut tmp)
}

/// Compare GPU vs CPU output; assert max abs diff ≤ TOL; report worst pixel.
fn assert_parity(name: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{name}: length mismatch");
    let mut max = 0.0f64;
    let mut worst = 0usize;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let d = (f64::from(*c) - f64::from(*g)).abs();
        if d > max {
            max = d;
            worst = i;
        }
    }
    eprintln!("{name}: max_abs={max:.3e} worst_idx={worst} cpu={} gpu={}", cpu[worst], gpu[worst]);
    assert!(max <= TOL, "{name}: GPU diverged from CPU ({max:.3e} > {TOL:.0e})");
}

/// One context per enumerated device: the parity battery must hold on every
/// real GPU, not just the best-ranked one (M2 audit fix).
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

#[test]
fn gpu_blur_constant() {
    for (idx, context) in all_devices() {
        let cpu_img = vec![0.5f32; 64 * 48];
        let gpu = blur_gpu(&context, &cpu_img, 64, 48, 64).unwrap();
        assert_parity(&format!("constant_50[d{idx}]"), &cpu_blur(&cpu_img, 64, 48), &gpu);
    }
}

#[test]
fn gpu_blur_linear_gradient() {
    for (idx, context) in all_devices() {
        let img = linear_gradient(96, 64);
        let gpu = blur_gpu(&context, &img, 96, 64, 96).unwrap();
        assert_parity(&format!("linear_gradient[d{idx}]"), &cpu_blur(&img, 96, 64), &gpu);
    }
}

#[test]
fn gpu_blur_random() {
    for (idx, context) in all_devices() {
        for &(w, h, seed) in &[
            (64usize, 64usize, 0xCAFEBABE_u32),
            (97, 53, 0x1234_5678),
            (128, 96, 0xDEAD_BEEF),
        ] {
            let img = random_image(w, h, seed);
            let gpu = blur_gpu(&context, &img, w, h, w).unwrap();
            assert_parity(&format!("random_{w}x{h}[d{idx}]"), &cpu_blur(&img, w, h), &gpu);
        }
    }
}

#[test]
fn gpu_blur_step_edge() {
    for (idx, context) in all_devices() {
        let img = step_edge(96, 96);
        let gpu = blur_gpu(&context, &img, 96, 96, 96).unwrap();
        assert_parity(&format!("step_edge[d{idx}]"), &cpu_blur(&img, 96, 96), &gpu);
    }
}

#[test]
fn gpu_blur_impulse() {
    for (idx, context) in all_devices() {
        for &(w, h) in &[(64usize, 64usize), (33, 37)] {
            let img = impulse(w, h);
            let gpu = blur_gpu(&context, &img, w, h, w).unwrap();
            assert_parity(&format!("impulse_{w}x{h}[d{idx}]"), &cpu_blur(&img, w, h), &gpu);
        }
    }
}

#[test]
fn gpu_blur_strided_subimage() {
    for (idx, context) in all_devices() {
        // 96×64 buffer with 96 stride, viewing inner 80×48 — the sub-image
        // shape from the CPU equiv battery. Strided input must match the CPU
        // blur of the same strided view.
        let full = random_image(96, 64, 0xA5A5_A5A5);
        let (w, h) = (80usize, 48usize);
        let mut strided = vec![0f32; 96 * h];
        for y in 0..h {
            for x in 0..w {
                strided[y * 96 + x] = full[(y + 4) * 96 + (x + 8)];
            }
        }
        let gpu = blur_gpu(&context, &strided, w, h, 96).unwrap();
        assert_parity(
            &format!("strided_subimage[d{idx}]"),
            &cpu_blur_strided(&strided, w, h, 96),
            &gpu,
        );
    }
}

#[test]
fn gpu_blur_tiny_sizes() {
    for (idx, context) in all_devices() {
        for w in 1..=8usize {
            for h in 1..=8usize {
                let img = random_image(w, h, 0xBEEF_F00Du32.wrapping_add((w * 99 + h) as u32));
                let gpu = blur_gpu(&context, &img, w, h, w).unwrap();
                assert_parity(&format!("tiny_{w}x{h}[d{idx}]"), &cpu_blur(&img, w, h), &gpu);
            }
        }
    }
}

#[test]
fn gpu_blur_mul_parity() {
    for (idx, context) in all_devices() {
        for &(w, h) in &[(64usize, 48usize), (5, 7), (2, 3), (1, 1)] {
            let a = random_image(w, h, 0x0DD_BA5E);
            let b = random_image(w, h, 0xF00D_5EED);
            let gpu = blur_mul_gpu(&context, &a, &b, w, h, w, w).unwrap();
            assert_parity(&format!("blur_mul_{w}x{h}[d{idx}]"), &cpu_blur_mul(&a, &b, w, h), &gpu);
        }
    }
}

#[test]
fn gpu_blur_mul_strided_parity() {
    for (idx, context) in all_devices() {
        let (w, h) = (37usize, 21usize);
        let stride1 = w + 5;
        let stride2 = w + 11;
        let a = random_image(stride1, h, 0x11_2233);
        let b = random_image(stride2, h, 0x44_5566);
        let gpu = blur_mul_gpu(&context, &a, &b, w, h, stride1, stride2).unwrap();

        let img1 = imgref::ImgVec::new_stride(a.clone(), w, h, stride1);
        let img2 = imgref::ImgVec::new_stride(b.clone(), w, h, stride2);
        let mut tmp: Vec<std::mem::MaybeUninit<f32>> =
            (0..w * h).map(|_| std::mem::MaybeUninit::uninit()).collect();
        let cpu = dssim_core::blur::blur_mul(img1.as_ref(), img2.as_ref(), &mut tmp);
        assert_parity(&format!("blur_mul_strided[d{idx}]"), &cpu, &gpu);
    }
}
