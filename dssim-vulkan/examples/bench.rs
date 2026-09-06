//! Phase H / M9: measured CPU-vs-GPU timing baseline (plan §4 Phase 7:
//! "Performance is measured separately from correctness... before/after
//! benchmark table"). Run with:
//!
//! ```text
//! cargo run -p dssim-vulkan --example bench --release
//! ```
//!
//! Times the full score path — `create_image` (pyramid build) + `compare` —
//! for both CPU and GPU, excluding only the synthetic image generation
//! (decode cost is identical on both paths in real use). Serial, discrete
//! GPU, warmup + a handful of timed iterations; no stress loops (TDR policy).

use dssim_core::Dssim;
use dssim_vulkan::GpuSsim;
use imgref::{Img, ImgVec};
use std::sync::Arc;
use std::time::Instant;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn noise_pair(w: usize, h: usize, seed: u64) -> (ImgVec<dssim_core::RGBAPLU>, ImgVec<dssim_core::RGBAPLU>) {
    let mk = |seed: u64| {
        let mut s = seed;
        let buf: Vec<dssim_core::RGBAPLU> = (0..w * h)
            .map(|_| {
                let r = (xorshift(&mut s) & 0xFF) as f32 / 255.0;
                let g = (xorshift(&mut s) & 0xFF) as f32 / 255.0;
                let b = (xorshift(&mut s) & 0xFF) as f32 / 255.0;
                dssim_core::RGBAPLU::new(r, g, b, 1.0)
            })
            .collect();
        Img::new(buf, w, h)
    };
    (mk(seed), mk(seed ^ 0xDEAD_BEEF))
}

fn time_ms<T, F: FnMut() -> T>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let _ = env_logger::try_init();
    let context = Arc::new(dssim_vulkan::Context::new().expect("Vulkan context"));
    eprintln!("device: {}", context.device_name());
    assert!(
        context.device_type() == ash::vk::PhysicalDeviceType::DISCRETE_GPU,
        "bench targets the discrete GPU only (TDR caution); got {:?}",
        context.device_type()
    );

    let sizes: &[(usize, usize)] = &[(320, 200), (1024, 1024), (2048, 2048)];
    let iters = 7;
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10}",
        "image", "cpu_ms", "gpu_ms", "ratio", "dssim_check"
    );

    for &(w, h) in sizes {
        let (a, b) = noise_pair(w, h, 0x1234_5678 ^ w as u64);

        // CPU baseline: decode excluded; create_image x2 + compare.
        let attr = Dssim::new();
        let cpu = time_ms(
            || {
                let r = attr.create_image(&a).unwrap();
                let m = attr.create_image(&b).unwrap();
                let (score, _) = attr.compare(&r, m);
                f64::from(score)
            },
            2,
            iters,
        );

        // GPU path, same shape.
        let gpu = GpuSsim::new(context.clone()).unwrap();
        let gpu_score = {
            let r = gpu.create_image(&a).unwrap();
            let m = gpu.create_image(&b).unwrap();
            gpu.compare(&r, &m).unwrap()
        };
        let gpu_ms = time_ms(
            || {
                let r = gpu.create_image(&a).unwrap();
                let m = gpu.create_image(&b).unwrap();
                let _ = gpu.compare(&r, &m).unwrap();
            },
            2,
            iters,
        );

        // Sanity: the GPU score is a real DSSIM value in a plausible range.
        assert!(gpu_score >= 0.0 && gpu_score < 1.0, "implausible score {gpu_score}");

        println!(
            "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10.6}",
            format!("{w}x{h}"),
            cpu,
            gpu_ms,
            gpu_ms / cpu,
            gpu_score
        );
    }

    eprintln!("\nNote: GPU path is now the optimized Phase-H shape — create_image");
    eprintln!("uploads each pyramid once, compare runs one batched submit over");
    eprintln!("GPU-resident planes (only the tiny SSIM maps come back to CPU).");
    eprintln!("Ratio < 1.0 means GPU beats CPU; small images still pay fixed");
    eprintln!("submit/setup overhead, so the win grows with image size.");
}
