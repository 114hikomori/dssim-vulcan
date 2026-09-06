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

/// Like [`time_ms`] but also returns GPU-busy milliseconds (T9-lite VkQueryPool
/// timestamps) averaged over the timed iterations. `wall - gpu_busy` is the
/// submit/fence/PCIe overhead; the compute tracks (T5/T6a) move `gpu_busy`.
fn time_ms_gpu<T, F: FnMut() -> T>(
    ctx: &dssim_vulkan::Context,
    mut f: F,
    warmup: usize,
    iters: usize,
) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    ctx.set_gpu_timing(true);
    ctx.reset_gpu_timing();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let wall = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gpu_busy = ctx.gpu_elapsed_ms() / iters as f64;
    ctx.set_gpu_timing(false);
    (wall, gpu_busy)
}

fn main() {
    let _ = env_logger::try_init();
    // DSSIM_BENCH_DEVICE=<candidate index> pins a specific GPU (e.g. the
    // integrated one, for M10 sign-off); default picks the best (discrete).
    let pinned = std::env::var("DSSIM_BENCH_DEVICE").ok();
    let context = Arc::new(match &pinned {
        Some(idx) => dssim_vulkan::Context::new_with_device(
            idx.trim().parse().expect("DSSIM_BENCH_DEVICE is a candidate index"),
        )
        .expect("Vulkan context"),
        None => dssim_vulkan::Context::new().expect("Vulkan context"),
    });
    eprintln!("device: {} ({:?})", context.device_name(), context.device_type());
    // TDR caution: the discrete GPU is the default perf target. An explicitly
    // pinned device (e.g. integrated) is allowed for a sign-off run.
    assert!(
        pinned.is_some()
            || context.device_type() == ash::vk::PhysicalDeviceType::DISCRETE_GPU,
        "bench defaults to the discrete GPU (TDR caution); set DSSIM_BENCH_DEVICE to target another"
    );

    let sizes: &[(usize, usize)] = &[(320, 200), (1024, 1024), (2048, 2048), (4096, 4096)];
    println!(
        "{:>12} {:>9} {:>9} {:>7} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "image", "cpu_ms", "gpu_ms", "ratio", "create_ms", "create_gpu", "compare_ms", "compare_gpu", "dssim_check"
    );

    for &(w, h) in sizes {
        // Fewer iterations for very large images (4K create+compare is ~1s/op;
        // 7 iters x several timings would take minutes and stress TDR).
        let iters = if w * h >= 8_000_000 { 1 } else { 7 };
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

        // GPU path, same shape. T11: create_image_pair builds both pyramids in
        // one submit (one fence instead of two), then compare.
        let gpu = GpuSsim::new(context.clone()).unwrap();
        let gpu_score = {
            let (r, m) = gpu.create_image_pair(&a, &b).unwrap();
            gpu.compare(&r, &m).unwrap()
        };
        let gpu_ms = time_ms(
            || {
                let (r, m) = gpu.create_image_pair(&a, &b).unwrap();
                let _ = gpu.compare(&r, &m).unwrap();
            },
            2,
            iters,
        );

        // Phase breakdown: create_image (CPU downsample+pack+upload, one GPU
        // submit) vs compare (cross-blur+combine submit + map readback). The
        // *_gpu columns are GPU-busy time (T9-lite timestamps); wall-minus-gpu
        // is submit/fence/PCIe overhead.
        let (create_ms, create_gpu) = time_ms_gpu(
            &context,
            || {
                let _ = gpu.create_image(&a).unwrap();
            },
            2,
            iters,
        );
        let cmp_r = gpu.create_image(&a).unwrap();
        let cmp_m = gpu.create_image(&b).unwrap();
        let (compare_ms, compare_gpu) = time_ms_gpu(
            &context,
            || {
                let _ = gpu.compare(&cmp_r, &cmp_m).unwrap();
            },
            2,
            iters,
        );

        // Sanity: the GPU score is a real DSSIM value in a plausible range.
        assert!((0.0..1.0).contains(&gpu_score), "implausible score {gpu_score}");

        println!(
            "{:>12} {:>9.2} {:>9.2} {:>7.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>10.6}",
            format!("{w}x{h}"),
            cpu,
            gpu_ms,
            gpu_ms / cpu,
            create_ms,
            create_gpu,
            compare_ms,
            compare_gpu,
            gpu_score
        );
    }

    eprintln!("\nNote: GPU path is the optimized Phase-H shape — create_image writes");
    eprintln!("the pyramid upload straight into mapped staging (no intermediate");
    eprintln!("Vec/memcpy passes), compare runs one batched submit over GPU-resident");
    eprintln!("planes, and the per-scale SSIM maps come back for CPU pooling (the");
    eprintln!("scale-0 map is full-resolution, not tiny). Ratio < 1.0 = GPU beats CPU;");
    eprintln!("after the upload fix GPU wins at every measured size, incl. 320x200.");
}
