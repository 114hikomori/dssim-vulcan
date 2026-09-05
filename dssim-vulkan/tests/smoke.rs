//! Phase B / M1 smoke test (VULKAN_PORT_PLAN.md §4 Phase 1): upload an f32
//! buffer, run the ×2 compute shader, download the result, and assert every
//! element is exactly doubled. Exit observation: passes on a real GPU (and
//! lavapipe in CI) with validation layers clean.

use std::sync::{Arc, Mutex, MutexGuard};

use dssim_vulkan::{run_smoke, Context};

/// One GPU test at a time — driver-timeout (TDR) caution, see blur_parity.rs.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn smoke_doubles_on_gpu() {
    let _gpu = gpu_lock();
    let _ = env_logger::try_init();

    let context = Arc::new(Context::new().expect("Vulkan context"));
    eprintln!(
        "device: {} ({:?}); candidates: {:?}",
        context.device_name(),
        context.device_type(),
        context.device_candidates
    );

    // Deliberate values, including extremes that must survive f32 exactly.
    let input: Vec<f32> = vec![
        0.0,
        1.0,
        -2.5,
        0.15625,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
    ];
    // Plus enough bulk to cross multiple workgroups (64 invocations each).
    let input: Vec<f32> = input
        .into_iter()
        .chain((0..5000).map(|i| (i % 251) as f32 / 32.0 - 4.0))
        .collect();

    let output = run_smoke(&context, &input).expect("smoke dispatch");
    assert_eq!(output.len(), input.len());

    for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(*b, a * 2.0, "element {i}: {a} doubled should be {}, got {b}", a * 2.0);
    }
}

#[test]
fn context_lists_devices() {
    let _gpu = gpu_lock();
    let context = Context::new().expect("Vulkan context");
    assert!(!context.device_candidates.is_empty());
    // Selection policy: the chosen device must be the best-ranked candidate.
    assert_eq!(context.device_name(), context.device_candidates[0]);
}

/// Run the smoke dispatch on every enumerated device (discrete and integrated
/// here) so both real-GPU drivers are exercised, not just the default pick.
#[test]
fn smoke_on_every_device() {
    let _gpu = gpu_lock();
    let _ = env_logger::try_init();

    let probe = Context::new().expect("Vulkan context");
    let candidates = probe.device_candidates.clone();
    drop(probe);

    for idx in 0..candidates.len() {
        let context = Arc::new(Context::new_with_device(idx).expect("pinned context"));
        eprintln!("candidate {idx}: {}", context.device_name());
        let input: Vec<f32> = (0..300).map(|i| (i as f32) / 7.0 - 20.0).collect();
        let output = run_smoke(&context, &input).expect("smoke dispatch");
        for (a, b) in input.iter().zip(&output) {
            assert_eq!(*b, a * 2.0, "device {}: wrong result", context.device_name());
        }
    }
}

