//! Phase G CLI integration test: `--gpu` output format is identical to the
//! CPU path, and values agree within the plan's 5e-6 score bound.
//! (Bit-identical 8-decimal output is NOT expected — the GPU path has ~1e-7
//! FP drift; plan §6 Phase G requires identical *formatting* and P5's
//! ≤5e-6 headline tolerance.)
//!
//! F7: this test must not pass vacuously. If `--gpu` silently falls back to
//! CPU (no Vulkan device), comparing CPU-vs-CPU would "pass" while proving
//! nothing about the GPU path. So we (a) require a *positive* signal that the
//! GPU ran (the `dssim: gpu device:` line the CLI prints), and (b) distinguish
//! "no Vulkan on this machine" (skip, with a clear message) from "Vulkan is
//! available but `--gpu` did not engage" (a real regression -> fail).

#![cfg(feature = "gpu")]

use std::process::Command;
use std::sync::{Mutex, MutexGuard};

/// F8: the two tests below each spawn a `dssim --gpu` subprocess. Cargo runs
/// them on parallel harness threads within this one test binary, so without
/// serialization there are up to 2 concurrent GPU contexts — which the AMD
/// Windows driver can reject with INCOMPLETE (see CHECKPOINT M1). This
/// process-local mutex serializes them (one subprocess at a time). It cannot
/// span other test binaries; the suite is run with --test-threads=1 for that.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const TOL: f64 = 5e-6;

struct Cli {
    success: bool,
    stdout: Vec<String>,
    stderr: String,
}

fn run_cli(args: &[&str]) -> Cli {
    let out = Command::new(env!("CARGO_BIN_EXE_dssim"))
        .args(args)
        .output()
        .expect("dssim binary runs");
    Cli {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::to_owned)
            .collect(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Probe Vulkan availability in-process (same driver/ICD state the CLI
/// subprocess sees). Creating a `Context` is lightweight — instance + physical
/// device selection, no dispatch — and it is dropped immediately.
fn vulkan_available() -> bool {
    dssim_vulkan::Context::new().is_ok()
}

fn parse_scores(stdout: &[String]) -> Vec<(String, f64)> {
    stdout
        .iter()
        .map(|line| {
            let (score, file) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("output line not in `{{dssim:.8}}\\t{{file}}` format: {line:?}"));
            let score: f64 = score
                .parse()
                .unwrap_or_else(|e| panic!("score not a number in {line:?}: {e}"));
            (file.to_owned(), score)
        })
        .collect()
}

#[test]
fn gpu_cli_matches_cpu() {
    let _serial = gpu_lock();
    let cpu = run_cli(&["tests/test1-sm.png", "tests/test2-sm.png"]);
    assert!(cpu.success, "CPU run failed:\n{}", cpu.stderr);
    let gpu = run_cli(&["--gpu", "tests/test1-sm.png", "tests/test2-sm.png"]);
    assert!(gpu.success, "GPU run failed:\n{}", gpu.stderr);

    // Positive signal the GPU path actually ran (robust to rewording of the
    // fallback note, which a negative-only check would silently depend on).
    let used_gpu = gpu.stderr.contains("dssim: gpu device:");
    if !used_gpu {
        if vulkan_available() {
            panic!(
                "--gpu did not engage the GPU although a Vulkan device is available \
                 (silent CPU fallback = vacuous parity test).\nstderr:\n{}",
                gpu.stderr
            );
        }
        eprintln!(
            "SKIPPED: no Vulkan device available; --gpu fell back to CPU.\nstderr:\n{}",
            gpu.stderr
        );
        return;
    }
    // Belt-and-suspenders: the device line and the fallback note are mutually
    // exclusive; if both appear something is inconsistent.
    assert!(
        !gpu.stderr.contains("falling back to CPU"),
        "GPU device line present but also reported a CPU fallback:\n{}",
        gpu.stderr
    );

    assert_eq!(cpu.stdout.len(), gpu.stdout.len(), "line counts differ");

    let cpu_scores = parse_scores(&cpu.stdout);
    let gpu_scores = parse_scores(&gpu.stdout);
    assert_eq!(cpu_scores.len(), 1);
    assert_eq!(cpu_scores[0].0, gpu_scores[0].0, "file column differs");

    let diff = (cpu_scores[0].1 - gpu_scores[0].1).abs();
    eprintln!("cpu={} gpu={} diff={diff:.3e}", cpu_scores[0].1, gpu_scores[0].1);
    assert!(diff <= TOL, "CLI --gpu diverged from CPU: {diff:.3e} > {TOL:.0e}");

    // BH7: check the format on BOTH outputs. The GPU print macro could drift
    // (e.g. to {:.6}) and the parity diff above would still pass -- exactly the
    // F7-class rot this test guards against. Assert the 8-decimal shape on the
    // GPU stdout too, not just the CPU one.
    for (tag, lines) in [("cpu", &cpu.stdout), ("gpu", &gpu.stdout)] {
        for line in lines {
            let score = line.split('\t').next().unwrap();
            assert!(score.contains('.'), "{tag}: no decimal point in {score:?}");
            let frac = score.split('.').nth(1).unwrap_or("");
            assert_eq!(frac.len(), 8, "{tag}: expected 8 decimals, got {score:?}");
        }
    }
}

#[test]
fn gpu_cli_applies_icc_profile_identically_to_cpu() {
    // F13 demonstration: `profile.png` carries an embedded iCCP profile and
    // `profile-stripped.png` has the same color baked into its (differing) raw
    // pixels with the profile removed. `load_image::load_path` -- which BOTH
    // the CPU and GPU decode paths call -- applies the profile, so the pair
    // converges to dssim ~0. That convergence only happens if the profile is
    // actually applied (raw profile.png pixels differ from stripped), so
    // asserting BOTH paths give ~0 proves the GPU path handles ICC identically
    // to CPU -- the corrected F13 claim, now demonstrated not just asserted.
    let _serial = gpu_lock();
    let pair = ["tests/profile.png", "tests/profile-stripped.png"];

    let cpu = run_cli(&pair);
    assert!(cpu.success, "CPU run failed:\n{}", cpu.stderr);
    let gpu = run_cli(&["--gpu", "tests/profile.png", "tests/profile-stripped.png"]);
    assert!(gpu.success, "GPU run failed:\n{}", gpu.stderr);

    let used_gpu = gpu.stderr.contains("dssim: gpu device:");
    if !used_gpu {
        if vulkan_available() {
            panic!("--gpu fell back to CPU on profiled input despite Vulkan being available:\n{}", gpu.stderr);
        }
        eprintln!("SKIPPED: no Vulkan device; profile parity is CPU-only here.");
        return;
    }

    let cpu_score = parse_scores(&cpu.stdout)[0].1;
    let gpu_score = parse_scores(&gpu.stdout)[0].1;
    // ~0 on both => profile applied on both (raw pixels differ, so this is not
    // a trivial identity). A path that skipped the profile would score >0.
    assert!(cpu_score < 1e-4, "CPU did not converge profile vs stripped ({cpu_score}) -- profile not applied?");
    assert!(gpu_score < 1e-4, "GPU did not converge profile vs stripped ({gpu_score}) -- GPU ICC handling diverges!");
    let diff = (cpu_score - gpu_score).abs();
    eprintln!("profile parity: cpu={cpu_score:.8} gpu={gpu_score:.8} diff={diff:.3e}");
    assert!(diff <= TOL, "GPU/CPU profile parity: {cpu_score} vs {gpu_score} ({diff:.3e})");
}

#[test]
fn gpu_cli_falls_back_to_cpu_when_no_vulkan_device() {
    // C: exercise the CPU-fallback branch (run_gpu's Context::new() error path)
    // end-to-end. Every normal environment has a Vulkan device, so without
    // this the fallback is never actually run -- the other tests only prove
    // "GPU engaged," never "fallback works." Force the loader to see zero ICDs
    // (VK_ICD_FILENAMES / VK_DRIVER_FILES -> nonexistent file), then require
    // that --gpu falls back, exits 0, reports NO device, and matches the CPU.
    let _serial = gpu_lock();
    let bogus = if cfg!(windows) {
        "C:\\nonexistent-dssim-icd.json"
    } else {
        "/nonexistent-dssim-icd.json"
    };

    let cpu = run_cli(&["tests/test1-sm.png", "tests/test2-sm.png"]);
    assert!(cpu.success, "CPU baseline failed:\n{}", cpu.stderr);

    let out = Command::new(env!("CARGO_BIN_EXE_dssim"))
        .args(["--gpu", "tests/test1-sm.png", "tests/test2-sm.png"])
        .env("VK_ICD_FILENAMES", bogus)
        .env("VK_DRIVER_FILES", bogus)
        .output()
        .expect("dssim binary runs");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout_lines: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().map(str::to_owned).collect();

    assert!(out.status.success(), "forced-fallback run must exit 0, got:\n{stderr}");
    assert!(
        stderr.contains("falling back to CPU"),
        "expected the CPU fallback to engage with no Vulkan device, but stderr was:\n{stderr}"
    );
    assert!(
        !stderr.contains("dssim: gpu device:"),
        "must NOT report a GPU device when forced to fall back:\n{stderr}"
    );

    let fb = parse_scores(&stdout_lines);
    let cs = parse_scores(&cpu.stdout);
    assert_eq!(fb.len(), 1);
    let diff = (fb[0].1 - cs[0].1).abs();
    eprintln!("fallback parity: cpu={} fallback={} diff={diff:.3e}", cs[0].1, fb[0].1);
    assert!(diff <= TOL, "fallback score {} != CPU {}", fb[0].1, cs[0].1);
}

#[test]
fn gpu_cli_size_mismatch_matches_cpu_error() {
    let _serial = gpu_lock();
    // Different-size inputs: both paths must fail (the error text goes to
    // stderr; we only assert failure, which is the behavioral contract — it
    // holds whether or not the GPU path engaged, so no skip logic needed).
    let cpu = Command::new(env!("CARGO_BIN_EXE_dssim"))
        .args(["tests/test1-sm.png", "tests/gray1-rgba.png"])
        .output()
        .expect("dssim binary runs");
    let gpu = Command::new(env!("CARGO_BIN_EXE_dssim"))
        .args(["--gpu", "tests/test1-sm.png", "tests/gray1-rgba.png"])
        .output()
        .expect("dssim binary runs");
    assert!(!cpu.status.success(), "CPU path should fail on size mismatch");
    assert!(!gpu.status.success(), "GPU path should fail on size mismatch");
}
