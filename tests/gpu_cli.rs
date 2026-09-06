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

    // Formatting check: the score column must be exactly 8 decimals.
    for line in &cpu.stdout {
        let score = line.split('\t').next().unwrap();
        assert!(
            score.contains('.'),
            "no decimal point in {score:?}"
        );
        let frac = score.split('.').nth(1).unwrap_or("");
        assert_eq!(frac.len(), 8, "expected 8 decimals, got {score:?}");
    }
}

#[test]
fn gpu_cli_size_mismatch_matches_cpu_error() {
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
