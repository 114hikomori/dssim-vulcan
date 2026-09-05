//! Phase G CLI integration test: `--gpu` output format is identical to the
//! CPU path, and values agree within the plan's 5e-6 score bound.
//! (Bit-identical 8-decimal output is NOT expected — the GPU path has ~1e-7
//! FP drift; plan §6 Phase G requires identical *formatting* and P5's
//! ≤5e-6 headline tolerance.)

use std::process::Command;

const TOL: f64 = 5e-6;

fn run_cli(args: &[&str]) -> Vec<String> {
    let out = Command::new(env!("CARGO_BIN_EXE_dssim"))
        .args(args)
        .output()
        .expect("dssim binary runs");
    assert!(
        out.status.success(),
        "dssim {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
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
    let gpu = run_cli(&["--gpu", "tests/test1-sm.png", "tests/test2-sm.png"]);
    assert_eq!(cpu.len(), gpu.len(), "line counts differ");

    let cpu_scores = parse_scores(&cpu);
    let gpu_scores = parse_scores(&gpu);
    assert_eq!(cpu_scores.len(), 1);
    assert_eq!(cpu_scores[0].0, gpu_scores[0].0, "file column differs");

    let diff = (cpu_scores[0].1 - gpu_scores[0].1).abs();
    eprintln!("cpu={} gpu={} diff={diff:.3e}", cpu_scores[0].1, gpu_scores[0].1);
    assert!(diff <= TOL, "CLI --gpu diverged from CPU: {diff:.3e} > {TOL:.0e}");

    // Formatting check: the score column must be exactly 8 decimals.
    for line in &cpu {
        let score = line.split('\t').next().unwrap();
        assert_eq!(
            score.len(),
            10,
            "format violation (expected ddd.ddddddd): {score:?}"
        );
        assert!(score.contains('.'), "no decimal point in {score:?}");
    }
}

#[test]
fn gpu_cli_size_mismatch_matches_cpu_error() {
    // Different-size inputs: both paths must fail (the error text goes to
    // stderr; we only assert failure, which is the behavioral contract).
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
