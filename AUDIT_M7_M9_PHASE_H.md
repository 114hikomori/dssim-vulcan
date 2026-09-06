# Audit — M7 + M9 (Phase H first slice)

Date: 2026-09-06. Auditor: opencode (fable-judge style review, read-only).
Pinned to HEAD `acf9fca` (commits `1dabf06`, `0b29845`, `de10c0e`, `acf9fca` and the
cumulative state through M6). **No code was changed.** This file only records findings.

Scope note: at audit time the working tree contained an *in-progress, uncommitted*
Phase H optimization refactor by the implementing agent (`blur.rs`, `color.rs`,
`pipeline.rs`, `score.rs`, `ssim.rs` modified; does not compile mid-edit — batch
dispatch APIs `h5_mul_into`/`v5_into`/`combine_into` referenced but not yet
implemented). That is expected mid-phase state, not a defect; it is excluded from
findings except where a finding below interacts with it (F5).

## Verified good (re-checked this session, so the next pass need not redo it)

- **SPIR-V freshness**: all 7 `.spv` blobs are byte-identical (SHA-256) to a fresh
  `glslc --target-env=vulkan1.3 -O` recompile of their current `.comp` sources.
- **NoContraction counts** (spirv-dis): blur_h5=31, blur_h5_mul=51, blur_v5=31,
  ssim_combine_3ch=43, ssim_combine_1ch=13, rgba_to_lab=53 — consistent with the
  M3/M7 checkpoint claims.
- **Blur shaders** vs `dssim-core/src/blur.rs`: 4-case boundary structure, per-op
  association order, edge-weight order, `saturating_sub`→ternary clamps, `min()`
  clamps, and the fused `blur_h5_mul` product all match the CPU transcription.
- **rgba_to_lab** vs `tolab.rs` + `image.rs::to_rgb`: cbrt_poly seed/Halley order,
  `fma_matrix` association, piecewise branch, 1.05 / 86.2/220 / 107.9/220 fudges,
  dither bits 16/8/32 and the literal `a < 255.0` guard all match.
- **ssim_combine 3ch/1ch** vs `compare_scale_3ch`/`compare_scale`: the two `mul_add`
  sites, inv3 multiply, subtraction order, and the `numer == denom → 1.0` identity
  select match.
- **score.rs pooling** matches `dssim.rs:334-337,357,481-483` exactly (`sum/len`,
  `powf(0.5.powf(n))`, `score.mul_add(weight, ssim_sum)`, `to_dssim`).
- **Scale-count semantics**: CPU generates 6 scales / uses 5 (weights zip); GPU
  generates ≤ 5 and uses all — equivalent results, including the 8-px downsample
  cutoff (`image.rs:209`).
- **CI**: runs #1/#2 on GitHub Actions genuinely executed every GPU parity suite on
  llvmpipe (logs inspected; per-test timings plausible). "both runs green" is true.

## Findings

### Correctness / robustness

**F1 (medium, latent) — staging transfers assume coherent host memory.**
`dssim-vulkan/src/transfer.rs:208-219` (`write_mapped`) never calls
`flush_mapped_memory_ranges` before the GPU copy reads the staging buffer, and
`read_bytes` (`transfer.rs:223-231`) never invalidates before the host reads the
readback. Correct only while the allocation is `HOST_COHERENT` (true on this host
and lavapipe, not guaranteed by the spec). Fix: flush/invalidate explicitly, or
assert coherence via memory-property flags at allocation.

**F2 (low) — fence/command-buffer leak on submit failure.**
`dssim-vulkan/src/context.rs:285-290`: if `queue_submit` or `wait_for_fences`
returns `Err`, the fence created at 260-263 and the command buffer are never
destroyed (the `?` skips the cleanup at 292-293). Error-path only.

**F3 (low-medium) — `GpuSsim::compare` lacks a per-scale dimension assert.**
`dssim-vulkan/src/score.rs:150-161` checks channel count but not that ref/mod
width/height match. The CPU 1ch path asserts (`dssim.rs:445-446`); the GPU would
silently compute garbage for equal-pixel-count/different-shape inputs (e.g. 4x3 vs
3x4 — `blur_mul`'s length assert passes). The CLI checks sizes first
(`src/main.rs:194`), so this is a library-API gap, not a CLI bug.

**F4 (very low, theoretical) — pixel-count u32 truncation.**
`dssim-vulkan/src/blur.rs:88,141` (`(width*height) as u32`) and the shaders'
`idx >= w * h` guard overflow for > 2^32 pixels; the CPU asserts each dim < 2^24
but not the product. Unreachable in practice (≈40 GB f32); a debug assert would
close it.

**F5 (medium for the in-flight work) — descriptor pool cap is 64 with no growth.**
`dssim-vulkan/src/pipeline.rs:105-118`: comment claims "grows via a new pool if
ever exceeded" but no growth exists; exceeding it fails `allocate_descriptor_sets`
mid-sequence. Today's max is 2 passes/sequence, but the Phase H batching refactor
(one sequence with many passes) will hit this. The comment should be fixed and the
cap sized to the batch plan.

**F6 (low) — `Error::source()` not implemented.**
`dssim-vulkan/src/error.rs:29`: the `Loader`/`Vulkan`/`Allocator` variants wrap
inner errors but don't expose them, so `src/main.rs:50` (`e.source()`) never prints
the underlying cause. Hurts field diagnosis of device-lost/allocator failures.

### Test quality

**F7 (medium) — `gpu_cli` passes even when the GPU path never ran.**
`tests/gpu_cli.rs:44-68`: if `Context::new()` fails, the CLI prints a note and
falls back to CPU with exit 0; the test then compares CPU vs CPU and still passes.
The "GPU parity" claim can silently rot on any Vulkan-less machine. Fix: assert
stderr does not contain "falling back to CPU" (or assert a device line).

**F8 (medium) — cross-process GPU concurrency in `gpu_cli`.**
The two tests in `tests/gpu_cli.rs` run in parallel harness threads, each spawning
`dssim --gpu` subprocesses → up to 2 concurrent GPU processes. `GPU_LOCK` is
per-process and does not span them; the M1 entry documents that concurrent instance
creation on the AMD Windows driver returns `INCOMPLETE`. Flake/TDR risk on this
host. Fix: `--test-threads=1` for that binary, or a file lock.

**F9 (low) — "identity" scenario in `ssim_parity` is mislabeled.**
`dssim-vulkan/tests/ssim_parity.rs:90`: comment says "identity (same image twice)"
but the call passes `img1` and `img2` (different images) — it is a duplicate of the
full scenario. Real identity coverage exists in
`gpu_ssim_identity_is_exactly_one`, so no hole, but the label lies and the run
wastes a dump cycle.

**F10 (low) — `gpu_lab_dither_is_active` is vacuous.**
`dssim-vulkan/tests/lab_parity.rs:129-158`: it only asserts the GPU outputs of two
*different* images differ — true even if the shader dropped the dither entirely.
Actual dither coverage is `gpu_lab_matches_cpu` (CPU applies the dither). The
doc-comment overstates what the guard catches; either strengthen it (compare GPU
vs CPU on a coordinate-shuffled copy) or rename/demote it.

**F11 (very low) — CLI format assertion assumes a 1-digit integer part.**
`tests/gpu_cli.rs:62` (`score.len() == 10`) breaks for dssim ≥ 10 (possible for
very different images). Only small fixtures are exercised today.

**F12 (cosmetic) — temp dump dirs leak on failure.**
`remove_dir_all` runs only on the success path in `ssim_parity.rs`,
`blur_dump_parity.rs`, `dump_goldens.rs`; a panicking run leaves files in TEMP.

### Documentation / claim accuracy

**F13 (medium) — the "GPU path skips ICC profiles" claim appears false.**
`src/lib.rs:24-27` (root crate) and CHECKPOINT M6 deviation (2) say the GPU decode
skips ICC and "may differ from the CPU path's decoding". But `load_image_rgba` and
the CPU `load()` call the *same* `load_image::load_path`, which applies embedded
profiles and converts to sRGB (load_image 3.4.1 docs). The only differences are the
pixel-format traits (`to_rgblu` vs `to_rgbaplu`, provably equivalent for the Lab
path). Either correct the comment/checkpoint, or demonstrate the claimed divergence
with a profile-bearing fixture on both paths.

**F14 (low) — stale "byte-identical" claim in `run_gpu`.**
`src/main.rs:146-147` still says output format is "byte-identical to the CPU path";
M6 explicitly retracted that (GPU FP drift shows in the 8th decimal; the golden
test asserts format + ≤5e-6). The comment contradicts the recorded decision.

**F15 (low) — push-constant byte counts wrong in 4 shader headers.**
`blur_h5_mul.comp:6` and `blur_v5.comp:7` say "48 bytes" (actual 56);
`ssim_combine_3ch.comp:12` and `ssim_combine_1ch.comp:6` say "56 bytes" (actual
48). The Rust structs and `const _: () = assert!` sizes are correct; only comments
lie.

**F16 (low) — stale dssim.rs line citations in combine shaders.**
`ssim_combine_3ch.comp:2-3` cites "lines 335-395" (actual 377-437);
`ssim_combine_1ch.comp:2-3` cites "397-436" (actual 440-478). AGENTS.md makes
line-cited references load-bearing; they drift silently.

**F17 (low) — manifest hash doc/code mismatch.**
`dssim-core/src/dumps.rs:21` says "FNV-1a hash"; the code uses `DefaultHasher`
(SipHash, `dumps.rs:193`). Also `DefaultHasher` output is not guaranteed stable
across Rust releases, so byte-comparing manifests across toolchain upgrades can
break; pin the hash if manifests become long-lived goldens.

**F18 (cosmetic) — crate doc says "currently Phase B".**
`dssim-vulkan/src/lib.rs:6`.

**F19 (medium — checkpoint honesty) — M7 gray-fixture claim overreaches.**
CHECKPOINT M7 entry: "gray pair (synthetic + CLI gray1 fixtures via image_gray
tests)". `image_gray` (`src/main.rs:268`) is a CPU-only test; no GPU test loads any
`gray1-*.png` fixture (zero grep hits in `dssim-vulkan/tests/`). GPU gray coverage
is synthetic-only. Either add a GPU CLI gray test or fix the entry wording.

**F20 (low) — README GPU paragraph omits caveats.**
`README.md:32-37` mentions tolerance and `-o`; it does not mention the profile
question (F13) or 16-bit handling. One line each would keep user-facing docs honest.

### Process / CI

**F21 (low-medium) — CI never exercises validation layers or clippy.**
`.github/workflows/ci.yml` installs `mesa-vulkan-drivers + libvulkan1` only;
`VK_LAYER_KHRONOS_validation` is absent, so the context auto-skips validation and
CI's "validation clean" is effectively never tested (local-only observation).
Adding `vulkan-validationlayers` is one apt line. No clippy/fmt job exists despite
checkpoints repeatedly claiming "clippy clean".

**F22 (cosmetic) — `cargo fmt` is vacuous repo-wide.**
`.rustfmt.toml` sets `disable_all_formatting=true`, so the one-line statement
glitches (`context.rs:200`, `phase_e.rs:159`, `dumps.rs:177`) are caught by nothing.
Either re-enable fmt or stop treating formatting as a checked property.

**F23 (low) — M9 deliverable has clippy warnings.**
`dssim-vulkan/examples/bench.rs:13`: unused imports `Downsample as _`,
`ToLABBitmap as _` (2 warnings via `cargo clippy --workspace --all-targets`).
Separately, `--all-targets` fails on stable because `benches/compare.rs` uses
`#![feature(test)]` (pre-existing upstream; `cargo bench` needs nightly — worth a
README note).

**F24 (informational) — HEAD is 5 commits ahead of origin.**
The precise-blur fix and the M7 matrix have not run on llvmpipe yet (CI runs stop
at `505cfa0`); commit `0b29845`'s "CI re-runs it on llvmpipe" is a future statement.
Pushing is gated on user authorization (AGENTS.md §5) — listed as a pending
follow-up, not a defect.

## Caveats of this audit

- The M7/M9 *runtime* numbers (17x9 ≤ 5e-6 on both GPUs, bench table) were **not**
  re-observed: the tree does not compile mid-refactor and stashing another agent's
  work would interfere. They are taken as reported; the static claims above
  (blob freshness, NoContraction counts, transcription fidelity, CI logs) were all
  independently re-verified.
- Findings F1, F5, F13 are the ones most likely to bite the in-flight Phase H
  optimization work; worth resolving in that phase's next commit.
