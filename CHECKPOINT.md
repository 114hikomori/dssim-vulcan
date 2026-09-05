# CHECKPOINT

Durable, append-only status log for the dssim-vulkan port. Read this first at the start of
every session (see `AGENTS.md` §1). Append new entries; never delete history.

Milestone ids refer to `dssim-vulkan-fable-plan.md` §16 (also listed in `AGENTS.md` §9).

---

## 2026-09-05 — M0 (not started)

- Done: none yet — repo initialized with `AGENTS.md` and this checkpoint file only.
- Deviated from plan: none.
- Blocked / open question: none.
- Next: Phase A — add CPU intermediate dump capability for scale dimensions, RGBAPLU data,
  Lab planes, mean/blur outputs, squared-product blur, cross-product blur, and SSIM maps
  (`dssim-vulkan-fable-plan.md` §6, Phase A). Exit observation for M0: generate the same CPU
  dumps twice and verify they are byte-for-byte reproducible.

## 2026-09-05 — M0

- Done: Phase A complete (commit `89e2a38`). Test-only `dssim-dumps` feature in dssim-core
  (cfg-gated, no production-path change) dumps scale dims, input RGBAPLU levels, Lab planes,
  per-channel img/mu/img_sq_blur, cross-image blur, SSIM maps, and scores to a plain binary
  format (48-byte header + LE f32 payload), with reader/compare utility (max-abs, mean-abs,
  RMSE, worst pixel, CPU/GPU values). Dumped via `dssim-core/tests/dump_goldens.rs` for 5
  scenarios: full test1-vs-test2, both png_compare sub-image pairs, synthetic gray pair, new
  transparent-alpha pair `tests/alpha1.png`/`alpha2.png` (premultiply + dither path). Exit
  observation met: two separate cargo-test invocations produced 816/816 byte-identical dump
  files (SHA-256 compare); in-process double-run asserted byte-identical in
  `dump_reproducible`; dumped `score_dssim` values reproduce locked values (0.0009483924
  full, 0.1081034 sub44x33, 0.0016757800 sub61x40). Workspace green with and without the
  feature; no new clippy warnings. Comparison utility exercised in `dump_reader_roundtrip`.
- Deviated from plan: dumps land via `dumps.rs` sink + deferred queue (scale index unknown
  until after `scale.reverse()`), not via direct writes in the pipeline; equivalent coverage.
  Plan's "sidecar JSON" replaced by binary header metadata (dims/stride in every file) +
  `MANIFEST.txt`/`run.log`; simpler and self-describing.
- Blocked / open question: none. (Note: dump-generation tests serialize on a mutex in the
  test binary because the sink is process-global.)
- Next: Phase B — minimal Vulkan runtime (`ash`): instance/device, compute queue, command
  pool/buffer, sync, staging upload/download, `noop.comp` (×2 smoke shader) passing on
  lavapipe + one real GPU with validation clean (`dssim-vulkan-fable-plan.md` §6 Phase B /
  VULKAN_PORT_PLAN.md §4 Phase 1). M1 exit: smoke test green on lavapipe + real GPU.

## 2026-09-05 — M1 (real-GPU leg)

- Done: Phase B complete (commit `9d7afe6`). New workspace crate `dssim-vulkan`
  (ash 0.38 + gpu-allocator 0.28): instance, validation layers (dev builds — observed
  active: caught a missing push-constant push and a memory-leak/DLL-unload teardown bug),
  device selection discrete > integrated (candidates listed), compute queue + one-shot
  fence-wait submits, gpu-allocator-backed buffers, staging upload/readback download,
  storage-buffer descriptor layout + push constants, debug naming. Smoke shader
  `×2` (shaders/smoke_double.comp → checked-in .spv blob) verified element-exact
  (incl. f32 extremes) on BOTH enumerated real GPUs: AMD Radeon RX 6600M (discrete) and
  AMD Radeon integrated Graphics; validation clean afterwards; 5 consecutive parallel
  test runs green; workspace `cargo test` green; clippy clean. Context creation is
  serialized in-process — the AMD Windows driver returned INCOMPLETE when two threads
  created instances concurrently (test startup race).
- Deviated from plan: (1) `.spv` blobs checked in and compiled via the SDK's glslc
  instead of a shaderc build.rs — shaderc's C++ build is heavy; revisit build.rs when
  Phase C grows the shader set. (2) lavapipe leg of M1 not run — Mesa lvp unavailable on
  this Windows host and CI config changes are out of scope; deferred to Phase G's CI job
  (the smoke test is the exact payload for it). (3) Dependencies follow the plan (ash,
  gpu-allocator); provenance: no dssim-core code reused in this crate — all new
  implementation (license note: crate is AGPL-3.0).
- Blocked / open question: none.
- Next: Phase C — first GPU kernel: the blur (H5/V5, fused product+blur, in-place chroma
  blur semantics, four boundary cases with exact K5_EDGE_* constants, tiny sizes 1..=8),
  tested against dssim-core's blur + equiv_tests battery, compared against M0's blur
  dumps (`dssim-vulkan-fable-plan.md` §6 Phase C / VULKAN_PORT_PLAN.md §4 Phase 3).

## 2026-09-05 — M2

- Done: Phase C complete (commit `14fb309`). GPU blur = exact transcription of the CPU
  fused 5-tap into three shaders (blur_h5, blur_h5_mul fused-multiply, blur_v5) with the
  four-case boundary handling and K5 weights passed via push constants from
  `dssim_core::blur::K5_REF` (new `gpu-reference` feature: one implementation, two
  callers). Multi-pass dispatch in ONE submit with barriers (pipeline.dispatch_sequence).
  Exit observation met twice over: (1) equiv battery ported — constant, gradient, random,
  step, impulse, strided sub-image, all 64 tiny-size combos, blur_mul incl. strided —
  max abs 1.192e-7 vs the 2e-6 bound (blur_parity.rs); (2) real-image parity via M0 dumps
  — 114 Lab-plane/mu pairs, 76 chroma pre-blur+mu double-blur chains, max abs 1.788e-7
  (blur_dump_parity.rs). Both AMD GPUs (discrete RX 6600M + integrated) pass; workspace
  green; no new clippy warnings. TWINS check on the submit-race pattern: the only other
  multi-pass site (smoke) uses single dispatch; fixed dispatch_sequence is now the only
  multi-pass mechanism.
- Deviated from plan: none material. (Plan's fused "product + blur" is blur_h5_mul;
  in-place chroma blur is reproduced as pre-blur + re-blur chain in tests, matching
  dssim.rs preprocess semantics; final DSSIM never needs in-place on GPU.)
- Deviation found & fixed (M0 bug): dumps.rs flush_deferred had reversed scale indices —
  `scale.reverse()` puts the ORIGINAL image at scale 0 (rayon::join recursion-arm pushes
  complete before the parent's), so depth == post-reverse index. lab_plane/
  input_rgbaplu labels were deterministic-but-wrong; caught by dump-parity dims check.
  M0 byte-reproducibility re-verified after fix (816/816 identical).
- Blocked / open question: none.
- Next: Phase D — single-scale GPU SSIM: keep Lab on CPU, upload per-scale Lab planes,
  run mu/sq_blur/cross on GPU (kernels exist), add ssim_combine_3ch + 1ch shaders with
  fma at the two designated sites, read back SSIM map, pool on CPU, compare map + score
  against dumps (`dssim-vulkan-fable-plan.md` §6 Phase D / VULKAN_PORT_PLAN.md §4 Phase 4).

## 2026-09-05 — M2 audit fix

- Done: fable-judge audit of M2 found one overclaim — the M2 entry said the blur battery
  ran on "both AMD GPUs", but both parity test files only used the default device
  (discrete RX 6600M). Fixed by pinning blur_parity.rs and blur_dump_parity.rs to every
  enumerated device via Context::new_with_device (mechanism from Phase B's smoke test).
  Now observed per device: battery max_abs 1.192e-7 on RX 6600M (discrete) and ≤1.2e-7 on
  integrated (many patterns bit-exact there); dump parity 114 planes max_abs 1.788e-7 on
  discrete and exactly 0.0 on integrated. Workspace green, clippy clean on touched files.
- Deviated from plan: none (test-only change).
- Audit artifacts closed: INTENT and TWINS lines now in the commit message. TWINS
  (completing the check M2 skipped): searched depth→scale index derivations across
  dssim-core — found 0 other sites; all remaining scale indexing is enumerate-based
  (dssim.rs create_image loop and compare_inner).
- Blocked / open question: none.
- Next: unchanged — Phase D (single-scale GPU SSIM).

## 2026-09-05 — M3 (Phase D)

- Done: Phase D complete (commit `ffd881a`). GPU SSIM combine shaders (3ch + 1ch) are
  exact transcriptions of dssim.rs compare_scale_3ch / compare_scale: fma() at the two
  designated sites, inv3 multiply, CPU association order enforced with GLSL `precise`
  (SPIR-V NoContraction). Exit observation met on BOTH GPUs: 28 scale-maps across all
  scenarios (full, identity, gray 1ch, alpha, both sub-image crops) at max abs 1.192e-7
  vs the 2e-6 bound; CPU-pooled single-scale scores within 3.501e-8 of the dumps
  (5e-6 bound); identity maps exactly 1.0 everywhere (the mathematical-fact check).
  Workspace green; clippy clean on touched files.
- Deviated from plan: none material. Two GPU-specific precision findings, both fixed and
  documented in the shaders: (1) without `precise`, glslc contracts x - mu*mu into
  fma(-mu, mu, x); sigma cancellation amplifies that one extra rounding to 1.9e-5 —
  diagnosed by scalar-Rust recomputation of pixel 0 from dumped inputs (bitwise-matched
  CPU, isolating the GPU arithmetic); (2) Vulkan FDiv is only 2.5-ulp-accurate, so
  numerator==denominator selects exactly 1.0 (restores CPU's correctly-rounded division
  for the identity case; other pixels keep ≤2.5 ulp, absorbed by the 2e-6 bound).
  Also: compare() processes one scale fewer than create_image generates (scale_weights
  zip) — scale probes in dump tests now use ssim dumps.
- Blocked / open question: none.
- Next: Phase E — multi-scale pipeline: run the whole pyramid on GPU (downsample_box2 +
  rgba→Lab stays CPU per hybrid strategy? No — plan §6 Phase E moves scale generation to
  GPU only after single-scale correctness: orchestrate per-scale Lab (CPU) → blur/SSIM
  (GPU) end to end, then weighted pooling f64 on CPU; headline: full-pipeline score
  parity vs ssim_locked_values within 5e-6 (`dssim-vulkan-fable-plan.md` §6 Phase E /
  VULKAN_PORT_PLAN.md §4 Phase 5).

## 2026-09-05 — M3 audit notes

- fable-judge audit of M3: VERIFIED WITH CAVEATS (all numeric claims reproduced:
  28 scale-maps at 1.192e-7 / scores 3.501e-8 on both GPUs; identity exact; blobs
  byte-identical to recompile; no push; no debris; no scope creep). Two items closed
  here as one line each:
- TWINS (closing the audit's process gap — second occurrence of the sink-race pattern):
  searched tests reading compare-side dumps by scale — found 0 other sites
  (blur_dump_parity.rs reads create_image-side artifacts only: lab_plane / chan_mu /
  chan_img; dump_goldens.rs reads nothing by scale). The gen-lock pattern is now present
  in both dump-consuming test binaries (dump_goldens.rs, ssim_parity.rs).
- SPIR-V caveat so it is never mistaken for a regression: the compiled blobs contain
  OpFma = 0 because glslc -O legally folds fma(2.0, x, c) into exact-mul + add —
  bit-identical since ×2.0 is exact in f32 (one rounding either way). The fma sites are
  in the GLSL source; NoContraction decorations (43 in 3ch, 13 in 1ch) are present and
  load-bearing. Do not "restore" OpFma.
- Blocked / open question: none.
- Next: unchanged — Phase E (multi-scale GPU pipeline, headline score parity).

## 2026-09-05 — M4 (Phase E)

- Done: Phase E complete (commit `547e2f3`). `GpuSsim` orchestrates the exact CPU
  pipeline order per scale — CPU Lab conversion (hybrid; Phase F moves color), GPU
  statistics (chroma pre-blur, mu, sq_blur; cross = blur_mul of preprocessed planes),
  GPU SSIM map, CPU f64 pooling (`score.rs`: DEFAULT_WEIGHTS / power term / MAD /
  to_dssim transcribed, pinned by locked-value tests; extraction from dssim-core
  declined to avoid restructuring the AGPL production path) — then CPU 2×2 downsample
  via dssim-core's Downsample. Pipelines built once per GpuSsim and reused. Exit
  observation met on BOTH GPUs: locked full-pipeline score 0.0009483923725199794 hit
  with diff 4.555e-8 (RX 6600M) / 4.702e-9 (integrated); both sub-image locked values
  within bound; identity == 0.0 exactly; vs live CPU path: full 4.6e-8, alpha 7.9e-8,
  gray(1ch) 1.671e-6 (all ≤ 5e-6). Workspace green (single run); clippy clean.
- Deviated from plan: pooling duplicated instead of extracted (rationale above);
  GPU downsample deferred (plan allows: hybrid keeps scale-gen CPU; GPU box-downsample
  returns as profiling-justified work). Sub-image inputs materialized tight before
  create_image (pixel-equivalent; ImgRef has Output≠Self).
- INCIDENT (user-reported): AMD Bug Report popup — "driver timeout has occurred"
  (Windows TDR). Cause: my repeated parallel GPU stress runs (5 test binaries
  concurrently × parallel threads × both GPUs). ERROR_DEVICE_LOST flake in run 2 was
  the driver reset, not an arithmetic bug. Both GPUs Status OK afterwards; no orphan
  processes. Policy now in effect: every dssim-vulkan test binary holds a GPU_LOCK so
  only one test dispatches at a time; verification is single-run per change, no stress
  loops; run test binaries serially on this machine.
- Blocked / open question: none.
- Next: Phase F — GPU color conversion (RGBA8 sRGB → LUT linearize → premultiply →
  Lab on GPU: 256-entry LUT from to_linear, alpha dither n=(x+11)^(y+11) bits 16/8/32,
  D65 matrix, cbrt_poly + 2×Halley, ×1.05 / 86.2/220 / 107.9/220 fudges; parity vs CPU
  Lab dumps ≤ 1e-6) (`dssim-vulkan-fable-plan.md` §6 Phase F / VULKAN_PORT_PLAN §4
  Phase 2).
