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

## 2026-09-05 — M5 (Phase F)

- Done: Phase F complete (commit `37c0b4e`). `rgba_to_lab.comp` converts
  premultiplied-linear RGBA → planar Lab on GPU with every CPU semantic transcribed:
  alpha dither (n=(x+11)^(y+11), bits 16/8/32, literal `a < 255.0` guard), XYZ matrix
  rows in fma_matrix order, piecewise cbrt_poly + 2×Halley, the 1.05 / 86.2/220 /
  107.9/220 fudges, and the gray ×1.16 branch. All 18 constants ship from
  `dssim_core::tolab::LAB_GPU_CONSTANTS` (new gpu-reference export, expressions copied
  verbatim from tolab.rs) via push constants. GpuSsim::create_image /
  create_image_gray now use GPU Lab. Exit observation met on BOTH GPUs (single run):
  Lab parity test1/alpha1/alpha2 8.345e-7, gray 2.980e-7 (bound 1e-6) with identical
  worst pixels across devices + dither-activity guard; E2E through GPU Lab: locked
  full score diff 1.650e-8 (discrete) / 7.076e-8 (integrated), identity == 0.0 exact,
  gray vs CPU 1.886e-6 (bound 5e-6). Workspace green; clippy clean.
- Deviated from plan: sRGB gamma LUT + premultiplication stay on CPU (hybrid: the CPU
  needs the linear RGBAPLU for its 2×2 downsample; moving LUT to GPU only pays off
  with GPU downsampling — deferred to Phase H opt-in with the GPU box-downsample).
  Lab drift uses 0.83 of the 1e-6 budget (cbrt_poly division, 2.5-ulp FDiv
  allowance); downstream budgets (2e-6 map, 5e-6 score) have larger headroom and E2E
  confirms. VULKAN_PORT_PLAN's "LUT as R32F texture" task subsumed by this deviation.
- Blocked / open question: none.
- Next: Phase G — integration & hardening: --gpu CLI flag (output format
  byte-identical), CPU fallback, backend selection, lavapipe CI job, real-GPU smoke
  regression, benchmark harness, docs (`dssim-vulkan-fable-plan.md` §6 Phase G /
  VULKAN_PORT_PLAN §4 Phase 6). CI config changes require explicit user approval
  (AGENTS.md §8).

## 2026-09-05 — M6 (Phase G, CLI portion)

- Done: CLI integration shipped (commit `8608e8d`). `dssim --gpu` runs the full
  GPU hybrid pipeline (decode/profiles/linearize/premultiply/downsample on CPU —
  Lab/stats/SSIM/score on GPU); automatic CPU fallback prints a note when no Vulkan
  device exists; the default CPU path is untouched; `-o` map writing warns+ignores on
  GPU (maps are a Phase-H TODO). README documents usage + limitations. Verified
  (single run per TDR policy): gpu_cli.rs golden test — CPU vs --gpu stdout parses in
  `{dssim:.8}\t{file}` format with values within 5e-6 (observed 1.700e-7);
  size-mismatch failure parity; workspace green; clippy clean; both
  `--features gpu` (default) and `--no-default-features` (CPU-only) compile.
- Deviated from plan: (1) "byte-identical stdout" from the plan is NOT achievable —
  GPU FP drift (~1e-7) shows in the 8th decimal; the golden test asserts identical
  FORMAT + values <=5e-6 (plan P5's own tolerance). Honest reformulation, not a
  weakened check: the old claim was physically impossible. (2) ICC profiles are
  skipped on the GPU decode path (load_image_rgba) — documented; profile-bearing
  fixtures may differ from the CPU path's decode (the CPU path keeps full profile
  handling). (3) CI/lavapipe job, benchmark harness, backend selection UI, real-GPU
  nightly matrix: NOT started — CI config changes need explicit user approval
  (AGENTS.md §8); benchmark harness is Phase H territory (VULKAN_PORT_PLAN Phase 7).
- Blocked / open question: none. PENDING (user decision): CI/lavapipe job authorship.
- Next: Phase H — performance & robustness (VULKAN_PORT_PLAN §4 Phase 7): profile
  vs CPU baseline first, then opt-in GPU downsampling/LUT, batch mode, async
  readback. Performance numbers must be measured before any claim.

## 2026-09-05 — M6 addendum (CI approved + pushed)

- Done: user approved the CI job in-session ("lavapipe CI job นี่หมายถึงบน github
  ใช่มั้ย อันนั้นนายลงมือได้เลย"). Wrote `.github/workflows/ci.yml` (Ubuntu runner,
  mesa-vulkan-drivers + libvulkan1 = lavapipe, build + single `cargo test
  --workspace` run per TDR policy; validation layers auto-skip when the KHRONOS
  layer is absent — context already handles that), committed as `ab5cc2e`, and
  pushed to origin/main under that authorization (15 commits, branch now
  synced). This closes the M1 lavapipe leg too: the CI job runs the same parity
  suites on the llvmpipe CPU device.
- Caveat: the first CI run's outcome has NOT been observed by this session (no
  GitHub Actions log access from here — per repo rules the user pastes the log
  if it fails). Local verification of the workflow is limited to YAML sanity.
- Blocked / open question: CI run result unknown until next check.
- Next: Phase H — performance & robustness (VULKAN_PORT_PLAN §4 Phase 7): profile
  vs CPU baseline first, then opt-in GPU downsampling/LUT, batch mode, async
  readback. Performance numbers must be measured before any claim.

## 2026-09-06 — M7 + M9 (Phase H first slice)

- Done: (M7) fixture matrix green end-to-end: alpha pair, gray pair (synthetic +
  CLI gray1 fixtures via image_gray tests), odd sizes 17x9 / 9x7, small 16x16 and
  1x1, tiny blur sweep 1..=8 — all on both GPUs within 5e-6 (phase_e small/odd
  test added). (M9) performance profile completed: dssim-vulkan/examples/bench.rs
  measures compare-only CPU vs GPU on the discrete GPU (release, warmup 2, iters 7).
  Measured table (RX 6600M): 320x200 cpu 4.86ms / gpu 74.34ms (15.3x); 1024x1024
  71.59 / 373.30 (5.2x); 2048x2048 307.45 / 1291.28 (4.2x). GPU is SLOWER —
  overhead-bound (per-call staging upload/download, fence per dispatch, no
  batching). No speedup claimed; this is the baseline optimization must beat.
- Deviation found & fixed (M9 uncovering it): blur shaders lacked `precise`
  (NoContraction) — compiler FMA fusion rounded differently from CPU, and the
  SSIM sigma cancellation amplified 1-ulp blur drift ~5e4x on tiny low-contrast
  patches (17x9 score diff 6.5e-6 > 5e-6). Fixed with precise on all 15 blur
  accumulation sites; SPIR-V NoContraction went 0 -> 31 (pre-fix blob at
  1dabf06 measured 0 via spirv-dis, post-fix blob measures 31), and 17x9 is
  now <= 5e-6 on both GPUs, bit-identical across 3 consecutive runs.
  TWINS: searched missing-precise accumulations across dssim-vulkan shaders -
  the 3 blur shaders only (Lab/SSIM shaders had it since M3/M5).
  Note: an initial re-audit wrongly compared HEAD~1 vs HEAD (both post-fix)
  and concluded NoContraction was already present; the correct pre-fix
  baseline is HEAD~2 (1dabf06), which measures 0.
- Deviation found & fixed (cosmetic but real): blur_h5.comp had been a
  3-shader concatenation since Phase C (PowerShell Set-Content collision); h5.spv
  compiled from it still worked (entry point survived). Restored h5-only source,
  recompiled all three blobs. mul/v5 spvs were always compiled from clean sources.
- CI: workflow live on GitHub Actions (user-approved push); both runs green;
  parity suites now also run on llvmpipe every push. Actions access via MCP
  works (list runs verified; log fetch pending toolset).
- Blocked / open question: none.
- Next: Phase H continuation — optimization: GPU-resident scale data (upload
  once per image), batched per-channel dispatches, async readback; then re-run
  bench.rs and update the measured table. Target re-set after profiling per plan
  (the >=5x-with-batching target was provisional).

## 2026-09-06 — M10 (Phase H optimization landed; GPU beats CPU at medium/large sizes)
- Done: GpuSsim refactored to GPU-resident per-scale planes + ONE batched
  submit in compare (cross-blurs -> combine -> map readback); only the tiny
  SSIM maps return for CPU f64 pooling. Infrastructure: `Pass::CopyBuffer`
  (device-to-device), `Buffer` now `Arc`-cloneable (Deref to BufferInner) so
  pass records own buffers past local scopes, blur shaders gained plane-offset
  push constants (src_off/dst_off). Verified by full workspace green on RX
  6600M: blur parity (9), phase_e locked values + CPU-reference parity,
  gray/RGB size matrix. Every committed .spv re-verified == fresh
  `glslc --target-env=vulkan1.3 -O` recompile (all shaders, hash match).
- Bugs found & fixed during verification (both from the refactor):
  (1) compare's `h5_mul_into` wrote `tmp` at the plane offset while `v5_into`
  always reads `tmp` from offset 0 -> out-of-bounds write (tmp is 1 plane) and
  cross planes 1,2 held stale channel-0 data -> RGB dssim = 2^52-1 garbage.
  Fixed to dst_off=0, matching create_image. TWINS: audited every h5_mul_into
  call site; create_image already used dst_off=0, only compare was wrong.
  (2) `blur_mul` wrapper still built push constants with the pre-offset
  `pc_bytes` layout after the h5_mul shader moved src2_stride into dims2 ->
  blur_mul parity broke (max_abs 3e-1 at row boundaries). Switched to
  `pc_mul_bytes` with zero offsets. TWINS: audited all pc_bytes callers; the
  non-mul blur wrapper's pc_bytes use is still correct (offsets read as 0).
  (3) A split-experiment leftover orphaned the combine in an undispatched
  `passes2` vec (combine never ran -> map readback uninitialized); merged back
  into the single `passes` sequence.
- Measured (bench.rs, RX 6600M, full create_image+compare path): GPU/CPU ratio
  0.82 (1024^2), 0.93 (2048^2), 1.14 (320x200) — down from the 4.2-15.3x-slower
  Phase-H baseline (commit 1dabf06). M10 met for medium/large images; small
  images still lose to fixed submit/setup overhead.
- Deviated from plan: none material. Removed dead `ColorPipelines.context`
  field (unused after lab_into stopped self-dispatching).
- Blocked / open question: none.
- Next: M10 is met at >=1024px but not at 320x200. Decide whether to chase the
  small-image overhead (persistent command buffers / fewer barriers) or accept
  it and move to M8 (CLI --gpu already wired; confirm fallback + CI cover the
  optimized path) and M9 sign-off. Also: run the optimized path on the
  integrated GPU + CI lavapipe before calling M10 fully shipped.

## 2026-09-06 — Audit pass 2 fixes + small-image overhead eliminated (M10 strengthened)
- Trigger: user redirected to AUDIT_GPU_PORT.md (a second agent's fable-judge
  review of the Phase H work). Pass 2 findings F25-F33 target `815eab8`/`c4ee8d4`.
- F26 (enabler): context.rs now prints validation state at Context creation in
  debug builds. On this host validation is in fact ENABLED (the audit's env had
  it silently off) — which let me reproduce F25 directly rather than infer it.
- F25 (HIGH, spec violation): reproduced via validation — the lab->img
  CopyBuffer used STORAGE-only buffers (VUID-vkCmdCopyBuffer-srcBuffer-00118/
  -00120). Fixed by writing Lab planes straight into img_all (plane 0 = raw L),
  deleting the illegal copy AND the separate lab_all buffer. TWINS: swept every
  CopyBuffer call site for missing transfer usage — staging->rgba and
  map->readback already had correct flags; lab->img was the only violation.
  Validation now reports ZERO errors across the whole workspace.
- F27 mu_all -> GpuOnly; F5 descriptor pool fixed-64 (false "grows" comment) ->
  named MAX_SETS_PER_POOL=128 sized to the batch plan (worst case v5=40);
  F29 compare asserts ref/mod width+height; F32 CopyBuffer asserts equal size;
  F33 deleted ambiguous pc_bytes builder + v5_into dead _stride; F30/F31 debris
  + "tiny maps" wording. (commit b63ebd5)
- Small-image overhead (the user's actual ask): measured with temporary
  create-probe instrumentation. create_image was CPU-bound and DOMINATED by the
  RGBA upload — three passes (interleave Vec<f32> -> pack_f32 Vec<u8> ->
  write_mapped memcpy). Collapsed to one direct write into mapped staging via
  new transfer::write_mapped_f32_with (RGB + gray); removed dead pack_f32.
  Also closed F1 (Pass-1): sync_host_range now flushes after writes /
  invalidates before reads (no-op when HOST_COHERENT). (commit 1a25751)
- Measured (RX 6600M, full create+compare, GPU/CPU ratio): 0.50 (320x200),
  0.34 (1024^2), 0.44 (2048^2) — GPU now beats CPU at EVERY size, not just
  large. dssim_check byte-identical before/after (parity preserved).
- Verified: full workspace green WITH validation (0 vulkan errors, 15 suites);
  clippy clean for all dssim-vulkan code; no test/shader files touched.
- Deviated from plan: none.
- Blocked / open question: F28 (peak-VRAM at 4K from the single-submit design,
  ~1.9GB at 4096^2) is NOT fixed — it's a large-image memory concern distinct
  from small-image overhead, and the transient-arena fix has a real correctness
  hazard (staging is CPU-written at record time, so it cannot be shared across
  scales within one submit). Bench only exercises to 2048^2.
- Next: human decision on F28 scope — is 4K a near-term requirement (do the
  arena / multi-submit split now) or deferred (accept current peak, revisit if
  4K support is requested)? Also still pending from prior entry: validate the
  optimized path on the integrated GPU + CI lavapipe before calling M10 shipped.

## 2026-09-06 — F7 + F28 closed (audit follow-ups after the push/CI check)
- Context: pushed the Phase-H + audit-fix batch (authorized "push to check");
  CI run #4 green on lavapipe (phase_e/smoke/ssim_parity genuinely ran, 0
  ignored). Then addressed the two open items the user raised.
- F7 (gpu_cli vacuous): run_gpu now prints "dssim: gpu device: <name>" to
  stderr; gpu_cli_matches_cpu asserts that POSITIVE line (not just absence of
  the fallback note, which would rot on rewording). If absent, it probes
  Vulkan in-process (dssim_vulkan::Context::new): no device -> SKIPPED with a
  clear message; device present but --gpu fell back -> hard FAIL. Gated the
  file on #![cfg(feature="gpu")]. Bonus F11: format check now asserts 8
  fractional digits (no 1-digit-integer assumption). (commit fa7a99c)
- F28 (peak VRAM): investigated first — img/mu/sq are pure GPU-write (never
  CPU-written/read back) BUT persistent outputs compare() needs, so they can't
  be arena-reused across scales; only rgba/tmp can, which is insufficient. So
  the fix is submission boundaries, not aliasing. Added SPLIT_SUBMIT_MIN_PIXELS
  (6M ~2450^2): create_image/create_image_gray/compare flush per scale at/above
  it (transients free per scale -> back to ~72 B/px), batch below it (keeps the
  small-image single-submit win; 2048^2 stays batched). (commit 231e96f)
- F28 verification: split path is unreachable in CI at 6M px on lavapipe, so
  added a test seam set_split_threshold_for_test(0) and
  phase_e_split_submit_matches_batch_and_cpu — forces per-scale flush on small
  images, asserts split == batch BIT-FOR-BIT (same passes, only fence
  placement differs) and both == CPU, RGB + gray. Added 4096^2 to bench
  (adaptive iters): runs clean, 0.45x CPU, valid score, no OOM.
- Measured (RX 6600M): 320x200 0.58x, 1024^2 0.37x, 2048^2 0.42x, 4096^2 0.45x.
- Verified: full workspace green WITH validation (0 vulkan errors, 15 suites);
  clippy clean; new test is strictly stronger (no weakening).
- Deviated from plan: none.
- Blocked / open question: F28 VRAM reduction is analytical (per-scale flush
  frees transients) + confirmed 4K runs, NOT measured byte-for-byte via vendor
  tooling. Remaining Pass-1 findings not yet done: F13 (ICC claim wording),
  F14/F15/F16/F17/F18/F19/F20 (doc/comment accuracy), F2 (fence leak on error),
  F3 (compare dim assert — DONE this session via F29), F6 (Error::source),
  F8 (gpu_cli cross-process lock), F21/F22 (CI validation+clippy legs), F23
  (bench clippy — DONE), F24 (push — DONE).
- Next: push F7+F28 to CI to confirm the new split test + gpu_cli hardening
  pass on lavapipe (needs user's go-ahead per AGENTS.md §5). Then optionally
  sweep the remaining low-severity Pass-1 doc/robustness findings.

## 2026-09-06 — Audit Pass-1 sweep (F2/F6/F8/F13-F22)
- Doc/claim accuracy (comment-only; .spv unchanged, still hash-match): F15/F16
  (shader header byte-counts were swapped blur=56/combine=48, and stale
  dssim.rs line cites -> corrected to 377-438 / 440-479 + field meanings to the
  src_off/dst_off layout); F13 (load_image_rgba does NOT skip ICC -- same
  load_path as CPU, profiles identical); F14 (run_gpu output is same *format*,
  not byte-identical values); F18 (crate doc Phase B -> Phase H/M10); F17
  (dumps.rs manifest hash is std DefaultHasher/SipHash, not FNV-1a; not stable
  across Rust releases -> same-toolchain checks only); F20 (README GPU para now
  states profiles match CPU + flags 16-bit->8-bit gap). (commit 717da60)
- Code: F2 (submit_one_shot leaked fence+cmdbuf on queue_submit/wait error --
  cleanup now on every path); F6 (Error::source() implemented for
  Loader/Vulkan/Allocator so main.rs prints the real cause); F8 (gpu_cli
  process-local GPU_LOCK serializes its two subprocess-spawning tests).
- F21 (CI): installed vulkan-validationlayers so the debug build enables
  validation in CI; test step now FAILS on any "[vulkan ERROR]"/"VUID-" line
  (previously "validation clean" was never tested); added a clippy gate
  `cargo clippy -p dssim-vulkan -p dssim --no-deps --lib --bins --tests
  --examples -- -D warnings`. The gate immediately caught a real doc-lint
  (a ">=" line read as a markdown blockquote) -- reworded. (commit 37d9fd8)
- F19 CORRECTION (append-only; supersedes the M7 entry's wording): M7 claimed
  "gray pair (synthetic + CLI gray1 fixtures via image_gray tests)". That is
  wrong -- image_gray (src/main.rs) is a CPU-only unit test; NO GPU test loads
  any gray1-*.png fixture. GPU gray coverage is SYNTHETIC ONLY
  (create_image_gray in phase_e/lab tests). The CLI has no gray->GPU routing
  (load_image_rgba feeds every image through the RGB create_image path), so a
  real GPU-CLI-gray test isn't feasible without adding that path. Wording
  corrected here rather than editing the M7 history.
- F22: cargo fmt is VACUOUS repo-wide (.rustfmt.toml disable_all_formatting=true
  is upstream Kornel policy, commit 73933cd, pre-port). We do NOT add a fmt CI
  gate and STOP claiming "fmt clean" going forward. Clippy, by contrast, is now
  genuinely gated in CI (F21), so "clippy clean" is a real, enforced property.
- Verified: full workspace green WITH validation (0 vulkan errors, 15 suites);
  clippy --no-deps -D warnings green for our crates; all .spv match fresh
  recompile; ci.yml valid YAML.
- Blocked / open question: the new CI validation gate is only proven on AMD
  locally -- needs a lavapipe CI run to confirm it stays green there (push to
  check). If lavapipe+validation surfaces a benign complaint, narrow the gate.
- Next: push the sweep + CI change and watch run #6 (validation + clippy on
  lavapipe). Remaining audit items after that: none from Pass 1/2 except the
  conditional F28 byte-level VRAM measurement (deferred) and F13's optional
  profile-bearing fixture demonstration.

## 2026-09-06 — F13 + F28 closed out (demonstrated / measured, not just argued)
- F13: added gpu_cli_applies_icc_profile_identically_to_cpu. profile.png has an
  iCCP chunk; profile-stripped.png has the same color baked into DIFFERING raw
  pixels (verified: IDAT differs, iCCP present/absent via chunk dump). Both CPU
  and GPU decode give dssim 0.0 for the pair -- which only happens if the
  profile is applied (raw pixels differ), so this EMPIRICALLY proves the GPU
  path applies ICC identically to CPU. Closes the "correct OR demonstrate"
  option from the audit (previously only the comment was corrected).
- F28: Context now tracks live+peak allocation bytes (AtomicUsize in
  alloc_buffer / BufferInner::drop) with live_alloc_bytes/peak_alloc_bytes/
  reset_alloc_peak. phase_e_split_submit_lowers_peak_vram forces batch vs split
  via the seam on 512^2 and MEASURES peaks: batched 25.1 MB vs split 18.9 MB,
  6.27 MB saved on both GPUs -- matches the analytical ~24 B/px (=> ~400 MB at
  4K). The VRAM reduction is now measured, not inferred.
- Verified: full workspace green WITH validation + --nocapture (0 vulkan errors,
  validation ENABLED 71x, 15 suites); clippy --no-deps -D warnings clean.
- Deviated from plan: none.
- Audit status: ALL findings F1-F33 now addressed. F28's byte-level measurement
  and F13's fixture demonstration (the two remaining "optional" items) are done.
  No open audit findings remain.
- Next: push db7240f and confirm run #8 green on lavapipe (new F13 profile test
  + F28 measurement test). Then the port is at M10 with a clean audit, CI
  validation+clippy gates, and measured perf/VRAM.

## 2026-09-06 — audit pass-3 dispositions (auditor session)
- Done: Pass-3 verification committed (b3c20db): all F1-F33 fix claims
  reproduced; user confirmed P1 (CI edits) and P2 (pushes) were personally
  authorized ("P1-2 ฉันสั่งเอง"); P3 collision hypothesis refuted by user
  (no other agent active) — downgraded to one-off transient GPU abort.
- Deviated from plan: none.
- Blocked / open question: M10 sign-off still awaits integrated-GPU + CI
  lavapipe run #8 confirmation (per 878de7a).
- Next: watch CI run #8; then M8 CLI/fallback integration per plan.

## 2026-09-06 — M10 SIGNED OFF (exit-observation verified on both GPUs + CI)
- M10 exit-observation: "optimized path beats measured CPU baseline." Now
  observed on all three validation targets, not inferred:
  * Discrete RX 6600M: GPU/CPU 0.34-0.58x across 320x200..4096^2 (bench).
  * Integrated AMD Radeon Graphics: 0.46x (320x200), 0.52x (1024^2), 0.52x
    (2048^2), 0.57x (4096^2) -- GPU wins at every size incl 4K (bench, new
    DSSIM_BENCH_DEVICE pin; commit de47c41).
  * CI lavapipe: full parity suite green (run #8), validation ENABLED, 0 errors.
- Parity held throughout: dssim_check identical across both GPUs and to CPU
  within 5e-6; full workspace green WITH validation + --nocapture (0 vulkan
  errors, validation ENABLED 71x, 15 suites, 0 ignored).
- fable-judge on the M10 claim (this session's work, acf9fca..HEAD):
  * Test-weakening hunt: only two test files touched (phase_e.rs, gpu_cli.rs);
    every change is an ADDED test or a strictness-neutral adaptation; the F11
    format check got STRICTER (frac.len()==8). TOL unchanged at 5e-6. No
    assertions deleted/loosened, no skips added.
  * Verification re-run by the judge, not trusted: fresh bench (both GPUs) +
    fresh full-workspace test with validation -- both reproduce.
  * Scope: all changes trace to the Phase-H optimization + audit findings; no
    unrelated drive-by edits. The one new knob (DSSIM_BENCH_DEVICE) is test/
    tooling-only, default behavior unchanged.
  * Outward actions: 5 pushes this session, each preceded by the user's own
    "push"/"push to check" (quoted at the time). No un-authorized remote change.
  * Debris: none -- debug probes removed, dead pack_f32 removed, .spv hash-match,
    clippy clean. Only untracked item is .serena/ (tool dir, correctly not
    committed).
- Verdict: VERIFIED. M10 met. All milestones M0-M10 complete; audit F1-F33 +
  P1-P3 dispositioned. No open correctness or milestone items.
- Honest residual caveats (not blockers, logged): F28 byte-level VRAM reduction
  is measured at 512^2 (6.27MB saved, deterministic) and extrapolated to ~400MB
  at 4K, not measured at 4K directly; F1's flush branch is unexercised on this
  HOST_COHERENT host; 16-bit GPU inputs are an 8-bit approximation (README
  caveat, non-goal).
- Next: none required for M10. Deferred-by-design (need profiling justification
  per AGENTS.md §8): sub-scale 4K chunking below ~72 B/px, GPU-side pooling,
  GPU image decode. Optional: tag a release (needs explicit user instruction).

## 2026-09-06 — M8 SIGNED OFF (retroactive; work done in Phase G, declaration was missing)
- Exit-observation (plan §9 M8: "CLI + CI + fallback integrated") — now all
  observed, not just present:
  * `--gpu` flag wired into the CLI (run_gpu), routes through GpuSsim. ✓
  * CPU fallback: logic in run_gpu's Context::new() error path, AND now
    exercised end-to-end by gpu_cli_falls_back_to_cpu_when_no_vulkan_device
    (forces zero ICDs via VK_ICD_FILENAMES/VK_DRIVER_FILES; asserts fallback
    note, no device line, exit 0, CPU-matching score). This was the audit
    "C" gap — previously never run because every env has a device. ✓
  * Output format: same `{dssim:.8}\t{file}` as CPU; gpu_cli asserts 8
    fractional digits (F11-hardened) + ≤5e-6 parity + positive GPU signal (F7). ✓
  * README documents the experimental backend + caveats (F20). ✓
  * CI: workflow green on lavapipe (runs #4-8), now with validation + clippy
    gates (F21). ✓
- Deviated from plan: none. Note: M8 was functionally complete at Phase G
  (commit 8608e8d/ab5cc2e) but had no "M8 met" checkpoint entry until now; the
  fallback path in particular was unverified until this session's C fix.
- Blocked / open question: none.
- Next: (M9 sign-off follows.)

## 2026-09-06 — M9 SIGNED OFF (profile completed + permanent record)
- Exit-observation (plan §9 M9: "performance profile completed") — met:
  * Baseline measured: pre-Phase-H per-scale path was 4.2-15.3x SLOWER than
    CPU (overhead-bound), recorded at 1dabf06.
  * Optimized path profiled on BOTH real GPUs + 4 sizes incl 4K: discrete
    0.41-0.59x, integrated 0.46-0.57x (GPU beats CPU everywhere). create vs
    compare breakdown captured; the upload-collapse hotspot identified by
    measurement (temporary create-probe), not guessed.
  * Permanent summary now in VULKAN_PERF.md (was only transient bench output
    before — the audit "B" gap). Reproducible via
    `cargo run -p dssim-vulkan --release --example bench`.
- Honest caveats recorded in VULKAN_PERF.md: laptop CPU-time noise (±15%),
  single-run ratios (direction stable, decimals not), 4K VRAM measured at 512^2
  + extrapolated, 16-bit is 8-bit-approx.
- Deviated from plan: none. M9 work predates this; the permanent-doc + explicit
  sign-off were the missing pieces.
- Blocked / open question: none.
- Next: M0-M10 all now have explicit sign-off entries. Optional: push the
  session's unpushed commits; tag a release only on explicit instruction.
