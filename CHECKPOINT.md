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
