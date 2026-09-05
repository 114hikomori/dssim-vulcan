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
