# Phase H — Consolidated Optimization Research (Perplexity · Gemini · Claude · Grok)

Date: 2026-09-07. This file merges the four per-AI "Phase H addon" research
documents (now archived under `archive/phase-h-addons/`) into one English plan:

- `OPTIMIZATION_UNTRIED_Perplexity.md`
- `Phase_H_Gemini_addon.txt`
- `PERF_PHASE_H_UNEXPLORED_Cluade.md`
- `Phase_H_Additional_Grok.txt`

Research-only: no code changed here. Scope is the Phase H optimization addons
only (the deep bug-hunt document is explicitly out of scope).

**State at handoff (post-editing, 2026-09-07):** the bug-hunt fix range
`1ca1748..f490b23` has landed and CI run #14 is green — including new gates
experimenters must respect: BH2 device-limit enforcement (workgroup/storage →
CPU fallback), BH5 a global `submit_lock` serializing all submits, BH6 an
`upload path:` log line + a CI staging leg. See §2 for what each means here.

The four addons were written around, and partly before, the Round-2
measurement pass. They disagree with each other and, in places, with the
authoritative record. This consolidation reconciles every idea against the two
source-of-truth documents — `PERF_EXPERIMENTS_PHASE_H.md` §8 (track status,
pass-4 addendum) and `VULKAN_PERF.md` "Round 2" (measured numbers) — and tags
each idea **DONE / REFUTED / REJECTED / NON-GOAL / PARKED / OPEN / FEATURE**.
Read §1 before §4: the measured state is what makes most of the addons'
headline advice already-settled, and the point of this document is to stop you
re-running closed experiments.

---

## 1. The measured state that reorders everything

Three facts from the authoritative docs, in order of how much they change the
plan:

1. **GPU compute is ~5% of create wall.** T9-lite (VkQueryPool timestamps,
   `8c40165`) on the discrete RX 6600M: at 2048² `create_image` is ~91 ms wall
   but only ~4.8 ms GPU-busy; at 4096² ~421 ms wall / ~21 ms GPU-busy
   (`VULKAN_PERF.md` Round-2 table). Every shader-side idea in the four addons
   (2D dispatch, workgroup size, fusion, subgroups, swizzling, specialization
   constants) can only touch that ~5% slice.

2. **The "prep ≈ 1 GB/s" anomaly the addons chase was already diagnosed and
   fixed.** The permanent `upload_diag` (`#[ignore]`'d) found the cost was
   **not** first-touch page faults (first-write == warm-write) and **not** the
   memory type (a cached `Vec` write was *slower* than the write-combined
   mapped write). Allocation churn was never isolated by the diagnostic — the
   inference that it doesn't matter comes from `prep` scaling with *bytes* at
   constant pass-count, not from a direct measurement. The measured cause was
   the per-pixel `PixelsIter` loop — a disguised memcpy — replaced by
   `to_contiguous_buf → copy_from_slice`. 4K create 421→338 ms (~20%), 2048²
   ~6% (`VULKAN_PERF.md` "Diagnostic"). This refutes the page-fault /
   write-combining-stall premise behind Perplexity #1, Gemini 1.1/1.2, and
   Grok F.

3. **The two big host levers already landed.** T7 zero-copy (write straight
   into a `DEVICE_LOCAL | HOST_VISIBLE` type — true UMA *or* ReBAR VRAM) is
   **DONE and measured** after the F34 `contains()` correction: discrete
   create 2048² 67 vs 91 ms, 4096² 278 vs 323 ms; both GPUs on this host take
   it. T11 `create_image_pair` (both pyramids in one submit) is **DONE**
   (320×200 full-path ratio ~0.42).

**Where the time is now** (discrete, 2048², post-memcpy + post-T7):
`create_image` wall ≈ 66 ms with only ~5 ms GPU-busy (pass-7 A/B measurement;
the 91 ms figure above is the pre-fix baseline). The gap is CPU 2×2 downsample
(~10 ms, a *standing non-goal* to move to GPU) + the host→VRAM transfer
(~89 MB pyramid at ≈2.7 GB/s effective) + command recording. The GPU path
already beats the SIMD CPU at every size (round-2 ratios ~0.36–0.51; bench now
also reports an honest `cli_ms`/`cli_ratio` column matching the CLI's
2×create+compare shape, BH15), so M10's contract is met and **all remaining
addon work is optional polish.**

The one host-side question the diagnostic did **not** close is called out in
§4 Tier 0 — it is the only large-size lever left worth an experiment.

---

## 2. Standing constraints (the gate every experiment passes)

- **Parity gate (all tracks):** `blur_parity`, `blur_dump_parity`,
  `lab_parity`, `ssim_parity`, `phase_e` green at existing tolerances
  (2e-6 map / 5e-6 score / 1e-6 Lab); locked values hit; identity exactly 0.0.
  `precise`/NoContraction sites are invariant. An optimization that changes any
  per-pixel op order is **rejected, not re-toleranced** (AGENTS.md §8). Never
  widen a tolerance to absorb a float-order change.
- **Non-goals** (AGENTS.md §8) that some addon ideas collide with — GPU-side
  downsample/pooling, GPU image decode, GPU ICC, float16, multi-GPU. These
  return only as profiling-justified, opt-in work with explicit human
  authorization, never as a default experiment.
- **Do not start with** (Perplexity "don't start" list, endorsed): a full fused
  blur before the host path is settled; a bigger workgroup judged on wall time
  alone (read the GPU timestamp); fused-ssim benchmark numbers as a direct
  prediction (different algorithm/baseline/precision); tolerance-widening.
- **Gates added by the bug-hunt fixes (HEAD `f490b23`):**
  - BH2: `Context` now enforces `maxComputeWorkGroupCount` /
    `maxStorageBufferRange` (dispatch-time Err + CLI pre-check fallback).
    Any Tier-3 dispatch-geometry change must keep `supports_size` honest, and
    experiments on min-spec devices will now *error* where they previously
    did UB.
  - BH5: a global `submit_lock` makes every `dispatch_sequence` atomic
    (pool + queue + query pool under one mutex). This is CORRECTNESS first —
    any H22-style async overlap must deliberately redesign this lock
    (per-queue locks + cross-queue semaphores), not just "submit on another
    thread" and expect it to run concurrently.
  - BH6: the effective upload path is printed (`upload path: zero-copy|
    staging`) and CI has a `DSSIM_UNIFIED=0` staging leg — Tier-0/2 harnesses
    can ASSERT which path ran instead of assuming it.
  - Stable-signal rule (pass-6 observation): compare via the `*_gpu` timestamp
    columns and peak-alloc counters; wall ratios carry ±15% laptop noise.

---

## 3. Status ledger — every addon idea, deduplicated

Merged across the four files; the "Source(s)" column shows which AI raised it.
Status is against §1's authoritative record.

| # | Idea (merged) | Source(s) | Status | Basis / note |
|---|---|---|---|---|
| H1 | Persistent staging ring + scratch arena (page-fault/alloc-churn premise) | Perplexity 1, Grok F | **REFUTED** | Diagnostic: not page faults, not churn; loop was the cause (fixed). Ring re-rejected. |
| H2 | Non-temporal streaming stores to fix WC stall | Gemini 1.1 | **REFUTED (premise)** | WC stall was the scalar loop, now contiguous; WC "absorbs it". Residual → Tier 0. |
| H3 | `HOST_CACHED` staging instead of direct VRAM write | Gemini 1.2 | **REFUTED (naive) / MOSTLY-MEASURED (DMA)** | Cached-Vec write measured *slower* than WC. The same-queue DMA variant was measured by the F34 A/B (staging lost 87 vs 66 ms @2048²); only NT-stores and dedicated-queue SDMA remain open → Tier 0. |
| H4 | Zero-copy for UMA / ReBAR host-visible (`contains()`, not OR) | Perplexity 2 | **DONE** | T7, F34-corrected, A/B-measured on discrete too. |
| H5 | Import prep buffer via `VK_EXT_external_memory_host` | Claude T13 | **OPEN (capability-gated)** | Different mechanism from T7; marginal on this host (already direct). One API query decides. |
| H6 | GPU-side downsample + color (upload level-0 only) | Gemini 1.3 | **NON-GOAL** | Conflicts with standing non-goal (VULKAN_PERF L55; AGENTS §8). Biggest theoretical upside; needs human authorization. |
| H7 | Descriptor pre-alloc + dynamic offsets (T3) | Perplexity 3 | **OPEN (diminishing)** | Not tried; CPU-side, dwarfed by transfer+downsample floor at large sizes; small-size only. |
| H8 | `descriptor_buffer` / push-descriptor / update-template | Gemini 3.3, Grok C | **OPEN (diminishing)** | Modern replacement for H7; same ceiling. |
| H9 | `VK_EXT_descriptor_heap` (supersedes H8) | Claude T15 | **OPEN (capability-gated)** | Ratified 2026; AMD Windows + RADV 26.2. Tooling-maturity caveat. |
| H10 | Tighten pipeline barriers (T10) | Perplexity 5 | **OPEN (diminishing)** | Over-barriered by design (~40/create); needs timestamps to prove win. |
| H11 | Lazy/deferred barriers (llama.cpp pattern) | Grok E | **OPEN (diminishing)** | Concrete algorithm for H10; medium race risk, GPU-AV gate. |
| H12 | DCC decompress bubbles + split barriers / events | Gemini 2.3 | **LIKELY N/A (buffers)** | DCC applies to images/render-targets; our pipeline is storage-buffers only (same reasoning as H21). Check in RGP only if Tier-2 barrier work underperforms expectations. |
| H13 | Timeline semaphore / fence ring (T4) | Perplexity 4 | **DEMOTE** | Premise (120 fences) gone; ~1–2 submits left; residual ~0.1 ms. |
| H14 | 2D dispatch + workgroup tuning (T5) | Perplexity 6 | **REFUTED (low-value)** | GPU-busy ~5%; touches ~2% of wall. Parked behind "GPU dominates". |
| H15 | Thread-group ID swizzling for L2 locality | Grok D | **PARKED** | Only after H14 and only when GPU-bound; 47% figure is a full-screen denoiser, not us. |
| H16 | Specialization constants / pipeline variants | Perplexity 8, Grok A | **PARKED** | GPU-side; low FP risk; pipeline-count cost; only pays once GPU-bound. |
| H17 | Subgroup shuffle / arithmetic for blur & reductions | Gemini 2.2, Grok B, Claude T16 | **PARKED** | 0% parity for shuffle (register exchange, same op order); reduction-tree changes are medium risk. GPU-side. |
| H18 | Shared-memory fusion (H5+V5; blur→combine) (T6) | Perplexity 7 | **REFUTED (low-value now)** | T6a refuted by T9-lite; highest upside but highest parity risk; bitwise gate. |
| H19 | Safe mu/sq input-read fusion (`h5(img)`+`h5_mul(img,img)`) | Perplexity 7, PERF §8.3 | **PARKED** | Parity-safe by construction (halves img read traffic); still GPU-side → behind "GPU dominates". |
| H20 | Rastergrid linear-sampling separable blur | Gemini 2.1 | **REJECTED** | Hardware sampler is fixed-point subtexel → breaks FP32 parity gate. |
| H21 | `VK_EXT_host_image_copy` | Gemini 3.1, Claude T14 | **OPEN (inventory-gated)** | Only helps image targets; pipeline uses **buffers** (tmp/mu/sq/cross) → likely N/A. Grep first. |
| H22 | Dedicated async transfer queue (DMA overlap) | Gemini 3.2 | **OPEN / FEATURE** | On this host zero-copy already writes direct; DMA path matters with H3-Tier0 and with batch (H23). RADV impl is Linux-only + flag-gated. |
| H23 | Batch mode / resident reference (T8) | Perplexity 9 | **FEATURE** | Needs CLI/API (1-vs-N); amortizes fixed cost; enables H22 overlap. |
| H24 | Pipeline-cache / `pipeline_binary` control | Grok G | **DETAIL** | Negligible now (pipelines built once); load-bearing only if H16 multiplies variants. |

Also checked-and-nothing-new (Claude §3): gpu-allocator is still 0.28.0 (the
version already cited); no newer arena scheme exists to adopt.

---

## 4. Priority ladder for effective experimentation

**Batching principle (the "not one-by-one" answer):** run each tier as one
matrix against a single shared harness, record one comparison table, and make
one go/stop decision — do not dribble isolated micro-tests. Tiers are ordered by
expected impact against the §1 profile; later tiers are explicitly gated on
earlier outcomes.

### Tier 0 — Settle the only open host-side question (do first, one harness)
The diagnostic fixed the loop but asserted the remaining ~24 ms/64 MB
(≈2.7 GB/s) write-combined host→VRAM transfer is a "hardware floor". Two
corrections to the addons' framing before spending effort:
1. **The plain DMA dataflow is NOT unmeasured.** The F34 A/B already compared
   zero-copy vs forced staging (`DSSIM_UNIFIED=0`) — and staging IS
   "CPU→cached host RAM, then copy→VRAM" (same-queue `vkCmdCopyBuffer`): it
   LOST, 87 vs 66 ms at 2048². What has genuinely never been probed is (c)
   NT/streaming stores into the mapped BAR pointer and (d) a DEDICATED
   transfer-queue (SDMA) copy + overlap. Set the prior accordingly: variant
   (b) is expected to lose again; the experiment's real question is (c)/(d).
2. **(d) collides with BH5's `submit_lock`** (see §2): overlap requires a
   deliberate lock redesign + cross-queue semaphore chain, which is a medium
   change to freshly-hardened plumbing — only green-light it if (c) alone
   fails and the projected win is large.
- **Harness:** extend the existing `upload_diag` + `DSSIM_UNIFIED` A/B seam;
  assert the path via the BH6 log line. Measure at 2048²/4096²:
  (a) current zero-copy BAR write, (b) forced staging [prior: loses ~20 ms],
  (c) NT-store microbench into the mapped pointer (Claude T12),
  (d) only if (c) is promising: dedicated-transfer-queue DMA of the same bytes.
- **Decision gate:** if (c) or (d) beats ~2.7 GB/s materially, restructure the
  discrete upload to it (pure host-side, 0% parity risk). If not, the floor is
  confirmed, host prep is **done**, and Tiers 2–4 stay parked — stop optimizing
  large-size wall time.
- **Why first:** it is the only lever that touches the ~95% of wall that is not
  GPU compute, and it is cheap because the instrumentation already exists.

### Tier 1 — Capability probes (one query / one grep each; admit-or-drop)
Batch into a single `vulkaninfo` + grep pass; no pipeline change until a probe
passes.
- H5 `external_memory_host`: `vkGetMemoryHostPointerPropertiesEXT` on the real
  prep allocation → non-empty type mask? (Claude's exit bar.)
- H9 `descriptor_heap`: `descriptorHeap` feature bit on the Windows AMD driver?
- H21 `host_image_copy`: grep the tree for image-backed upload targets — the
  comparison pyramid is **buffers**, so expect N/A; only if a real `VkImage`
  target exists, check `HOST_IMAGE_TRANSFER` format feature.
- **Gate:** each is a yes/no that costs minutes; a "no" closes the idea for this
  hardware without any implementation risk.

### Tier 2 — Small-size CPU overhead (only if 320×200 latency matters)
At large sizes these are below the transfer+downsample floor; they only move the
needle on the fixed-cost-dominated small-image path. One CPU-trace harness
(RGP/Nsight), run as a set:
- H7/H8/H9 descriptor path (target: zero `vkUpdateDescriptorSets` in steady
  state), H10/H11/H12 barrier tightening + lazy emission + DCC check (target:
  fewer full barriers, no RAW races under GPU-AV), H24 pipeline-cache hygiene.
- **Gate:** parity suites green + a measured small-size delta; revert if the
  delta is inside the ±15% wall noise.

### Tier 3 — PARKED behind "GPU dominates" (do NOT start now)
H14, H15, H16, H17, H18, H19 are all shader/compute changes that can only touch
the ~5% GPU-busy slice. Revisit **only if** Tier 0/2 drop host time enough that
GPU-busy becomes the majority of wall. When opened, run them as one GPU-timestamp
+ bitwise-parity matrix, in this order: H14 (2D) → H15 (swizzle) → H16 (spec
constants) → H17 (subgroup) → H19 (safe mu/sq fusion) → H18 (full fusion). Every
one carries the bitwise-map gate; H17/H18 additionally must not change reduction
order.

### Tier 4 — Feature-level (needs a product decision + CLI)
H23 batch/resident-reference, which is what makes H22 async-transfer overlap
meaningful (independent work to hide behind). Not a micro-optimization; scope it
only if a real 1-vs-N workflow exists.

### Tier 5 — Needs explicit human authorization (non-goal)
H6 GPU-side downsample + color conversion. Highest theoretical upside (cut PCIe
bytes >75%, drop CPU prep to sub-ms) but it reverses a documented non-goal and
carries low-to-medium parity risk (must reproduce the CPU downsample/Lab op
order exactly). Do not begin without the operator's word.

### Rejected / refuted — do not spend cycles
H1, H2, H3-naive, H13, H14-now, H18-now, H20, and Grok's left-out set
(cooperative matrices/tensor, vendor image extensions, full bindless, FP16
intermediates, async queue for a *single* compare). Each is closed by §1 or the
parity gate; re-listing them here is so the next pass does not rediscover them
as if new.

---

## 5. Per-technique detail (open / parked items only)

Compact, deduplicated. DONE/REFUTED/REJECTED items are fully covered by §3 and
§1; not repeated here.

**H3-DMA / H22 — beat the BAR-write floor (Tier 0).** Mechanism: CPU writes the
pyramid into cached host-visible system RAM (fast, cache-backed), then the GPU
copy engine (`vkCmdCopyBuffer`, ideally on a dedicated `VK_QUEUE_TRANSFER_BIT`
queue) DMAs it to VRAM at near link bandwidth, optionally overlapped with
compute. Parity risk 0% (transport only). **Prior: the same-queue form of this
already lost to zero-copy in the F34 A/B (87 vs 66 ms at 2048²)** — the
untested forms are NT/streaming stores into the mapped pointer and a
DEDICATED transfer queue with overlap. The overlap form additionally requires
redesigning BH5's global `submit_lock` (per-queue locks + cross-queue
semaphores) — a medium change to freshly-hardened plumbing; gate it behind a
positive NT-store result. Difficulty low-medium (NT probe) / medium-hard
(SDMA+lock redesign). Exit: measured GB/s vs the ~2.7 GB/s zero-copy floor at
2048²/4096². Caveat: on this host both GPUs currently take zero-copy, so
staging must be forced (`DSSIM_UNIFIED=0`) to exercise it; RADV's SDMA
transfer queue is Linux-only and flag-gated, so the Windows AMD driver is the
one to verify.

**H5 — `VK_EXT_external_memory_host` (Tier 1).** Import an application-owned
prep allocation as `VkDeviceMemory`; keep writing through your own pointer; GPU
reads the same bytes, no copy step. Distinct from T7 (which finds a
Vulkan-allocated both-flags type). Alignment to
`minImportedHostPointerAlignment`; ownership/sync stays yours; can fail
per-platform (`ERROR_INVALID_EXTERNAL_HANDLE`), so needs a capability check +
fallback. Marginal on this host (already direct via ReBAR) but may apply where
T7's intersection is absent.

**H7/H8/H9 — descriptor overhead (Tier 2).** Classic: pre-allocate sets per pass
slot or use dynamic storage-buffer offsets into one arena (respect
`minStorageBufferOffsetAlignment`), keep push constants. Modern:
`descriptor_buffer`/push-descriptor/update-template (update = `memcpy`), or
`descriptor_heap` (ratified 2026; removes pools/sets entirely; AMD Windows +
RADV 26.2; deps `maintenance5` + `buffer_device_address` +
`shader_untyped_pointers`; validation/tooling maturity is the risk). All target
CPU driver time that is already small at large sizes — value is at 320×200.

**H10/H11/H12 — barriers (Tier 2).** `dispatch_sequence` is deliberately
over-barriered (~40 full barriers/create). Tighten src/dst stage + access masks
(COMPUTE→COMPUTE for intermediates, TRANSFER only at copy); consider lazy/deferred
collection (emit one combined barrier only at a true RAW edge) and split
barriers / `SetEvent`+`WaitEvents` to let the driver overlap. Check RGP for AMD
DCC implicit-decompress bubbles from wide barriers / layout churn. Never remove
a barrier on a real write→read edge; validate with GPU-AV + parity.

**H14/H15/H16/H17/H18/H19 — compute (Tier 3, parked).** 2D dispatch removes
per-invocation div/mod (pure indexing, zero FP risk, but re-verify blob
freshness + NoContraction counts); swizzle workgroup IDs for L2 locality;
specialization constants fold width/height/stride/local-size at pipeline-creation
(LocalSizeId core 1.3) at a pipeline-count cost; subgroup shuffle moves taps
through the register file (0% parity, no LDS/barrier) while subgroup *arithmetic*
changes reduction order (medium risk); shared-memory fusion cuts global traffic
3–5× but is the highest transcription risk (sigma cancellation) — bitwise gate;
the safe first step is H19 (fuse the shared `img` read for mu and sq,
per-pixel-expression-identical).

**H21 — `VK_EXT_host_image_copy` (Tier 1).** `vkCopyMemoryToImageEXT` writes host
memory straight into a `VkImage` (no staging/CB/submit); core-optional in 1.4;
RADV enabled it by default for RDNA2+ with an AVX2 ADDRLIB swizzle path (~20 GiB/s,
single-sourced). Only relevant if an upload target is image-shaped — the pyramid
uses buffers, so expect N/A; grep before scoping.

**H23 — batch / resident reference (Tier 4).** Keep the original's GPU-resident
pyramid once, compare N modifieds; amortizes create/upload and enables H22
overlap. Feature-level: no CLI/API for multi-mod today.

**H6 — GPU-side downsample + color (Tier 5, non-goal).** Upload only level-0
RGBA, do sRGB→Lab and the pyramid in compute (or `vkCmdBlitImage` if a sampler is
acceptable — it is not, parity-wise). Cuts PCIe bytes >75% and CPU prep to
sub-ms. Reverses a standing non-goal; needs op-order-exact shaders and the
operator's authorization.

---

## 6. Reconciled experiment schedule

| Batch | Items (shared harness) | Harness | Decision gate |
|---|---|---|---|
| B0 | Claude-T12 NT-store microbench first; SDMA (H3-DMA/H22) only if it wins | extend `upload_diag` + `DSSIM_UNIFIED` A/B; assert path via BH6 log | beat ~2.7 GB/s? (prior: plain staging DMA already lost 87 vs 66) if no → host prep done, stop large-size work |
| B1 | H5, H9, H21 capability probes | one `vulkaninfo` + one grep | per-item yes/no; "no" closes it, no code |
| B2 | H7/H8, H10/H11/H12, H24 | one RGP/Nsight CPU trace | small-size delta > noise + parity green |
| B3 | H14→H15→H16→H17→H19→H18 | GPU-timestamp + bitwise-parity matrix | only if GPU-busy > ~50% wall; bitwise-equal maps before any bench claim |
| B4 | H23 (+H22 overlap) | bench.rs batch column {1,5,10} | honest table; ≥1× vs CPU at 2048² batch-10 |
| B5 | H6 | — | human authorization first (non-goal) |

---

## 7. Evidence base (merged)

Authoritative (project): `PERF_EXPERIMENTS_PHASE_H.md` §8 + pass-4 addendum;
`VULKAN_PERF.md` "Round 2" (T9-lite, Diagnostic, T7/F34, T11); `CHECKPOINT.md`
latest entries. Original S1–S12 live in `PERF_EXPERIMENTS_PHASE_H.md` §3.

Added by the four addons (opened 2026-09-07):
- **S13** ryg, *Write combining is not your friend* — WC buffering, implicit-read
  pitfall, `MOVNTDQA`.
- **S14** Mechanical Sympathy, *Write Combining* — fixed per-core WC/line-fill
  buffers.
- **S15** NVIDIA forums, WC vs pinned bandwidth (chipset-dependent; directional).
- **S16/S17** Khronos `VK_EXT_host_image_copy` ref page + *Copying Images on the
  Host* blog (`MEMCPY` fast path).
- **S18** Phoronix, RADV host-image-copy default for RDNA2+ (~20 GiB/s, single-sourced).
- **S19** Khronos `VK_EXT_external_memory_host` man page.
- **S20** Khronos, *Vulkan Roadmap 2026 + Descriptor Heap* (`VK_EXT_descriptor_heap`).
- **S21** AMD Adrenalin release notes — `descriptor_heap` on Windows AMD driver.
- **S22** Phoronix — RADV descriptor heap (Mesa 26.1 flag → 26.2 default).
- **S23** Khronos Vulkan Guide, *Subgroups* (`subgroupAdd`, dynamic size caveat).
- **S24** Khronos forum compute-reduction thread (illustrative only).
- **S25** Mesa MR — RADV dedicated transfer queue (SDMA; GFX9/10/10.3/11;
  `RADV_PERFTEST=transfer_queue`, Linux-only).
- Grok: NVIDIA Vulkan Dos&Don'ts (2025), Khronos specialization-constants sample,
  Khronos subgroup tutorial, Maister descriptor-buffer/heap series, NVIDIA
  thread-group swizzling (47% L2 figure), Vulkanised 2026 ggml/llama.cpp
  (barrier deferral + fusion), Arm GPU best practices, `VK_KHR_pipeline_binary`
  blog, RDNA4 specialization write-up, llama.cpp Vulkan PR history.
- Gemini: Rastergrid linear-sampling blur (link 404 in original pass; claim
  retained only to record the parity rejection), Zeux *Writing an efficient
  Vulkan renderer* (DCC decompress, split barriers).

---

## 8. Caveats & could-not-verify (merged + reconciliation notes)

- **Reconciliation is load-bearing here.** The four addons partly assume the
  prep bottleneck is page-faults / write-combining / memory-type selection and
  that zero-copy is untried. The authoritative Round-2 record shows those are
  fixed/done/refuted; §1 and §3 correct them. Do not treat an addon's
  "recommended priority #1" as live without checking §3's status column.
- **Claude opened no source code** (its §5): its buffer-vs-image question for
  H21 is resolved here — the comparison pyramid is buffer-backed, so
  `host_image_copy` is likely N/A. Its WC-buffer-count and CPU-model unknowns
  remain (qualitative mechanism transfers; the specific per-core buffer count
  and whether the *remaining* transfer is truly the floor are what Tier 0
  measures).
- **Whether the Windows AMD driver exposes `external_memory_host` /
  `host_image_copy`** was not found in AMD's notes (only `descriptor_heap` was);
  run `vulkaninfo` (Tier 1) before scoping H5/H21.
- **Single-sourced numbers** (Claude convention, kept): the ~20 GiB/s RADV
  swizzle figure (S18) and the 47% swizzle gain (Grok D) are directional, not
  predictions for this pipeline.
- **fused-ssim / IQM** are ideas-only (no license file found upstream); do not
  copy code (AGENTS.md §10). Their benchmark numbers are against a different
  algorithm and an unoptimized CPU baseline.
- **Laptop thermal noise** ±15% on CPU times: any Tier 2/3 delta must exceed
  noise or it is not a win; GPU-busy timestamps are the stable signal.
- **Nothing was executed for this consolidation** beyond reading the four addons
  and the two authoritative docs + checkpoint; every status tag cites one of
  those. No code, shader, or config was changed.
