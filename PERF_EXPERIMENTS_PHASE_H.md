# Phase H Performance Experiments — Research & Plan

Date: 2026-09-06. Companion to `AUDIT_M7_M9_PHASE_H.md`; feeds the "Phase H
continuation — optimization" line in `CHECKPOINT.md`. Research-only document:
no code changed here. Every technique below cites a source opened on
2026-09-06 (S-numbers in §3); project-specific numbers cite our own code/HEAD.

## 1. Baseline being attacked (measured, M9)

`dssim-vulkan/examples/bench.rs`, RX 6600M, release, compare-only:

| image | CPU ms | GPU ms | ratio |
|---|---|---|---|
| 320x200 | 4.86 | 74.34 | 15.3x |
| 1024x1024 | 71.59 | 373.30 | 5.2x |
| 2048x2048 | 307.45 | 1291.28 | 4.2x |

Diagnosis (from code, HEAD `acf9fca`): the GPU path is **submit-bound**, not
compute-bound. One `blur()` call = upload (1 submit+fence) + dispatch_sequence
(1 submit+fence) + download (1 submit+fence) + 3 fresh buffer allocations +
descriptor pool reset. One `blur_mul()` = 4 submits. Per 3-channel scale,
`create_image` ≈ 34 submits; `compare` ≈ 19/scale → **~120+ submits and fence
round-trips per single comparison**. Each fence round-trip costs ~0.1–1 ms of
driver + PCIe latency; that alone explains the 74 ms at 320x200 where the
actual math is sub-millisecond.

This matches the industry diagnosis exactly: "Calling vkQueueSubmit frequently
with small amounts of work is a major performance killer. Each submission has
a high fixed cost. Batch as many command buffers as possible into a single
submission." (S1) and "Try to minimize the number of queue submissions. Each
vkQueueSubmit() has a significant performance cost on CPU" + "Don't submit a
small amount of GPU work" + "Don't record tiny command buffers" (S2).

## 2. Strategy

Three levers, in order of expected impact on our numbers:

1. **Amortize** — one submit per scale (or per comparison) instead of ~50;
   keep intermediates GPU-resident; upload each image once.
2. **Pipeline** — stop waiting on every fence; async readback + timeline
   semaphores so CPU f64 pooling overlaps the next scale's GPU work.
3. **Compute better** — only once 1–2 are done does shader-level work matter
   (2D dispatch, shared-memory tiling, fused passes). Doing 3 first is
   profiling theater.

## 3. Evidence base (sources opened 2026-09-06)

- **S1** Khronos Vulkan-Guide, *Profiling* chapter (2026). Submission batching,
  "avoid vkDeviceWaitIdle in your main loop… use a ring buffer of fences",
  "vkUpdateDescriptorSets is a heavyweight CPU operation", restrictive-barrier
  bubbles, GPU timestamp queries via VkQueryPool, tooling (RGP/Nsight/VKtracer),
  CI perf-regression pattern (GFXReconstruct).
  https://raw.githubusercontent.com/KhronosGroup/Vulkan-Guide/main/chapters/profiling.adoc
- **S2** NVIDIA, *Tips and Tricks: Vulkan Dos and Don'ts* (2019, updated
  2025-01). Minimize submissions; reuse command buffers/pools ("Don't create or
  destroy command pools"); "Use memory sub-allocation. vkAllocateMemory() is an
  expensive operation on the CPU"; host-visible video memory for direct writes
  (DEVICE_LOCAL|HOST_VISIBLE); "Minimize the use of barriers… Group barriers in
  one call"; "Use push constants for per draw call updates"; "Use dynamic
  uniform/storage buffers"; "Do not test performance with validation layers
  enabled"; lock GPU clocks for stable measurement.
  https://developer.nvidia.com/blog/vulkan-dos-donts/
- **S3** AMD GPUOpen, *Vulkan Barriers Explained* (2016). Tight src/dst stage
  masks; produce early/wait late; avoid read→read barriers.
  https://gpuopen.com/learn/vulkan-barriers-explained/
- **S4** Khronos blog, *Vulkan Timeline Semaphores* (2020; core since 1.2).
  Single primitive for device↔host sync, wait-before-signal, "reducing
  host-side stalls"; WG "highly encourages… for all coarse-grained
  synchronization". https://www.khronos.org/blog/vulkan-timeline-semaphores
- **S5** Khronos Vulkan-Guide, *Compute Shaders* chapter (2026). Shared memory
  = "the L1 cache you can control… an important part of any performant shader";
  limits (maxComputeWorkGroupInvocations ~1024, shared ~32 KB); built-ins incl.
  WorkgroupId/LocalInvocationId (2D dispatch); dispatchIndirect; shared-memory
  race detection via GPU-AV.
  https://raw.githubusercontent.com/KhronosGroup/Vulkan-Guide/main/chapters/compute_shaders.adoc
- **S6** Khronos Vulkan-Guide, *Memory Allocation* chapter. Sub-allocation is
  "first-class"; OS-level alloc/dealloc "really slow"; maxMemoryAllocationCount;
  discrete GPUs: PCIe transfer is the bottleneck, dedicated
  VK_QUEUE_TRANSFER_BIT queues exist; **UMA/integrated: no staging needed,
  "transfer overhead is greatly reduced"**.
  https://raw.githubusercontent.com/KhronosGroup/Vulkan-Guide/main/chapters/memory_allocation.adoc
- **S7** Khronos Vulkan-Guide, *Push Constants* chapter. Push constants avoid
  "creating buffers or modifying and binding descriptor sets for each update";
  incremental updates within a command buffer are legal (per-pass push in one
  CB — already our pattern).
  https://raw.githubusercontent.com/KhronosGroup/Vulkan-Guide/main/chapters/push_constants.adoc
- **S8** Khronos Vulkan-Guide, *Descriptor Dynamic Offset* chapter. Dynamic
  storage-buffer offsets rebind a descriptor at record time without
  vkUpdateDescriptorSets; respect minStorageBufferOffsetAlignment.
  https://raw.githubusercontent.com/KhronosGroup/Vulkan-Guide/main/chapters/descriptor_dynamic_offset.adoc
- **S9** Vulkan spec man pages: `vkFlushMappedMemoryRanges` (non-coherent host
  writes must be flushed; "Unmapping… does not implicitly flush") and
  `vkQueueSubmit` (batch semantics; ONE_TIME_SUBMIT CBs go invalid after
  execution → reuse needs reset; fence ordering).
  https://registry.khronos.org/vulkan/specs/latest/man/html/vkFlushMappedMemoryRanges.html
  https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit.html
- **S10** gpu-allocator crate README (v0.28, current; Traverse-Research fork —
  the one we depend on). `AllocationScheme::GpuAllocatorManaged` vs
  `GpuVmaAllocation` (explicit sub-allocation control); memory visualizer tool.
  https://raw.githubusercontent.com/Traverse-Research/gpu-allocator/main/README.md
- **S11** cadik/IQM (2025) — Vulkan-compute image-quality library (SSIM, FSIM,
  FLIP, PSNR, LPIPS). Structure: separable Gaussian H-pass → temp image →
  V-pass → combine, 16x16 2D workgroups, r32f storage images, push constants.
  https://github.com/cadik/IQM
- **S12** rahul-goel/fused-ssim (SIGGRAPH Asia 2024 paper repo) +
  LaurensDiels/FastCUDASSIM.jl benchmark table. Fully-fused SSIM: "Convolutions
  in SSIM are spatially localized leading to fully-fused implementation without
  touching global memory for intermediate steps… Single convolution pass for
  multiple statistics"; 5–8x faster than pytorch-msssim; full-HD 3-ch SSIM
  0.27–0.87 ms GPU vs 242–556 ms CPU reference.
  https://github.com/rahul-goel/fused-ssim
  https://github.com/LaurensDiels/FastCUDASSIM.jl

## 4. Experiment tracks

Each track lists: change → expected effect → parity risk → exit observation.
**Global parity gate (all tracks):** blur_parity, blur_dump_parity, lab_parity,
ssim_parity, phase_e green at existing tolerances (2e-6 map / 5e-6 score /
1e-6 Lab), locked values hit, identity exactly 0.0. `precise`/NoContraction
sites are invariant — an optimization that changes any per-pixel op order is
rejected, not re-toleranced (AGENTS.md §8).

### T1 — One submit per scale (orchestration) — highest impact
Replace per-call `dispatch_sequence` submits with a per-scale (or
per-comparison) command buffer: record Lab + all channel blurs + mu/sq +
cross + combine as passes in ONE submit with barriers between dependent passes
(mechanism already exists: `dispatch_sequence` takes a pass list; the problem
is that every `blur()`/`blur_mul()`/`combine` call wraps its own submit +
upload + download). Intermediates (tmp, mu, sq, cross) become GPU-resident
buffers; only the SSIM map downloads.
Expected: submits per comparison ~120 → ~6–10. At 320x200 this is the whole
game (74 ms → low ms range).
Risk: barrier correctness (every write→read edge needs the barrier; S3: keep
stage masks tight — COMPUTE→COMPUTE for intermediates, TRANSFER only at the
final copy). Descriptor pool cap 64 (audit F5) must be raised or sets
pre-allocated per pass slot.
Exit: bench.rs 320x200 GPU ≤ 15 ms; parity suites green; validation clean.

### T2 — Persistent staging ring + buffer reuse (memory)
One large host-visible staging buffer (ring) reused across uploads instead of
alloc+map+free per call; device-local scratch arena sub-allocated per scale
(S2 "sub-allocation", S6 "first-class", S10: gpu-allocator supports explicit
schemes; consider `AllocationScheme::GpuVmaAllocation` for arena control).
Upload each image's RGBAPLU once per comparison, not once per scale.
Expected: removes ~40 alloc/free + map cycles per comparison; biggest win at
2048x2048 (134 MB of per-scale re-uploads collapse to one).
Risk: ring wraparound ordering (flush before reuse — S9); non-coherent memory
flush/invalidate (also fixes audit F1 for free).
Exit: gpu-allocator visualizer shows bounded allocation count; bench table
improves; parity green.

### T3 — Descriptor & pipeline binding overhead
Pre-allocate descriptor sets per pass slot (or dynamic storage-buffer offsets
into the T2 arena — S8) instead of allocate+reset per dispatch_sequence (S1:
descriptor updates are heavyweight CPU work). Keep push constants (already
best-practice per S7/S2).
Expected: removes per-pass CPU driver calls; matters once T1 makes the CPU
record path the bottleneck.
Risk: low (layout-compatible pipeline rules — S7 lifetime section).
Exit: Nsight/RGP CPU trace shows no vkUpdateDescriptorSets in steady state.

### T4 — Async readback + timeline semaphores
Replace per-submit fence-wait with a timeline semaphore (core 1.2; we request
1.3) or a fence ring (S1): submit scale N+1's work while CPU pools scale N's
map (S4: "reducing host-side stalls"; S2: "Don't wait for a queue submission
to finish, continue preparing the next").
Expected: hides remaining fence latency; overlaps CPU f64 pooling + CPU
downsample (which today serializes with GPU).
Risk: medium — reorders host logic; determinism must survive (same submit
order every run; dumps/locked values are the guard).
Exit: bench 1024x1024 GPU < 100 ms; double-run byte-identical maps.

### T5 — Dispatch geometry & workgroup tuning
Switch 1D `idx = gl_GlobalInvocationID.x; y=idx/w; x=idx%w` to 2D dispatch
(16x16, IQM pattern S11) — removes per-invocation integer div/mod; x/y come
from built-ins (S5). Raise workgroup size from 64 toward 256 (S5 limits).
Optionally batch the 3 channels along `gl_GlobalInvocationID.z` so one
dispatch does L+a+b.
Expected: 10–30% of pure GPU time at 2048x2048; zero at small sizes (that's
T1's territory).
Risk: none to FP semantics (pure indexing change) — but recompile blobs and
re-verify NoContraction counts + blob freshness (audit process).
Exit: identical parity results with 2D shaders; bench delta recorded.

### T6 — Fused passes (the fused-ssim idea, adapted)
Fuse H5+V5 into one kernel via shared-memory tile (S5: shared memory is the
controllable L1; S12: "without touching global memory for intermediate
steps"). Parity-safe **iff** the H5 values in shared memory are the same f32s
the current two-pass writes (they are — same per-pixel expression, same
`precise` sites) and boundary taps are handled per the existing 4-case logic.
Longer-term: fuse blur→combine per scale (one kernel computes mu, sq, cross,
SSIM for its tile) — this is the biggest compute win but the biggest
transcription risk.
Expected: at 2048x2048, global traffic drops ~3–5x (no tmp round-trips);
GPU compute time approaches memory-bound.
Risk: HIGH for parity (sigma cancellation amplifies any order change — M3/M7
history). Gate: bitwise-equal maps vs current shaders on all fixtures before
any bench claim.
Exit: map bitwise parity vs T5 shaders on all devices; bench table updated.

### T7 — UMA fast path (integrated GPU / CI)
On UMA devices (our integrated Radeon; llvmpipe in CI), skip staging entirely:
allocate host-visible device-local and write directly (S6: "no need to create
a staging buffer"; S2: host-visible video memory). Detect via memory-type
flags at context creation.
Expected: removes upload latency on the integrated leg and on CI; small win
on discrete.
Risk: low; keep discrete path unchanged.
Exit: integrated-device bench improves; CI runtime drops.

### T8 — Batch mode (CLI-level amortization)
`dssim --gpu original.png mod1..modN`: keep the original's GPU-resident
pyramid once (T1/T2 make this natural), pipeline pairs (T4). This is where
GPU can actually beat the SIMD CPU path per the M9 ratio trend (4.2x at
2048² and shrinking with size).
Expected: N-image batches amortize fixed costs ~N-fold; target: ≥1x vs CPU
at 2048² batch-10 (measure, don't claim).
Risk: low (orchestration only).
Exit: bench.rs extended with a batch column; honest table in CHECKPOINT.

### T9 — Measurement infrastructure (do FIRST)
- GPU timestamps per stage via VkQueryPool (S1) so "overhead-bound" becomes a
  per-submit number, not an inference.
- AMD Radeon GPU Profiler (RGP) capture of one compare (S1 tooling; we're on
  AMD) — count submits, fence waits, bubbles.
- Lock clocks during bench (S2) — or report variance across 3 runs instead.
- Keep validation off in release bench (already true: debug-only, S2 warns).
- Extend bench.rs: batch sizes {1, 5, 10}, sizes {320x200, 1024², 2048²},
  per-stage GPU-time breakdown column.
Exit: a table that attributes the 74 ms at 320x200 to named causes; every
later track's delta is measured against it.

## 5. Priority

T9 → T1 → T2 → T3 → T4 → T5 → T7 → T8 → T6.
(T6 last: highest compute upside but highest parity risk; only worth it once
orchestration overhead is gone and GPU time actually dominates.)

## 6. Honest expectations

- Small single pairs (320x200): CPU is 4.9 ms; GPU will likely stay slower
  even after T1–T5 — PCIe round-trips + launch latency are fixed costs. The
  goal there is "same order of magnitude", not "faster".
- Large images and batches: this is where GPU wins (S12 shows fused GPU SSIM
  beating CPU SSIM by ~100–600x — but against *unoptimized* CPU references;
  our CPU baseline is SIMD-optimized, so expect single-digit-x at best, and
  treat any number below as a hypothesis to measure, never a target to hit).
- The M9 table is the contract: every track must move it, or it's reverted.

## 7. Could-not-verify (this session)

- Arm "Vulkan Best Practices for Mobile Developers" PDF and Rastergrid's
  separable-blur article: URLs 404'd; not cited (their claims are covered by
  S1/S2/S5 anyway).
- IQM and fused-ssim declare **no license file** at the paths checked
  (LICENSE/LICENSE.md absent; pyproject/CMakeLists silent) → treat both as
  **ideas-only references; do not copy code** without resolving licensing
  (AGENTS.md §10).
- fused-ssim/FastCUDASSIM benchmark numbers are their own, against an 11x11
  Gaussian SSIM (different algorithm from DSSIM's fused 5-tap pyramid) —
  directional evidence only, not a prediction for us.
- No GPU work exists upstream in kornelski/dssim (issue/PR search: 0 hits) —
  no prior art to inherit there.
- CI lavapipe timing for the new (unpushed) Phase H blobs: unknown until push
  (audit F24).
