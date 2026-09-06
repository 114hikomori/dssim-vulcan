# Vulkan backend — performance profile (M9)

Measured with `cargo run -p dssim-vulkan --release --example bench`
(add `DSSIM_BENCH_DEVICE=<candidate index>` to target a non-default GPU).
Times the **full score path** — `create_image` (pyramid build) + `compare` —
for both CPU (dssim-core, all cores) and GPU, excluding only image decode
(identical on both paths in real use). Warmup + timed iterations, single run
per size (TDR policy — no stress loops). Ratio = gpu_ms / cpu_ms; **< 1.0 means
the GPU path beats the CPU baseline.**

## Optimized path vs CPU

### Discrete — AMD Radeon RX 6600M

| size | cpu_ms | gpu_ms | ratio | create_ms | compare_ms |
|---|---|---|---|---|---|
| 320×200 | 5.81 | 3.45 | **0.59** | 1.33 | 0.76 |
| 1024² | 80.82 | 32.87 | **0.41** | 13.86 | 8.05 |
| 2048² | 352.62 | 184.77 | **0.52** | 89.54 | 32.61 |
| 4096² | 1636.94 | 879.82 | **0.54** | 360.22 | 181.22 |

### Integrated — AMD Radeon Graphics (shared system RAM)

| size | cpu_ms | gpu_ms | ratio | create_ms | compare_ms |
|---|---|---|---|---|---|
| 320×200 | 6.78 | 3.15 | **0.46** | 1.11 | 0.86 |
| 1024² | 117.81 | 60.71 | **0.52** | 19.21 | 14.25 |
| 2048² | 421.36 | 220.36 | **0.52** | 95.50 | 47.15 |
| 4096² | 1813.01 | 1038.29 | **0.57** | 403.94 | 193.27 |

`dssim_check` matches the CPU score within the 5e-6 parity bound on both GPUs
(the `bench` prints it; the parity suites assert it).

## Optimization progression (discrete, GPU/CPU ratio)

| stage | 320×200 | 1024² | 2048² |
|---|---|---|---|
| M9 baseline — per-scale submits, CPU preprocessing (`1dabf06`) | 4.2–15.3× **slower** | (slower) | (slower) |
| + GPU-resident pyramid, one batched submit (`815eab8`) | ~1.1 | ~0.8 | ~0.9 |
| + upload collapsed to one pass (`1a25751`) | **0.59** | **0.41** | **0.52** |

Two changes did the work:
1. **Single batched submit** over GPU-resident planes removed the per-scale
   fence/round-trip overhead that made the naive port overhead-bound.
2. **Upload collapse** — `create_image` was CPU-bound on the RGBA upload, which
   did three passes (interleave→`Vec<f32>`, `pack_f32`→`Vec<u8>`, memcpy).
   Writing straight into mapped staging in one pass roughly halved
   `create_image` and pushed the GPU ahead of the CPU at every size.

## Where the time goes

`create_image` (pyramid build: CPU downsample + upload + one submit) dominates;
`compare` (cross-blur + combine + map readback) is the smaller half — e.g. at
320×200, create ≈ 1.3 ms vs compare ≈ 0.8 ms. The CPU 2×2 downsample is shared
cost (a standing non-goal to move it to the GPU).

## Caveats

- **Laptop GPU**: CPU times swing with thermal state (±15% run-to-run); GPU
  times are stable. Ratios are single-run; the direction (GPU wins everywhere)
  is robust, the exact decimal is not.
- **4K** uses the adaptive split-submit path (≥ 6 M px) to cap peak VRAM. The
  reduction is measured at 512² (6.27 MB saved, deterministic allocator
  counters) and extrapolates to ~400 MB at 4K; not measured at 4K directly.
- **16-bit** inputs are handled at full precision (u16→linear via the 65536-entry
  LUT, same as CPU). BH9 corrected an earlier note here that wrongly called this
  an 8-bit approximation / "documented non-goal."
- Re-measure any time with the command at the top; the numbers above are a
  snapshot, not a contract.

## T9-lite: GPU-busy vs wall (VkQueryPool timestamps)

The bench now reports GPU-busy ms (query-pool timestamps per submit) beside wall
ms. This separates compute from submit/fence/PCIe overhead and cuts through the
±15% wall noise. Discrete RX 6600M:

| size | create wall | create **GPU** | compare wall | compare **GPU** |
|---|---|---|---|---|
| 320×200 | 1.47 | 0.35 | 1.00 | 0.21 |
| 1024² | 14.20 | 1.80 | 9.60 | 2.03 |
| 2048² | 91.60 | **4.78** | 31.46 | 6.96 |
| 4096² | 421.11 | **20.96** | 190.32 | 26.67 |

**Finding (reframes the optimization plan):** GPU compute is a small fraction of
wall — at 2048², `create_image` is 91.6 ms wall but only 4.8 ms GPU-busy. A
CPU-phase probe attributed the gap: `downsample ≈ 10 ms`, `submit+fence ≈ 7 ms`,
and **`prep ≈ 52–92 ms`** (interleave→staging upload + alloc + pass recording).
`prep` scales with *pixels* (10 ms at 1024² → ~90 ms at 2048²) while the pass
count is constant, so it is the **host→staging upload**, not recording or compute.

Consequences for the track order:
- **T5 (2D dispatch / workgroup / compute tuning) is low-value** — it optimizes
  the 4.8 ms GPU slice of a 91 ms operation (~2% of wall). The doc's own §2.3
  warns "doing compute-better first is profiling theater"; the measurement
  confirms it here.
- **The real bottleneck is the upload path.** On discrete it is host→staging
  (PCIe/BAR); on integrated/UMA the staging write is cheap but the
  staging→device `CopyBuffer` is a redundant device-internal copy — which is
  exactly what **T7 (UMA zero-copy)** removes. T7 is therefore the top actionable
  track, not T5.
- Recording/descriptor cost (T3/T10) is a small constant (the ~7 ms `disp` plus
  part of `prep`), so it matters at small sizes, not large.

## Round 2 (post-M10): T9-lite → diagnostic → T7

**T9-lite** (VkQueryPool GPU-busy timing, opt-in) reframed the plan: at 2048²
`create_image` is ~91 ms wall but only ~5 ms GPU-busy. So the compute tracks
(**T5** 2D-dispatch/workgroup, **T6a** fuse h5+h5_mul) target ~5% of the time —
both refuted as low-value, exactly as T5's premise was.

**Diagnostic** (permanent `#[ignore]`'d `upload_diag`): the upload cost is NOT
first-touch page faults (first-write == warm-write) and NOT the memory type (the
same loop into a cached `Vec` was *slower* than the WC mapped write — WC
streaming absorbs it). It was the **per-pixel `PixelsIter` loop**, which is a
disguised memcpy (`RGBAPLU` is `repr(C) [f32;4]`). Fixed: `to_contiguous_buf →
copy_from_slice`. 4K create 421→338 ms (~20%), 2048² ~6%.

**T7** (zero-copy upload): when a memory type is **both** DEVICE_LOCAL and
HOST_VISIBLE (true UMA — integrated Radeon, llvmpipe/CI — **or ReBAR-mapped
VRAM on a discrete GPU**), `create_image` writes the shader's source buffer
directly, dropping the staging buffer and the `staging→device` `CopyBuffer`.
Detection is a `contains()` scan of a single memory type (F34: an earlier
`intersects()` OR-test wrongly matched a pure-DEVICE_LOCAL type, so the claim
"discrete keeps the staging path" was false — a ReBAR discrete also took
zero-copy). Both GPUs on this host have such a type, so **both** take zero-copy.
Measured A/B on the discrete RX 6600M (`DSSIM_UNIFIED` override, `create_ms`):
zero-copy wins at large sizes (2048² 67 vs 91 ms, 4096² 278 vs 323 ms), wash at
≤1024². So zero-copy-on-discrete is now a *measured* choice, not an accident; a
non-ReBAR discrete (no both-flags type) correctly falls back to staging. Parity +
validation clean on both paths.

**Where the time is now** (discrete, 2048²): create ≈ CPU downsample (~10 ms, a
non-goal to move) + WC transfer (~24 ms/64 MB, hardware floor) + recording.
GPU-busy ~5 ms. The remaining levers (T10 barrier-tighten, T11 merge the two
creates, T3 descriptor pre-alloc) are CPU-side micro-overhead with modest,
noise-limited gains — diminishing returns against the transfer/prep floor.

## T11 + stop

**T11** (create_image_pair): both pyramids share one submit (independent
passes), halving the create-side fence at small/medium. 320×200 full-path
`gpu_ms` ~3.5→3.0 ms, ratio 0.42. Large sizes unchanged (one fence is noise vs
the transfer floor). CLI's 1-vs-N streaming can't use it (original reused); it
serves single-pair callers + the bench.

**Stopping here** (per plan). Round-2 net on the discrete GPU, full path
(create+compare), GPU/CPU ratio:

| size | start of round 2 | after T9-lite→memcpy→T7→T11 |
|---|---|---|
| 320×200 | 0.54–0.59 | ~0.42 |
| 1024² | 0.36 | ~0.36 |
| 2048² | 0.43 | ~0.49* |
| 4096² | 0.52 | ~0.51 |

\*large-size ratios are CPU-thermal-noise-limited on this laptop (±15%); the
stable signals are `create_gpu` (~5 ms) and the wall drop from the memcpy+T7+T11
(4K create 421→331 ms). Remaining tracks (T10 barrier-tighten, T3 descriptor
pre-alloc) target CPU-side micro-overhead already dwarfed by the WC-transfer +
CPU-downsample floor — diminishing returns, not done. T5/T6a refuted by T9-lite.
