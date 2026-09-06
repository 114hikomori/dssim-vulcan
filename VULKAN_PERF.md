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
- **16-bit** inputs are handled as 8-bit on the GPU path (documented non-goal).
- Re-measure any time with the command at the top; the numbers above are a
  snapshot, not a contract.
