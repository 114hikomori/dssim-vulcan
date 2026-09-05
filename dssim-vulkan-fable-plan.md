# DSSIM Vulkan Port — Fable Method Plan

**Status:** Proposed implementation plan; no code changes made.

**Scope:** Port the existing `dssim-core` implementation to a Vulkan compute backend while maximizing reuse of the existing CPU implementation, tests, image pipeline, and scoring logic.

**Primary authority:** the actual `dssim-core` source in the target repository. Research material is secondary. When research and source disagree, `dssim-core` wins.

**Ground-truth companion:** [`VULKAN_PORT_PLAN.md`](./VULKAN_PORT_PLAN.md) is the source-line-cited technical reference this plan is built on. §2 below inlines every load-bearing constant so this document is self-sufficient, but `VULKAN_PORT_PLAN.md` remains the fuller citation trail (exact file/line numbers, full boundary-case derivations, both locked sub-image values) — consult it, don't re-derive constants from memory.

---

## 1. Outcome and Definition of Done

The finished work is a `dssim-vulkan` backend that can run the existing DSSIM algorithm on Vulkan without changing the established CPU behavior.

It is done when all of the following are observed:

- CPU `dssim-core` remains the default/reference path and its existing tests stay green.
- A Vulkan path can process the repository's existing fixtures and produce DSSIM scores within **≤5×10⁻⁶ absolute** of the CPU reference — the same bound `dssim-core` already locks (`ssim_locked_values`), including the headline value `0.0009483923725199794` for `test1-sm.png` vs `test2-sm.png`, and **exactly** `0.0` (asserted with `==`) for an image compared with itself.
- The GPU path handles the same important semantic cases as the CPU implementation: grayscale, RGB, alpha, odd dimensions, small images, multiple scales, and the repository's sub-image cases.
- Intermediate GPU results can be dumped and compared against CPU reference dumps, so a mismatch can be localized to a specific stage rather than diagnosed only from the final score.
- Vulkan correctness tests run on lavapipe in CI, with real-GPU validation performed separately.
- The implementation does not require a new image-decoding stack.
- Performance is measured separately from correctness and is not declared successful until end-to-end timings are observed on representative image sizes.

**Explicit non-goals for the first working version:** GPU image decoding, ICC color-profile handling (stays on CPU), float16 arithmetic, multi-GPU scheduling, and GPU-side pooling. GPU pooling can return later as a profiling-justified optimization (§11), never as a default.

### Load-bearing assumptions

- Existing `dssim-core` internals can be instrumented for test/debug output without changing their production semantics.
- The first useful milestone does **not** require the entire pipeline to be GPU-resident.
- CPU pooling/scoring can remain authoritative until the GPU arithmetic has been proven correct.

---

## 2. Evidence Base and Important Corrections

The initial research report was useful for identifying the broad GPU strategy, but several of its algorithm descriptions were too generic for a parity-oriented port. The implementation must follow the repository source instead.

### Confirmed implementation details to preserve

The current plan is based on the repository details already inspected, especially:

- `dssim-core/src/dssim.rs`
- `dssim-core/src/blur.rs`
- `dssim-core/src/blur/equiv_tests.rs`
- `dssim-core/src/image.rs`
- `dssim-core/src/linear.rs`
- `dssim-core/src/tolab.rs`

The important corrections are:

| Area | Research-report simplification | Port requirement | Source |
|---|---|---|---|
| Pyramid | Described as Gaussian | Repeated 2×2 box averaging; stops when `w < 8 \|\| h < 8`, capped at 5 scales | `image.rs:196-230` (`downsample`), `Average4` at `image.rs:129-154` |
| Blur | Generic/binomial 5-tap Gaussian example | Fused double-3×3 kernel with 4-case boundary handling — exact constants below | `blur.rs:1-29` |
| Lab | Presented as a generic RGB→XYZ→Lab conversion | Repository constants, polynomial cube-root approximation, Halley refinement, and output scaling — exact values below | `tolab.rs:12-63` |
| Scale processing | Implied that Lab can simply be converted once | Lab is re-derived every scale from a freshly downsampled RGBAPLU image, not computed once | `dssim.rs:219-256` (`make_scales_recursive`) |
| Pooling | Described generically | Exact per-scale power term, MAD pooling, weights, and final `1/x − 1` conversion — exact values below | `dssim.rs:299-326`, `dssim.rs:439` (`to_dssim`) |
| sRGB | Suggested hardware sRGB conversion as an option | Must reproduce the existing 256-entry LUT-based decode; do not assume a hardware sampler is equivalent | `linear.rs:34-41`, `linear.rs:62-69` |
| Alpha | Not addressed | Un-premultiplication uses a coordinate-dependent pseudo-random dither, not a plain divide — exact formula below | `tolab.rs:128`, `image.rs:160-180` |

These are not optional implementation choices. They are correctness constraints.

### Exact constants (do not re-derive from memory)

A wrong constant here typically moves the final score by ≥10⁻³, which fails the locked tests below. Values confirmed directly against the live `kornelski/dssim` source (`dssim-core` 3.5.1) during this review — the same version this plan targets.

**Blur (`blur.rs:1-29`):**
- 1-D base: `K_SIDE = 0.308_758_86`, `K_CENTER = 0.382_482_8`.
- Fused 5-tap: `K5_OUTER = K_SIDE²`, `K5_INNER = 2·K_SIDE·K_CENTER`, `K5_MID = 2·K_SIDE² + K_CENTER²`.
- Edge coefficients: `K5_EDGE_CENTER = K5_MID + K5_INNER`, `K5_EDGE_NEAR = K5_OUTER + K5_INNER`, `K5_EDGE_FAR = K5_OUTER`.
- Four boundary cases: first/last row-or-column (edge coefficients above); near-edge (`j=1`/`w-2`, plain clamped 5-tap); interior (plain 5-tap); tiny sizes (`w<5`/`h<5`, cases collapse — swept for 1..=8 in `blur/equiv_tests.rs`). **A plain clamped 5-tap over-weights the corner pixel by `K5_OUTER`** — this is the specific bug the equivalence tests exist to catch; do not simplify the boundary to one clamped case.

**Lab conversion (`tolab.rs:12-63`):**
- D65 white point: `D65x = 0.9505`, `D65y = 1.0`, `D65z = 1.089`.
- `EPSILON = 216/24389`; piecewise `cbrt` via a polynomial seed + 2× Halley refinement (`cbrt_poly`), not `pow(x, 1/3)`.
- Output scaling: `L' = Y × 1.05` (not 1.16), `a' = fma(500/220, X−Y, 86.2/220)`, `b' = fma(200/220, Y−Z, 107.9/220)` — these fudge constants are load-bearing, not placeholders to "fix".

**Alpha dither (`tolab.rs:128`, `image.rs:160-180`):** `n = (x+11) ^ (y+11)`; a channel gets `+ (1 − a)` when bit 16 (R), 8 (G), or 32 (B) of `n` is set. Opaque pixels (`a=1`) are unaffected. Must be bit-exact or any alpha-bearing image diverges.

**Pooling (`dssim.rs:85, 299-326, 439`):** `DEFAULT_WEIGHTS = [0.028, 0.197, 0.322, 0.298, 0.155]`; per scale, `avg_n = (Σ map_n / len).max(0.0) ^ (2^-n)` (the power term is easy to drop); `score_n = 1 − (Σ|avg_n − p| / len)`; final `DSSIM = 1 / max(weighted, f64::EPSILON) − 1`.

**SSIM combine (`dssim.rs:335-395`):** explicit `fma()`/`mul_add` at exactly two sites — `2·fma(mu1_mu2, C1)` and `2·fma(sigma12, C2)`, `C1 = 0.01²`, `C2 = 0.03²` — same association order as the source. A scalar 1-channel variant (`compare_scale`) must be preserved for gray-input fixtures.

**Locked test values (`dssim.rs`, `ssim_locked_values`):** `test1-sm.png` vs `test2-sm.png` → `0.0009483923725199794` at ≤5×10⁻⁶ absolute; identity asserted with `==` for **exactly** `0.0`. Two sub-image scenarios are locked to specific values too (44×33 and 61×40 offset/crop) — see `VULKAN_PORT_PLAN.md §1` for those figures rather than re-deriving them.

---

## 3. Core Strategy: Hybrid First, Full GPU Later

The biggest architectural change from the earlier plan is to **avoid moving every stage to Vulkan at once**.

Start with:

```text
Existing image loading / profile handling
              |
              v
       Existing dssim-core
       preprocessing semantics
              |
              v
       CPU Lab / scale 0 data
              |
              v
      Vulkan blur/statistics/SSIM
              |
              v
       SSIM map readback
              |
              v
       Existing CPU pooling
              |
              v
          DSSIM score
```

Only after this works should more preprocessing move to the GPU.

This creates two independently solvable problems:

1. **Can Vulkan reproduce DSSIM's computational core?**
2. **Can the remaining CPU preprocessing be moved without changing semantics?**

Do not debug both at the same time.

---

## 4. Reuse Boundary

### Reuse directly

Keep these on the existing CPU path initially:

- PNG/JPEG/other image decoding already provided by the repository
- ICC/profile handling
- input normalization
- existing public CLI behavior
- existing DSSIM pooling and final score conversion
- existing reference constants and weights
- existing CPU tests and fixtures

### Port/adapt

Implement Vulkan versions of only the computational stages that benefit from GPU execution:

- blur
- fused product + blur
- local statistics
- cross-image blur
- per-pixel SSIM combine
- 2×2 scale generation
- eventually linear/Lab conversion

### Do not rebuild

Do not create a second image I/O framework, independent DSSIM scoring implementation, or general-purpose Vulkan framework.

---

## 5. Project Shape

A small backend crate is preferred:

```text
dssim-vulkan/
├── Cargo.toml
├── build.rs
├── shaders/
│   ├── blur_h.comp
│   ├── blur_v.comp
│   ├── blur_mul_h.comp       # or specialization-based variant
│   ├── blur_mul_v.comp
│   ├── ssim_combine.comp
│   ├── downsample_box2.comp
│   └── rgba_to_lab.comp      # later phase
└── src/
    ├── lib.rs
    ├── context.rs
    ├── resource.rs
    ├── transfer.rs
    ├── pipeline.rs
    ├── blur.rs
    ├── statistics.rs
    ├── ssim.rs
    ├── scale.rs
    └── debug_dump.rs
```

The exact module split is deliberately provisional until the first implementation proves what abstractions are actually needed.

### Rust/Vulkan direction

Preferred direction:

- Rust
- `ash` for Vulkan bindings
- one allocator abstraction
- build-time shader compilation

Do not add dependencies merely because they are common in Vulkan examples. Each dependency should solve a concrete problem in this repository.

A C++ sidecar is a fallback only if the Rust stack creates a real blocker, not a parallel implementation to maintain from day one.

---

## 6. Implementation Sequence

### Phase A — Reference instrumentation

**Target:** 3–5 days

Create test-only CPU dumps for:

- scale dimensions
- RGBAPLU data where needed
- Lab planes
- mean/blur outputs
- squared-product blur
- cross-product blur
- SSIM maps

Create a small comparison utility that reports:

```text
max absolute error
mean absolute error
RMSE
worst pixel location
CPU value
GPU value
```

The dump format should be deliberately boring: fixed metadata + contiguous `f32` payload.

### Exit observation

Generate the same CPU dumps twice and verify they are byte-for-byte reproducible.

---

### Phase B — Minimal Vulkan runtime

**Target:** about 1 week

Implement only what the algorithm currently needs:

- instance/device selection
- validation in development builds
- compute queue
- command pool/buffer
- synchronization
- allocator/resource helpers
- shader loading
- staging upload/download
- debug naming

First shader:

```text
input -> multiply by 2 -> output
```

Run it on lavapipe and one real GPU.

### Exit observation

The smoke test produces the expected output and Vulkan validation reports no relevant errors.

---

### Phase C — First GPU kernel: blur

**Target:** 1–2 weeks

Do **not** start with RGBA→Lab.

Start with the most reusable spatial primitive: the repository's blur.

Implement the actual `dssim-core` behavior:

- H5 pass
- V5 pass
- fused product + blur
- in-place chroma blur semantics where needed
- all four boundary cases, using the exact `K5_EDGE_*` constants in §2 — a plain clamped 5-tap over-weights the corner pixel and is the single most likely bug here
- tiny dimensions (1..=8, per the equiv-test sweep)

The first implementation should be simple and explicit rather than optimized.

Use the repository's `blur/equiv_tests.rs` as the source of test cases.

### Exit observation

The Vulkan blur output passes the translated equivalence suite against the CPU implementation within a measured tolerance.

---

### Phase D — Single-scale GPU SSIM

**Target:** 1–2 weeks

Keep Lab generation on CPU.

Upload the Lab planes for a single scale and perform on GPU:

```text
chroma pre-blur
        |
        +--> mean blur
        +--> square blur
        +--> cross-product blur
                         |
                         v
                 SSIM combine
                         |
                         v
                     SSIM map
```

Read the SSIM map back and let the existing CPU pooling code calculate the score.

### Exit observation

The GPU SSIM map matches the CPU map closely enough to meet the measured per-pixel bound, and the CPU-pooled single-scale score matches the reference.

---

### Phase E — Multi-scale GPU pipeline

**Target:** about 1 week

Move scale generation to Vulkan only after single-scale correctness is established.

Preserve the exact order used by the CPU implementation:

```text
current RGBAPLU scale
        |
        +--> Lab conversion for this scale
        |
        +--> SSIM work
        |
        +--> 2×2 box downsample RGBAPLU
                        |
                        v
                    next scale
```

Do not substitute:

- Gaussian downsampling
- Lab-space downsampling
- a different scale-stop condition

Keep the maximum scale count and image-dependent termination exactly aligned with the source.

### Exit observation

Every repository fixture produces a final GPU result within the target tolerance, including the sub-image and small-image scenarios.

---

### Phase F — GPU color conversion

**Target:** 1–2 weeks

Only now move:

```text
RGBA8 sRGB
   ↓
existing linear LUT semantics
   ↓
alpha premultiplication
   ↓
RGBAPLU
   ↓
Lab
```

The first GPU implementation should deliberately reproduce the CPU semantics rather than use an allegedly equivalent hardware path.

Examples of behavior that must be preserved include:

- LUT-based linearization
- alpha premultiplication order
- alpha dither `n = (x+11) ^ (y+11)`, bits 16/8/32 for R/G/B (§2) — a plain un-premultiply divide will not match
- repository XYZ/Lab constants: D65 white point, `EPSILON`, `cbrt_poly` (§2)
- polynomial + Halley cube-root approximation
- non-standard output scaling: `×1.05`, `86.2/220`, `107.9/220` (§2) — load-bearing fudges, not bugs to "fix"

Once parity exists, hardware-assisted alternatives may be benchmarked separately.

---

### Phase G — Integration and hardening

**Target:** about 1 week

Add:

- GPU backend selection
- CPU fallback
- CLI integration: a `--gpu` flag selects the Vulkan path; default output format (`{dssim:.8}\t{file}`, `-o` map writing, `main.rs`) stays byte-identical to the CPU path
- lavapipe CI job
- real-GPU smoke/regression testing
- benchmark harness
- documentation

The existing CPU path must remain unchanged as the default.

---

## 7. Numerical Correctness Policy

Correctness should be defined by observations, not by an assumption that GPU and CPU must have identical bit patterns.

### Initial targets

| Stage | Initial target |
|---|---:|
| LUT/premultiply | ≤ 1e-7 max abs |
| Lab | ≤ 1e-6 max abs |
| 2×2 downsample | ≤ 1e-6 max abs |
| blur | ≤ 2e-6 max abs |
| SSIM map | ≤ 2e-6 max abs |
| final DSSIM | ≤ 5e-6 abs — checked against the locked `0.0009483923725199794` value, not just "close to CPU" |
| identity (image vs itself) | exact `== 0.0`, no tolerance |

These are **starting targets**, not excuses to widen tolerances automatically.

When a target fails:

1. locate the first stage that diverges;
2. inspect the exact pixel/operation;
3. determine whether the cause is algorithmic, memory/layout related, or floating-point ordering;
4. fix the cause before changing the tolerance.

### Floating-point rules

The first correctness path should avoid:

- fast-math assumptions
- nondeterministic reductions
- unnecessary subgroup operations
- aggressive kernel fusion
- mixed precision

Preserve explicit FMA behavior where the CPU source deliberately uses `mul_add`.

Where exact operation contraction matters, inspect the generated SPIR-V and use the appropriate precise/no-contraction controls rather than assuming the shader compiler will preserve the source expression tree.

---

## 8. Identity Semantics

The repository's identity test expects exactly `0.0` for an image compared with itself.

Do not depend on a floating-point pipeline naturally producing exact `1.0` at every operation.

Where the public contract requires exact identity, use a deterministic fast path before GPU work when the input images are semantically identical.

The GPU path should still have an identity regression test; the fast path is a contract safeguard, not a substitute for validating the GPU algorithm.

---

## 9. GPU Data Representation

Do not lock the project to one representation before measurement.

The initial correctness implementation should favor a representation that makes debugging easy, even if it is not the fastest possible one.

Evaluate both:

```text
VkBuffer / SSBO
VkImage / storage image
```

against the actual access pattern.

### Initial preference

Use straightforward contiguous float storage where it simplifies:

- CPU/GPU dumps
- staging
- stride handling
- debugging

Use image resources where they provide a measured advantage for spatial kernels.

The optimized representation can differ from the correctness representation if the numerical behavior remains controlled.

---

## 10. Synchronization and Dispatch Strategy

Correctness version:

```text
dispatch A
   ↓
explicit dependency
   ↓
dispatch B
```

Do not optimize barriers before the dependency graph is proven correct.

After profiling:

- reduce unnecessary barriers
- reuse command buffers/resources
- reuse descriptor infrastructure
- batch compatible dispatches
- consider synchronization2 where it materially simplifies precise dependencies

Never remove a dependency merely because it appears unnecessary in a small test.

---

## 11. Performance Strategy

Performance is a separate milestone from functional correctness.

Profile these components independently:

```text
image decode
CPU preprocessing
upload
GPU preprocessing
blur
statistics
SSIM
readback
CPU pooling
```

Report both:

- kernel-only throughput
- end-to-end pair latency

### Optimization order

1. Resource reuse
2. Pipeline/descriptor reuse
3. Batch/reference reuse
4. Reduce avoidable transfers
5. Kernel fusion where profiling justifies it
6. Shared-memory tiling for expensive spatial accesses
7. Channel packing where beneficial
8. GPU-side reduction only if CPU readback/pooling is proven to be the bottleneck

Do not establish a CPU/GPU crossover threshold until it is measured on representative hardware.

---

## 12. Validation Matrix

Minimum functional matrix:

- repository locked RGB fixtures
- grayscale fixture
- alpha-bearing fixture
- odd dimensions
- dimensions near the scale cutoff
- very small images
- sub-image/offset scenarios
- identical-image case

The validation corpus should expand before performance claims are made.

A larger IQA corpus can later be used to detect systematic drift, but the primary oracle remains `dssim-core`, not agreement with another implementation.

---

## 13. CI / Hardware Matrix

### Pull requests

Run correctness on:

```text
lavapipe
CPU reference
```

### Nightly / scheduled validation

Run on available real GPUs from the target vendor set.

The purpose is to catch:

- format feature mismatches
- shader compiler behavior differences
- synchronization mistakes exposed by different drivers
- resource/layout assumptions

Do not use one vendor's successful execution as proof of portability.

---

## 14. Research/Reuse Sources Worth Keeping Open

### Primary algorithm source

- `dssim` / `dssim-core`: https://github.com/kornelski/dssim

This remains the authoritative implementation reference.

### Existing GPU architecture reference

- Vship: https://github.com/Line-fr/Vship

Useful as a source of GPU image-processing, batching, and kernel design patterns. It is **not** a drop-in DSSIM implementation.

### Vulkan infrastructure references

- Vulkan-Hpp: https://github.com/KhronosGroup/Vulkan-Hpp
- Volk: https://github.com/zeux/volk
- Vulkan Memory Allocator: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
- Vulkan SDK: https://www.lunarg.com/vulkan-sdk/
- NVIDIA Vulkan performance guidance: https://developer.nvidia.com/blog/vulkan-dos-donts/

These are infrastructure/engineering references, not DSSIM algorithm specifications.

---

## 15. License / Provenance Rule

`dssim-core` is licensed AGPL-3.0 (confirmed in `dssim-core/Cargo.toml`). `dssim-vulkan`, as a derivative that reuses its algorithm and any of its source, inherits the same AGPL/commercial dual-license obligation — this is a legal constraint on the crate from day one, not a checklist item to resolve later. No AGPL-derived code may be absorbed into a permissively-licensed consumer.

Do not assume that studying an implementation makes copied code reusable under the desired license.

Maintain a simple provenance record for every non-trivial piece of reused source:

```text
source project
file/function
license
copied / adapted / independently reimplemented
reason for use
```

In particular, distinguish:

1. studying `dssim-core` as the algorithm specification;
2. directly reusing source code from it;
3. independently reimplementing the algorithm from observed behavior;
4. incorporating third-party Vulkan helper code.

The final licensing decision should be based on the actual code and licenses involved, not a blanket assumption about what a Vulkan port legally is.

---

## 16. Recommended Milestones

Use observable milestones instead of calendar promises.

```text
M0  CPU dumps are reproducible
M1  Vulkan compute smoke test works on lavapipe + one real GPU
M2  Vulkan blur passes the CPU equivalence suite
M3  Single-scale GPU SSIM map matches CPU
M4  Full DSSIM works with CPU preprocessing + GPU compute
M5  GPU multi-scale pipeline matches CPU
M6  GPU color/Lab path matches CPU
M7  Alpha/gray/odd-size/small-image matrix is green
M8  CLI + CI + fallback integrated
M9  Performance profile completed
M10 Optimized path beats measured CPU baseline on chosen workloads
```

**M4 is the critical functional milestone.** At M4, there is already a useful Vulkan DSSIM implementation even though preprocessing remains partly on CPU.

---

## 17. Revised Time Estimate

Assuming prior Vulkan/GPU programming experience:

| Milestone group | Expected effort |
|---|---:|
| Reference harness | 3–5 days |
| Vulkan runtime | ~1 week |
| Blur + kernel validation | 1–2 weeks |
| Single-scale GPU SSIM | 1–2 weeks |
| Multi-scale | ~1 week |
| GPU color/Lab | 1–2 weeks |
| Integration/CI | ~1 week |
| Initial optimization | 2–4 weeks |

A reasonable planning envelope is:

- **Functional GPU DSSIM:** roughly 4–8 weeks
- **Hardened + meaningfully optimized:** roughly 6–12 weeks

This is deliberately an engineering estimate, not a promise. The first hard checkpoint is M2/M3: if the repository's blur or arithmetic semantics prove substantially harder to reproduce than expected, the schedule should be revised from observed evidence rather than defended from the original estimate.

---

## 18. First Actions

The first implementation slice should be exactly this:

```text
1. Add CPU intermediate dump capability.
2. Reproduce the dump twice and verify determinism.
3. Create minimal ash-based Vulkan runtime.
4. Run a trivial compute shader through lavapipe.
5. Upload one CPU-generated Lab plane.
6. Port only the actual dssim-core blur.
7. Compare GPU blur against the CPU blur dump.
```

Do **not** start with:

- PNG/JPEG decoding
- a new CLI
- GPU reduction
- GPU Lab conversion
- shared-memory optimization
- fused multi-stage kernels
- multi-GPU support

Those are later concerns.

---

## 19. Decision Summary

The recommended approach is **not** “rewrite DSSIM in Vulkan.”

It is:

```text
existing dssim-core
        |
        +---- existing I/O / semantics / scoring
        |
        +---- CPU oracle + dumps
        |
        +---- progressively replace expensive stages
                         |
                         v
                  Vulkan compute backend
```

The shortest credible path is therefore:

**CPU oracle → Vulkan blur → single-scale SSIM → full GPU multi-scale → GPU color conversion → optimization.**

Alternatives considered:

- **All-GPU from day one:** rejected because it couples too many correctness risks at once.
- **Reimplement the entire DSSIM stack independently:** rejected because it throws away proven repository behavior and tests.
- **Start from Vship:** rejected as the primary implementation base because Vship targets different perceptual metrics; it remains a useful GPU engineering reference.
- **GPU pooling first:** rejected because CPU pooling is already cheap and provides a stable numerical oracle.

No implementation work is authorized by this plan. It defines the sequence and verification criteria for the next implementation pass.

---

## 20. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Implementing a generic/binomial blur instead of the repo's fused double-3×3 kernel | High if unmitigated | Fatal to parity (≥10⁻³ error) | §2 pins every blur constant; port `blur/equiv_tests.rs` wholesale in Phase C |
| Boundary-case divergence on blur | Medium | High (≥10⁻³ locally) | Four-case boundary and the corner-overweighting caution are explicit in §2 and Phase C; tiny-size sweep 1..=8 mirrored |
| FP reordering drift accumulates across scales | Medium | Medium | Explicit `fma()` sites preserved (§2); stage-wise tolerances in §7; CPU f64 pooling stays authoritative |
| Alpha dither divergence | Low (alpha images only) | High for those cases | Bit-exact `n=(x+11)^(y+11)` logic pinned in §2 and Phase F; alpha fixture in the validation matrix (§12) |
| Scale-count mismatch on small images | Medium | Medium | Exact `w<8 \|\| h<8` stop rule preserved (§2); sub-cutoff fixture in the validation matrix (§12) |
| Driver/vendor quirks (storage formats, row pitch, MoltenVK FP behavior) | Medium | Medium | lavapipe baseline + nightly real-GPU matrix (§13); feature queries with CPU fallback |
| Rust Vulkan ecosystem gaps (allocator maturity, shader tooling) | Low | Medium | Documented C++ sidecar fallback (§5) — subprocess, same CLI contract; last resort only |
| AGPL-3.0 licensing | Certain | Legal | `dssim-core` is AGPL-3.0; `dssim-vulkan` carries the same dual-license obligation as a derivative — see §15 |
| GPU-less or headless environments | Common | Low | CPU path remains the default; GPU is opt-in via `--gpu` (Phase G) |
