# Audit — GPU port (rolling)

## Pass 1 — M7 + M9 (Phase H first slice)

Date: 2026-09-06. Auditor: opencode (fable-judge style review, read-only).
Pinned to HEAD `acf9fca` (commits `1dabf06`, `0b29845`, `de10c0e`, `acf9fca` and the
cumulative state through M6). **No code was changed.** This file only records findings.
Findings F1–F24 belong to this pass. Pass 2 (below) audits the Phase H optimization
(`815eab8` + `c4ee8d4`) and continues the numbering from F25.

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

---

## Pass 2 — Phase H optimization (`815eab8` + `c4ee8d4`, M10 claim)

Date: 2026-09-06 (same day, later). Pinned to HEAD `c4ee8d4`. Read-only; no code
changed. Claims re-observed, not trusted:

| Claim | Reproduced? |
|---|---|
| full workspace green on RX 6600M | YES — `cargo test --workspace` all suites pass, 0 ignored, no silent skips |
| every committed .spv == fresh `glslc --target-env=vulkan1.3 -O` | YES — 7/7 SHA256 match |
| `h5_mul_into` dst_off=0 at every call site (bug-1 fix) | YES — 3/3 sites |
| ratios 0.82 / 0.93 / 1.14 | PARTIAL — re-run gave 0.76 / 0.95 / 1.03 (noise band); load-bearing conclusion (GPU wins ≥1024px, loses small) holds |
| no test weakening | YES — diff touches no test file, no tolerance; one new *stricter* assert |

### Findings

**F25 (high, spec violation — latent) — `lab_all`→`img_all` CopyBuffer without transfer usage flags.**
`score.rs` `create_image` (RGB): `s.lab` and `s.img` are allocated `STORAGE_BUFFER`
only, then used as `Pass::CopyBuffer` src/dst. `vkCmdCopyBuffer` requires
`TRANSFER_SRC`/`TRANSFER_DST` (VUID-vkCmdCopyBuffer-srcBuffer-00118/-00120) —
undefined behavior per spec. It runs today only because the AMD driver is permissive — validation was on
but its VUID report was invisible under libtest capture (see F26 correction). A strict
driver or a validation-enabled CI leg would fail. *(Pass 3: validation was in
fact on and reporting this — the VUID output was swallowed by libtest capture,
so it survived review unnoticed.)* Cheapest real fix: have
`lab_into` write plane 0 (L) directly into `img_all` at dst_off=0 — the whole
copy exists only to seed plane 0, so it disappears along with the violation.

**F26 (medium, process) — "verified on GPU" runs without validation and nobody knows.**
*(Pass-3 correction: the mechanism below was wrong — validation was enabled all
along; its output was invisible because libtest captures stderr without
`--nocapture`. Conclusion and fix direction stood. See Pass 3.)*
`context.rs:78-87` enables validation only `if installed`; on this machine the
Khronos layer exists at `C:\VulkanSDK\1.4.357.0\Bin\VkLayer_khronos_validation.json`
but is not registered (implicit-layer registry has only Steam overlay) and
`VK_LAYER_PATH` is unset → `validation_enabled = false` with zero output. Every
"GPU-validated" claim so far (including F25's survival) is validation-less.
Recommend: `eprintln!` the validation state at Context creation in debug builds,
and set `VK_LAYER_PATH` in the CI validation leg (ties to F21).

**F27 (medium, memory) — `mu_all` allocated host-visible but never read back.**
`s.mu` uses `GpuToCpu` + `TRANSFER_SRC`; `compare` consumes it only on-device
(`combine_into`), and the gray path correctly uses `GpuOnly`. Host-visible VRAM
is the scarce, BAR-resized pool on dGPUs — up to ~48MB/scale wasted at 2048².
Switch to `GpuOnly` (keep `TRANSFER_SRC` only if something later reads it).

**F28 (medium, memory) — peak-VRAM regression from the single-submit design.**
All scales' buffers — including per-scale transients (`staging`, `rgba`, `lab`,
`tmp`) — are allocated before the one `dispatch_sequence` and kept alive by
`passes` until it returns: ≈84 B/px summed over scales ≈ 470MB at 2048²,
≈1.9GB at 4096² (iGPU shares system RAM). The old per-scale code freed
transients immediately. Options: one max-size transient arena reused across
scales via plane offsets, or split into a few submits (still amortizes far
better than per-scale).
**F29 (medium, latent API trap) — `h5_mul_into` uses `stride1` for both sources; `compare` doesn't assert equal dimensions.**
`blur.rs` `h5_mul_into` builds `pc_mul_bytes(..., stride1, src1_off, stride1, src2_off, ...)`
— src2's stride is hardcoded to stride1. The deleted CPU path passed
`stride2 = mod_scale.width` explicitly. `compare` asserts only channel equality
(`score.rs`), so a ref/mod size mismatch (library misuse; the CLI guards it —
`gpu_cli` size-mismatch test) now reads out of bounds on the GPU instead of the
old CPU slice panic. Add `assert_eq!(ref_scale.width, mod_scale.width)` (+height)
in `compare`, or thread a real stride2.

**F30 (low, debris) — refactor leftovers.**
`score.rs:161-163` empty block `{ }` (the `passes2`-merge scar); no trailing
newline at EOF in `score.rs` and `color.rs`; column-0 `let mu`/`let sq` at
`score.rs:234,240`; over-indented `let pc` at `color.rs:216`; `ScaleKeep` and
`GrayKeep` are the same struct minus `channels` (collapse to one). Note:
`cargo fmt --check` passes vacuously — `.rustfmt.toml` `disable_all_formatting=true`
is upstream Kornel policy (`73933cd`, pre-port), already logged as F22; these
lines are debris relative to the file's own style, not a fmt-gate failure.

**F31 (low, claim accuracy) — "only the (tiny) SSIM maps come back".**
Scale-0's map is full-resolution: 16MB at 2048², ≈22MB total per compare
(4/3·P0·4B). "Tiny" fits the pooled maps of the old design's imagination, not
this one. The CPU-side pooling itself is correct per the standing non-goal
(AGENTS.md §8 — no GPU-side pooling without profiling justification); just fix
the wording in `score.rs` doc-comments and the bench note.

**F32 (low, robustness) — `CopyBuffer` silently truncates on size mismatch.**
`pipeline.rs`: `size: src.size.min(dst.size)`. Every current call site is
equal-sized; a future mismatch becomes a silent partial copy. Prefer
`assert_eq!(src.size, dst.size)` (or an explicit size field on the variant).

**F33 (info, fragility) — push-constant layout aliasing.**
`v5_into` takes `_stride` and ignores it (src must be tight — true for all
callers today, undocumented). `pc_bytes`'s `src2_stride` argument is read as
`src_off` by the non-mul h5 shader — safe only because callers pass 0 (the
checkpoint's twin-audit note). Both are one wrong-argument away from a
silent-corruption bug of exactly the class fixed as bug-2. Recommend: delete
`v5_into`'s dead param and migrate `blur()`/`blur_mul()` wrappers onto
`pc_bytes_off`/`pc_mul_bytes` so each field has one meaning per shader.

### Pass 2 verdict

The optimization is real and honestly reported (all load-bearing claims
reproduced; the three self-found bugs show genuine verification happened).
F25 is the one to fix before any validation-enabled CI leg exists; F26 is why
F25 survived review. F27/F28 are the price of the single-submit design and are
worth paying down before pushing to 4K sizes.

### Caveats of this pass

- Bench numbers are single-run on a warm laptop GPU; ±10% noise, direction stable.
- F25's "would fail on strict drivers" is spec-based inference, not observed —
  no validation-enabled run exists on this machine to demonstrate it.

---

## Pass 3 — fix batch verification (`b63ebd5`..`878de7a`, "all findings addressed" claim)

Date: 2026-09-06, pinned to HEAD `878de7a`. The implementing agent claims every
Pass-1/Pass-2 finding (F1–F33) is closed. Re-observed, not trusted:

| Claim | Reproduced? |
|---|---|
| full workspace green WITH validation, 0 vulkan errors | YES — `cargo test --workspace` exit 0, all suites ok; `--nocapture` probe: "validation layer: ENABLED" ×5, zero `[vulkan ERROR]`/`VUID-` lines |
| F25 fixed by deleting the illegal copy | YES — `lab_all`/`s.lab` gone (0 matches); Lab written straight into `img_all` |
| F27 mu_all → GpuOnly | YES |
| F29 compare asserts ref/mod dims | YES — score.rs:360 |
| F32 CopyBuffer asserts equal size | YES — pipeline.rs, with F32 comment |
| F33 pc_bytes deleted, dead `_stride` gone | YES — only `pc_bytes_off`/`pc_mul_bytes` remain |
| F28 split-submit seam + measured peak (25.1 vs 18.9 MB @512²) | YES — `SPLIT_SUBMIT_MIN_PIXELS=6M`, both split tests present and green |
| F1 flush/invalidate added | YES — transfer.rs `sync_host_range(flush)` |
| F7 gpu_cli can't pass vacuously | YES — asserts positive `dssim: gpu device:` line, hard-FAIL on fallback-with-device, explicit SKIP only when no device; strictly stronger |
| F13 ICC demonstrated empirically | YES — profile.png has iCCP, stripped lacks it, sizes differ; test green (dssim 0.0 only possible if profile applied) |
| F21 CI: validation install + grep gate + clippy `-D warnings` | YES — ci.yml:18/45-48/30-31; the `--nocapture` discovery (13b13cb) is the right fix for a vacuous grep gate |
| F19 correction (no gray→GPU CLI routing) | YES — `run_gpu` uses `load_image_rgba` only; `image_gray` is CPU-only. Their correction of MY pass-1 finding is accurate and was appended, not history-edited |
| no test weakening | YES — zero removed assert/skip lines on the minus side of all test diffs |
| GPU beats CPU at all sizes (0.50/0.34/0.44, 4K 0.45) | YES — re-run: 0.63/0.37/0.43/0.45 incl. 4096², no OOM; dssim_check byte-identical to Pass-2 values (parity preserved) |
| SPV freshness after comment-only shader edits | YES — 7/7 hash match |

### Correction to this audit's own Pass 2

**F26's mechanism was wrong.** I inferred "validation silently OFF on this
host" from (a) no validation lines in captured logs and (b) the implicit-layer
registry showing only Steam. (a) was libtest swallowing stderr without
`--nocapture`, and (b) the loader finds the SDK layer through paths my registry
probe didn't enumerate. Validation was in fact ENABLED all along — which means
the F25 VUID violation was being *reported* during Pass-2 runs, just invisibly.
The finding's conclusion (validation state must be printed; claims must be
provable) stands and their fix (print ENABLED / WANTED-but-NOT-FOUND + CI
`--nocapture` grep gate) closes it properly. Lesson recorded: absence of
captured output is not evidence of absence — same trap the CI gate fell into.

### New findings (process, not code)

**P1 (process) — CI config modified without a recorded authorization.**
RESOLVED 2026-09-06: user confirmed in the auditing session ("P1-2 ฉันสั่งเอง"
— "I ordered those myself"). No action needed beyond this record.
AGENTS.md §8 prohibits touching CI config "absent explicit instruction
otherwise". `37d9fd8`/`13b13cb` modify `.github/workflows/ci.yml`; the
checkpoints justify it under M8 scope but record "Deviated from plan: none"
and quote no user instruction for the CI change itself. If the human did
authorize it in the implementing session, append that quote to the checkpoint;
if not, this needs a look.
**P2 (process) — five pushes to origin/main this day.** RESOLVED 2026-09-06:
user confirmed the pushes were authorized ("P1-2 ฉันสั่งเอง"). The `d128cbb`
"push to check" quote is genuine. No action needed.

**P3 (resource safety) — one-off workspace-test abort, cause unknown.** First
full `cargo test --workspace` of this pass aborted mid-way (exit ≠0, no FAILED
line, smoke/ssim_parity never ran); the identical command immediately after was
green, and each missing suite passed standalone. My initial "concurrent GPU
collision" hypothesis is REFUTED — the user confirms no other agent is running
in this repo. So this was a transient, not contention: most likely a driver
TDR / device-lost / power-state blip on the laptop GPU during those two suites.
Left as an open observation, not a defect: if it recurs, capture
`driver reset`/`VK_ERROR_DEVICE_LOST` in the harness; no lock file needed.

**P4 (nit) — small-image ratio drift.** 320x200 measured 0.63 here vs 0.50
reported; still <1.0 so the "wins at all sizes" conclusion holds, but the
small-size margin is the noisiest and should be quoted as a range, not a point.

### Pass 3 verdict

**VERIFIED.** "ALL findings F1–F33 addressed" reproduces on every
observationally checkable claim, with no weakened checks, no deleted tests,
and one honest correction of this audit's own faulty inference. P1/P2
confirmed authorized by the user; P3 downgraded to a one-off transient
(collision hypothesis refuted). Remaining: the pre-existing open question
from `c4ee8d4` — M10 sign-off on the integrated GPU + CI lavapipe leg with
the optimized path (CI run #8 pending per `878de7a`).

---

## Pass 5 — upload fix (`e561e9f`) + my page-fault hypothesis: REFUTED by their data

Pinned to `e561e9f`. The implementing agent ran a permanent `#[ignore]`d
diagnostic (transfer.rs `upload_diag`) BEFORE fixing — and it refuted my
pass-4 guess (first-touch page faults): fresh-write 23.98 ms ≈ warm-write
23.13 ms for 64 MB, and the same loop into cached RAM was SLOWER (32.8 ms).
The cost was the per-pixel scalar loop, not memory behavior. I was wrong;
their experiment was right and is reproducible (re-ran: numbers match).

Fix verified:
- Load-bearing claim checked at source: `RGBAPLU = RGBA<f32>` = `rgb` crate
  `#[repr(C)] Rgba { r, g, b, a }` — field order + repr(C) + compile-time
  `size_of == 16` assert make the memcpy byte-identical to the old loop.
- Workspace green (10 suites, 0 failed, 0 ignored); dssim_check byte-identical
  to all prior runs (parity preserved); clippy `-D warnings` clean.
- Bench reproduced: 4096² create 421→358 ms (~15%, their claim 20% — noise
  band); 2048² 91.6→90.5 (their honest "~6%, near noise").

Residual (mine, for the next pass): after the fix, 2048² create is still
90.5 wall vs 5.1 GPU-busy. The remaining ~68 ms ≈ 89 MB pyramid upload at
~2.7 GB/s WC-memcpy bandwidth — now near the hardware floor for f32-precision
parity (u8 upload would be 4x faster but breaks the 2e-6 map tolerance by
construction). Real levers left: T7 on UMA (write lands in final memory,
no staging copy) or accepting the floor on discrete. Also: `e561e9f` has no
CHECKPOINT entry — commit message carries the record; acceptable for a
mid-phase fix, but the next milestone-level change should log one.

---

## Pass 6 — Phase H round 2 (`4bbe731` T7 + `dd8a886` T11 + stop decision)

Pinned to `7b01b59`. Reproduced:
- T7 parity: workspace green on both GPUs; CI run #11 green — llvmpipe is
  unified, so CI genuinely exercises the zero-copy path now.
- T7 win (iGPU): 320×200 create 2.21→1.22 ms, 2048² create 110→81 ms vs my
  own pre-T7 iGPU run — real, matches doc direction.
- T11: `create_image_pair` + `phase_e_create_pair_matches_separate_and_cpu`
  (phase_e now 6 tests; pair == two-separate == CPU); 320×200 dGPU gpu_ms
  3.57→3.07 matches the ~3.0 claim. Zero removed assertions in test diffs.
- Stop decision: documented, attributed to user, consistent with the
  measurement floor (WC-memcpy + CPU downsample).

**F34 (medium, claim accuracy + latent behavior) — UMA detection is OR, not AND;
"discrete keeps the staging path" is false.** `context.rs:207` uses
`property_flags.intersects(DEVICE_LOCAL | HOST_VISIBLE)` — `intersects` means
"has EITHER flag" while the adjacent comment says "both". vulkaninfo confirms
RX 6600M exposes a pure-DEVICE_LOCAL type (0x0001) → `is_unified_memory()` is
true on the discrete GPU too, and the discrete path DOES take zero-copy
(CpuToGpu source buffer, no staging, no CopyBuffer). Consequences:
(a) the claim in `4bbe731`/`fd6d099`/`VULKAN_PERF.md` ("Discrete keeps the
staging path / Discrete path unchanged") is contradicted by the code;
(b) the 2048² dGPU create win measured this pass (90.5→65 ms) is silently a
T7 effect, not T11 — unclaimed, unmeasured-as-such, positive but undocumented;
(c) on a non-ReBAR discrete GPU the shader would read the source from system
RAM — correct (parity would hold) but never benchmarked anywhere, since both
local GPUs + llvmpipe are unified-ish.
Fix: `contains()` instead of `intersects()`, and ideally gate on "the type
gpu-allocator would pick for GpuOnly is host-visible" rather than "some type
is"; then re-measure discrete staging-vs-zero-copy as an explicit A/B — the
current accidental data suggests zero-copy-on-discrete may be the better
default anyway, which is exactly why it should be a measured decision, not a
detection accident.

**P3 update — the mid-run abort RECURRED (2nd sighting), pattern emerging.**
Both occurrences were the FIRST `cargo test --workspace` after a rebuild;
immediate re-run green. Not "one-off": likely cold driver shader cache /
first-touch device contention across the 6 GPU test binaries. If it happens
a third time, capture whether the dying binary reports `VK_ERROR_DEVICE_LOST`.

**P5 (process) — round-2 batch was pushed (CI run #11 proves it) with no
authorization quote in the checkpoints** (the entries say "Next: push", not
"authorized: user said ..."). Same shape as P2 — likely fine, needs your word.

**Observation — compare_ms variance.** 2048² compare wall measured 33–55 ms
across runs this session while compare_gpu stayed 6.5–7.9 ms. The doc's
"±15% noise" caveat understates compare-wall variance; the `*_gpu` columns
are the stable signals and should be the primary comparison basis.

### Pass 6 verdict

**VERIFIED WITH CAVEATS.** Round-2 perf work is real, tested, and honestly
stopped; the caveat is F34 — a detection bug whose consequences happen to be
benign-to-positive on this hardware but whose claims are wrong and whose
non-ReBAR behavior is unmeasured. One-line fix + an explicit A/B would turn
an accident into a result.

---

## Pass 7 — F34 fix + A/B (`61bd7c8`), P5 closure (`ca5321f`)

Pinned to `ca5321f`. The agent took the whole F34 point — including the
subtlety that `contains()` alone still returns true on ReBAR discrete (this
host exposes a 0x0007 DEVICE_LOCAL∩HOST_VISIBLE∩COHERENT type on BOTH GPUs).
Its resolution went beyond my suggestion: `contains()` + a `DSSIM_UNIFIED=0/1`
override to A/B both paths on one device, then MEASURE.

Reproduced the A/B myself (discrete, release bench):
- 2048² create: staging 86.8 vs zero-copy 66.1 ms (claim 91 vs 67 ✓)
- 4096² create: 333.1 vs 286.5 ms (claim 323 vs 278 ✓)
- dssim_check byte-identical across both paths ✓
- So zero-copy-on-ReBAR-discrete is now a measured decision, not an accident;
  non-ReBAR discrete falls back to staging by construction (no DL∩HV type).

Also verified: workspace green (exit 0 — first-run-after-rebuild this time,
so P3 did NOT recur for a 4th consecutive green), clippy `-D warnings` clean,
docs (VULKAN_PERF + PERF_EXPERIMENTS §T7) corrected honestly.

P5: closed in `ca5321f` with the user's quote from the implementing session
("push ล่าสุดฉันบอกนายไปเอง"), and the agent explicitly refused to treat that
as forward authorization — correct §5 reading.

P3: their 3 forced-rebuild cycles + my 2 runs since = all green; still
un-reproducible; their refusal to add a masking retry is the right call.
Status: dormant observation, capture the dying binary if it returns.

### Pass 7 verdict

**VERIFIED.** F34 fixed with more rigor than the finding asked for; every
number in the fix commit reproduces. No new findings. Open items: none in
code; `fbf67f8`+`61bd7c8`+`ca5321f` (+ this pass) await a push decision.

---

## Pass 9 — Tier 0/1/4/5 execution (`99f233a`, `93f14c3`, `921feea`)

Pinned to `e738cb4` (pushed; CI run #15 green — device-prep bitwise tests,
BH6 staging leg, BH5 concurrency test all pass on llvmpipe). Two verifier
sub-agents (Tier-5 transcription; Tier-4 + wholesale hygiene) + main-agent
dynamic runs.

**Reproduced:**
- Tier 0: NT-store microbench re-run — memcpy 5.25–5.54 GB/s vs NT
  5.28–5.29 (NT ~1–4% SLOWER, claim "~4% slower" ✓). Floor confirmed;
  SDMA correctly not green-lit per the plan's own gate.
- Tier 1: probes dropped (descriptor_heap absent, host_image_copy absent +
  zero VkImage objects, external_memory_host present-but-moot). No code —
  matches plan.
- Tier 4: batch per-modified 38.6→26.7 ms (N 1→10, claim 35.5→24.6 — same
  amortization shape, noise band); pin test exact-equality N=3, non-vacuous.
- Tier 5: create 2048² 68.3→18.1 ms (0.26×, claim 66→18 ✓), 4096²
  299→179 ✓. Transcription verified instruction-level by the sub-agent:
  CPU `(((a+b)+c)+d)*0.25` left-assoc f32 == SPV OpFAdd chain ×4 then
  VectorTimesScalar, NoContraction present, **OpFma = 0** (main-agent
  spirv-dis), odd floor-drop + host-side stop condition match, alpha
  averaged like RGBAPLU. SPV 8/8 fresh-match; dssim-core diff EMPTY (the
  red-flag check — CPU ground truth untouched); default path byte-identical
  (only additions: one pipeline at init + mode dispatch); workspace green,
  clippy clean, no test weakening in range.

**New findings:**

**BH36 (medium, API robustness) — `compare_many` lacks the BH3-style up-front
dims guard.** score.rs:724-735 loops `create_image(m)` (full GPU dispatch)
BEFORE `compare()`'s mismatch checks fire — and those are `assert_eq!`
PANICS, not `Error::InvalidInput`. One mismatched modified wastes its whole
pyramid build, then panics mid-batch discarding already-computed scores —
wrong failure mode for a Result-returning batch API. `create_image_pair` has
the correct guard (score.rs:180-194); `compare_many` should copy it (check
each modified's dims/channels against reference before any dispatch) + a
mismatch test. Not memory-unsafe (panic precedes the mismatched compare's
dispatch; split threshold reads modified's own size).

**BH37 (low, discoverability) — `PrepMode::Device` + gray is a silent no-op.**
`create_image_gray` never consults `prep_mode` (always CPU-downsamples); a
library caller combining `with_prep_mode(Device)` with gray inputs gets
silently-unoptimized behavior. CLI can't hit it (RGB-only path). Fix: doc
line or debug_assert at the gray entry. Related nit: `--gpu-prep=device`
without `--gpu` parses but is silently ignored.

**BH38 (low, process) — no automated .spv-freshness gate in CI.** The
"8/8 fresh-match" claims are manual-audit each pass (true every time so far,
including this one); ci.yml never recompares committed blobs vs glslc output.
One CI step (compile + hash-compare) would make the invariant enforced
rather than ritual.

**Approval note — RESOLVED 2026-09-07:** Tier 5 is a standing non-goal; the
checkpoint recorded "as the user authorized" without a verbatim quote. The
user has now confirmed in the auditing session ("ฉันอนุญาตจริง" — "I really
did authorize"). Authorization is genuine; the only residual gap is the
missing quote in the implementer's checkpoint entry (process, same shape as
P5's standing correction: quote authorizations verbatim at record time).

### Pass 9 verdict

**VERIFIED WITH CAVEATS.** Tier 0/1 executed exactly per the plan's gates
(including honoring the negative result); Tier 4/5 claims reproduce and the
bitwise-transcription claim survives instruction-level adversarial
inspection; the CPU reference is untouched. One real API-robustness gap
(BH36), two nits (BH37/BH38). Tier-5 authorization confirmed by the user
post-hoc (see approval note above).

### Pass 9 resolution (BH36/37/38 fixed, 2026-09-07)

- **BH36** (f1781f6): `compare_many` now guards reference channels (==3) and each
  modified's scale-0 dims up front, returning `Error::InvalidInput` before any
  dispatch (mirrors create_image_pair's BH3). Added `GpuSsimImage::channels()` +
  `phase_e_compare_many_rejects_size_mismatch`.
- **BH37** (f1781f6): `create_image_gray` documents Device-is-RGB-only + a
  `debug_assert_eq!(prep_mode, Cpu)` at the gray entry (release = correct CPU
  fallback, not a silent unoptimized surprise). `--gpu-prep` without `--gpu` now
  warns it is ignored.
- **BH38** (f1781f6): CI step installs `glslc`, recompiles every `.comp` with the
  same `--target-env=vulkan1.3 -O`, byte-compares to the committed `.spv`, fails
  on any diff. Verified locally: all 8 `.comp` fresh-match. Non-vacuous by design
  (fails loudly if glslc is unavailable rather than skipping).

Tier-5 authorization quote: the implementer's checkpoint said "as the user
authorized" without a verbatim quote (same shape as P5); the user confirmed in the
auditing session. Standing rule already recorded (quote at push/record time).
