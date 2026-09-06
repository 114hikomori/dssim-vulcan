# Deep Bug Hunt — dssim-vulkan (post-M10, round 2)

Date: 2026-09-07. Auditor: opencode (main agent) + 5 parallel read-only sub-agents.
Pinned to HEAD `772233c`. **Record only — no code was changed.** This hunt
targets bugs that could still hide AFTER audit passes 1–7 (F1–F34, P1–P5),
with emphasis on interactions between the recently stacked features:
GPU-resident pyramid × split-submit (F28) × zero-copy (T7/F34) ×
create_image_pair (T11) × CLI.

Method: 5 sub-agents analyzed disjoint zones (orchestration / plumbing /
shaders / wrapper layer / CLI+tests) statically; every load-bearing claim was
then re-verified by the main agent at the cited lines. Legend:
- **[V]** verified by main agent at the cited code
- **[A]** agent-reported, consistent with code but not independently re-derived
- **[C]** needs a runtime/CI observation to settle

Numbering `BH1..` (bug-hunt); cross-references to the F-series where related.

---

## HIGH

**BH1 [V] — Error path frees a still-RECORDING command buffer; descriptor-pool reset is skipped on error.**
`context.rs:383-392` (`submit_one_shot`): if `record(cb)` fails after
`begin_command_buffer` succeeded, the cleanup calls
`free_command_buffers` on a buffer still in the recording state — spec
violation (VUID-vkFreeCommandBuffers-pCommandBuffers-00048). Reachable
triggers are the code's OWN guards: `record_pass` push-overflow
(`pipeline.rs:166-168`) and `allocate_descriptor_sets` failure past the
128-set cap (`pipeline.rs:113` documents this failure mode). Compounding:
`dispatch_sequence`'s `?` (`pipeline.rs:243`) returns BEFORE the reset loop
(`pipeline.rs:343`), so every failed sequence permanently leaks descriptor
sets toward the cap — after enough failures the pipeline is bricked until
process exit. Split mode multiplies exposure (one sequence per scale).
F2's "cleanup on every path" fix closed the leak but not these two states.
Fix shape: `reset_command_buffer` (or end-then-free) on the record-error
path; run the pool-reset on error too (e.g. RAII guard like ResetPool but
for the whole function).

**BH2 [V+C] — `maxComputeWorkGroupCount` is never queried; ≥2048² dispatches exceed the spec-minimum limit.**
`groups = ceil(w*h/64)` everywhere (`blur.rs:77/98/122`, `color.rs:131`,
`ssim.rs:127/186`); zero limit queries in `context.rs` (grep: 0 hits).
Vulkan guarantees only 65535 per axis → 4,194,240 pixels; 2048² =
4,194,304 px needs 65,536 groups — ONE over the floor; 4096² needs 262,144.
Exceeding it in `vkCmdDispatch` is UB (no VUID, validation stays silent).
AMD/NVIDIA report 2³¹−1 and llvmpipe reportedly higher than the floor —
which is why no local run or CI fixture has ever hit it — but a
minimum-conforming driver (some MoltenVK/embedded configs; llvmpipe value
needs confirming [C]) silently mis-executes at ≥2048² via `--gpu`.
Fix shape: read `props.limits.max_compute_work_group_count[0]` at context
creation; error (→ CPU fallback) when `groups` exceeds it; same for
`maxStorageBufferRange` (4096² `img_all` = 201 MB > 128 MiB floor [A]).

---

## MEDIUM

**BH3 [V] — `create_image_pair` has no same-size precondition and derives the split threshold from `reference` only.**
`score.rs:122-131`: `flush_per_scale = reference.width()*reference.height()
>= split_min_pixels` is computed from ONE image and applied to both. A
small-reference + large-modified call builds the ENTIRE large pyramid in
batch mode — exactly the transient pile-up F28 exists to cap — and the
mismatch only surfaces later as a PANIC in `compare` (F29 asserts), not a
`Result::Err`. Public API; only the bench calls it today (always same-size),
so it is an unguarded footgun, not a live failure. Fix: assert equal dims up
front (return Err), compute `flush_per_scale` per image (max of the two).

**BH4 [V] — Fence destroyed / cmdbuf freed while the submit may still be pending when `wait_for_fences` fails.**
`context.rs:399-413`: cleanup after the submit block runs unconditionally;
if `queue_submit` succeeded but `wait_for_fences` errored (device-lost is
the realistic case on this host — see P3), `destroy_fence` violates
VUID-vkDestroyFence-fence-01120 and the free violates 00047. Mostly moot
once the device is lost, but validation flags it and non-LOST wait errors
exist. Fix: on wait error, `wait_idle()` best-effort before destroying.

**BH5 [V] — `Context`/`GpuSsim` are `Sync` but the submit path has zero internal synchronization.**
All fields are Sync (ash handles, `Mutex<Allocator>`, atomics), so
`Arc<Context>` crosses threads and `&self` methods invite concurrent use —
yet `dispatch_sequence` races on the shared `command_pool`, `queue_submit`,
per-pipeline descriptor pools, and the 2-query `perf_query_pool` (wrong
timings). Tests avoid it only by convention (GPU_LOCK). Today's CLI is safe
(GPU calls on one thread), but a future batch-mode parallelization would
corrupt silently. Fix shape: a submit mutex in Context, or make it
provably single-thread (`PhantomData<*mut ()>`) + doc.

**BH6 [V] — `DSSIM_UNIFIED` is unasserted and unlogged; the staging upload path likely has ZERO CI coverage.**
Grep: no test references `DSSIM_UNIFIED` or `is_unified_memory` (0 hits).
llvmpipe reports DEVICE_LOCAL|HOST_VISIBLE → CI takes zero-copy; the
staging `Pass::CopyBuffer` branch (`score.rs:183-197`) never runs there.
An operator exporting `DSSIM_UNIFIED=0/1` silently retargets every "GPU
verified" claim — the exact invisibility F26 fixed for validation, now
reproduced for the upload path. Also `=2`/`=true` silently fall through to
detection (only exact "0"/"1" honored). Fix: eprintln the effective path at
Context creation (like the validation line); add a CI leg with
`DSSIM_UNIFIED=0`.

**BH7 [V] — gpu_cli's 8-decimal format check only inspects CPU stdout.**
`tests/gpu_cli.rs:120-129`: `for line in &cpu.stdout` — the GPU output's
formatting is never asserted; `parse_scores` accepts scientific/short forms.
If the GPU print macro drifted (`{:.6}`), the parity diff would still pass
and the "identical format" claim rots silently — the F7-class rot this test
was hardened against. Fix: loop over `gpu.stdout` too (one line).

**BH8 [V] — `gpu_ssim_identity_is_exactly_one` can pass vacuously.**
`ssim_parity.rs:242-277`: the scale loop is gated on `read_dump(...).is_some()`
with hardcoded run numbers; if dump numbering shifts (extra `next_run`, new
hook), zero iterations run and the test passes having checked nothing. The
sibling test guards with `assert!(map_checks >= 20)` (line 217); this one has
no floor. Fix: `assert!(scale > 0)` after the loop.

**BH9 [V] — README + lib.rs comment claim "16-bit inputs are 8-bit on the GPU path"; the code does full 16-bit.**
`src/lib.rs` `load_image_rgba` routes RGBA16/RGB16/GRAY16/GRAYA16 through
`to_rgbaplu()`, and `dssim-core/src/linear.rs:72` gives u16 a
`[f32; 65536]` LUT — full precision, no truncation found anywhere in the GPU
decode path. The stale note (and the README caveat I endorsed in pass-1 F20)
under-claims AND suppresses the obvious test: nothing pins 16-bit CLI parity
in either direction. Fix: verify empirically (16-bit pair, CPU vs --gpu),
then correct the docs or find the truncation I missed.

**BH10 [V] — Every `new_inner` error path after `create_instance` leaks the VkInstance + debug messenger.**
`context.rs:150-296`: ash 0.38 has no Drop for Instance/Device (verified in
registry source); `enumerate_physical_devices` failure, `NoDevice` (168-170),
`create_device`/`create_query_pool`/`create_command_pool`/`Allocator::new`
failures all return Err without destroying. The `NoDevice` path is a
SUPPORTED production path — every device-less `--gpu` run and the fallback
test hit it. Process exit hides it in the CLI; in-process probes/tests
accumulate. Fix: build the instance last / tear down on error paths.

**BH11 [V] — No buffer-size validation in ANY `*_into` wrapper (systemic root cause of both historical M10 bugs).**
`h5_into`/`v5_into`/`h5_mul_into`/`lab_into`/`combine_into` take Buffers +
offsets and never check `buffer.size` against what the shader will index;
descriptors bind the WHOLE buffer (`pipeline.rs:185`), so a wrong offset is
in-allocation OOB — invisible to standard validation (GPU-Assisted is off).
The legacy host-side wrappers DO assert input lengths (`blur.rs:129-132`);
the GPU-resident ones — the production path — do not. The repo hit this
exact class twice (M10 bug-1, F29). Fix: `debug_assert!` required-size
formulas per wrapper (formulas exist in the sub-agent's call-site table).

**BH12 [V] — `h5_mul_into` hardcodes src2_stride = stride1; the shader supports independent strides.**
`blur.rs:125` passes `stride1` twice to `pc_mul_bytes`. The CPU reference
semantics (and the passing `gpu_blur_mul_strided_parity` test with
stride1≠stride2) prove mismatched strides are a supported case; the fused
GPU-resident API cannot express it, and the constraint is undocumented at
the wrapper. F29's compare-side assert guards only today's call site.
Fix: thread a real `stride2` (zero shader change) or debug_assert + doc.

---

## LOW

**BH13 [V] — `-o` maps are dropped even when `--gpu` falls back to CPU.**
`main.rs:154-164`: the "ignoring -o" warning prints BEFORE `Context::new()`;
if the context fails, `run_cpu_simple(files)` — a CPU path that could write
maps — never receives `map_output_file`. Exit 0, no maps, misleading
warning. Fix: pass map_output into the fallback, or error on `-o`+`--gpu`.

**BH14 [V] — `--gpu` is silently ignored in gpu-less builds.**
`main.rs:76`: `opt_present("gpu") && cfg!(feature = "gpu")` — no warning,
help still advertises the flag. Fix: eprintln a warning when the flag is
given but the feature is off.

**BH15 [V] — bench headline measures `create_image_pair`; the CLI runs two separate creates.**
`bench.rs:128/133` use the pair (one create-side fence); `run_gpu` uses
`create_image`×2 + compare. The comment at line 124 even says "same shape"
while changing the shape. The headline ratio overstates the CLI path by the
fence savings. Fix: add a `cli_shape_ms` column or measure both.

**BH16 [V] — The F29 dimension/channel asserts in `compare` are themselves untested.**
No test drives a mismatched pair through `compare` (CLI guards upstream, so
it never reaches them either). A refactor deleting the assert ("CLI already
checks") would silently reintroduce the GPU-OOB hazard with no test failure.
Fix: two `#[should_panic]` tests (RGB-vs-gray mod; 64×64 vs 64×63).

**BH17 [V] — The CLI never exercises split-submit mode.**
Split triggers at ≥6M px (~2450²); every gpu_cli fixture is `*-sm.png`. The
library seam proves split==batch==CPU at small sizes, but the CLI
combination (streaming original held across N modifieds, allocator pressure
at 4K, allocator-error → exit 1 with NO CPU fallback for OOM) is untested.
Fix: an `#[ignore]`d real-4K CLI test or a generated 2450² PNG on CI.

**BH18 [A] — Pair batch mode ≈ 2× the F28 peak-VRAM guarantee.**
`SPLIT_SUBMIT_MIN_PIXELS=6M` was calibrated for ONE image; the pair path
accumulates both pyramids' transients in one submit below the threshold.
Halve the effective threshold for pairs (or per-image flush flags — see BH3).

**BH19 [V] — `MAX_SETS_PER_POOL` comment undercounts the pair worst case.**
Comment says worst case 40 ("3× headroom"); `create_image_pair` doubles
per-pipeline usage (v5: 8×5×2 = 80). Still < 128 — cap holds — but the
comment that instructs future maintainers is stale, and the exceedance
failure mode is BH1's illegal-free path, not a clean error.

**BH20 [V] — `blur_v5.comp` advertises `src_stride` (dims.z) but never reads it.**
grep: 0 `dims.z` uses in the shader; every read uses `w`. All callers pass
tight src today, so behavior is correct — but the field invites a future
strided caller into silent row corruption (the F33 class, relocated to the
shader side). Fix: use dims.z or delete the field from the contract.

**BH21 [V] — `pc_bytes_off` doc says "byte offsets"; the values are ELEMENT offsets.**
`blur.rs:269-271` self-contradicts ("src_off_elems" in the same comment). A
maintainer "fixing" a caller to pass `off*4` shifts plane reads by 3× the
plane size — in-descriptor-range, validation-blind OOB. Also
`blur_h5.comp:43-44` overstates a "3-element padding" contract that the
clamp logic makes unnecessary.

**BH22 [V] — `combine_into`/`ssim_combine_pipelines`: parameter order ≠ binding order.**
Signature `(mu_o, sq_o, mu_m, sq_m)` vs binding `[mu_o, mu_m, sq_o, sq_m]`
(`ssim.rs:86-96` vs `:125`, same split in `combine_into`). All five stat
buffers are the same length, so the length asserts cannot catch a swap —
the result is a finite, plausible, WRONG map. Both call sites are correct
today; this is one refactor from an F33-class silent corruption. Fix:
reorder the binding vec or rename params to bind-order.

**BH23 [V] — `lab_into` has no `channels ∈ {1,3}` assert.**
The assert lives in `rgba_to_lab_gpu` only; `lab_into` is called directly by
`score.rs`. `channels=2` would take the shader's gray branch on an
interleaved buffer — silent garbage, no OOB. In-crate callers pass literals
today. Fix: move the assert into `lab_into` (single choke point).

**BH24 [V] — `blur()`/`blur_mul()` don't assert `stride >= width`.**
The length assert passes for `stride < width`; rows then overlap and the
shader reads in-bounds garbage where the CPU (imgref `new_stride`) panics.
Test-only wrappers, but `pub`. Fix: one assert each.

**BH25 [V] — `alloc_buffer` leaks the VkBuffer when `bind_buffer_memory` fails.**
`transfer.rs:106-108`: the allocate-failure arm destroys the buffer; the
bind `?` does not. Rare failure, trivial fix (mirror the arm).

**BH26 [V] — `assert!`s inside the record closure bypass all cleanup.**
`pipeline.rs:165` (binding-count) and `:271-275` (F32 size assert) panic
AFTER `begin_command_buffer` — fence + cmdbuf leak, buffer left recording,
and `Context::drop` then destroys the pool with a live buffer. The record
API's contract is `Result`; these two checks break it (the push-overflow
check right below correctly returns Err — inconsistent handling of the same
caller-bug class). Fix: return Err instead of assert inside the closure.

**BH27 [V] — Mapped I/O bounds are `debug_assert`-only; `read_bytes` has no check at all.**
`transfer.rs:276/294` vanish in release; a caller bug becomes a silent OOB
copy into mapped memory. All current call sites are exact-sized (verified),
so this is hardening. `read_bytes(buffer, len)` trusts `len` entirely.

**BH28 [V] — `gpu_elapsed_ms` has no `timestamp_period_ns == 0` guard.**
`context.rs:343-346`: a driver reporting period 0 makes every T9-lite
GPU-busy number silently 0.00 ms — the exact number the T5/T6a refutation
rests on. One `assert`/fallback closes it.

**BH29 [A] — Allocator-mutex poisoning asymmetry.**
`CREATE_LOCK` survives poisoning; allocator locks `.expect(...)` — a panic
under the lock makes every later Buffer drop panic (abort risk), and
`Context::drop`'s `if let Ok` then skips allocator teardown, destroying the
device with live allocations. Requires a prior panic; low but real.

**BH30 [V] — `record_pass` permits short pushes and non-multiple-of-4 sizes.**
Only overflow is checked (`pipeline.rs:166`); a short push leaves shader
constants tail-undefined (silent wrong values), and size%4≠0 violates
VUID-vkCmdPushConstants-size-00369. All builders emit exact sizes today.

**BH31 [V] — Algorithmic constants duplicated outside the gpu-reference channel.**
`rgba_to_lab.comp`: `0.2/1.51/-0.5` (cbrt poly), `* 1.16` (gray branch),
`255.0/1.0` dither literals; `ssim.rs:32`: `0.01²`, `0.03²`, `1/3`. All
verified equal to dssim-core TODAY, but a CPU-side change drifts silently —
K5/LAB_GPU_CONSTANTS already established the push-constant channel; extend it
(or add a static-assert test) so shaders hold zero algorithmic literals.

**BH32 [A] — `numer==denom → 1.0` select diverges from CPU for non-finite planes.**
CPU does plain division (Inf/Inf → NaN → huge finite dssim); GPU yields 1.0
for the ±Inf case. Unreachable for finite inputs (denom > 0 proven); only a
corrupted upload could trigger it. Note, not action.

**BH33 [V] — Stale module docs.** `blur.rs:1-6` ("Each is one submit with two
passes") and `ssim.rs:4` describe the pre-M10 world; the production path uses
neither `blur_gpu` nor `ssim_combine_gpu` anymore.

**BH34 [V] — `create_image_pair` × split mode has zero test coverage.**
The split seam test uses single creates; the pair test runs at the default
(batch) threshold. The T11×F28 intersection (per-scale flush into a SHARED
passes Vec across two images) is correct by inspection but untested — and
BH3/BH18 hide precisely in this gap.

**BH35 [V] — CLI coverage matrix gaps (aggregate).** Untested at CLI level:
gray PNGs (expected-pass by analysis: both paths route gray→3ch; unproven),
alpha end-to-end, 16-bit (see BH9), 1-vs-N streaming (loop body never runs
>1 iteration), identity exact-string `0.00000000`, decode-error text/exit,
`-o`+`--gpu` (BH13), tiny RGB end-to-end (1×1/7×7 via create+compare — only
blur-level tiny sweeps exist), odd dims at ≥6M px.

---

## Checked and found CORRECT (condensed — so the next pass need not redo it)

- **No shader OOB against any current caller**: max-index algebra verified for
  all 7 shaders × all 19 call sites (plane offsets, taps clamped to
  `last`, zero-copy RGBA read bound `4P-1`, combine plane reads `CP-1`).
- **Barrier coverage**: the over-barriered design holds for every pass type
  incl. CopyBuffer and final readback; no missing write→read edge.
- **F1 flush/invalidate** ordering correct incl. zero-copy record-time writes;
  gpu-allocator CpuToGpu prefers HOST_VISIBLE|COHERENT (+DEVICE_LOCAL where
  available) — F34's discrete win is real and the fix is complete.
- **Lifetimes**: Arc-in-passes keep transients alive past scope until after
  the fence; split flush precedes local drops; error-path `?` unwinds are
  drop-safe (no VRAM leak; descriptor-pool gap is BH1).
- **compare() semantics**: F29 asserts precede every mod-buffer index;
  scale-count mismatch unreachable via public API (deterministic downsample);
  readback order == weight order; `pool_scale`/`to_dssim` transcription
  op-for-op incl. mul_add order and the inherited NaN quirk.
- **Push-constant layouts**: BlurPC/LabPC/SsimPC repr(C), zero padding,
  const size asserts, field-for-field vs shader blocks and SPV offsets;
  h5_mul dims2 order exact.
- **Edge sizes h∈{2..6}, w==1**: branch precedence reproduces CPU's
  sequential-if overwrite pattern; all shaders have the idx>=w*h guard.
- **K5/LAB constants**: no hardcoded duplicates in the vulkan crate (the
  drift risk is the OTHER literals, BH31).
- **CLI decode/ICC/alpha dither transcription, error chain (F6), fallback
  ordering (F7: device line strictly after both constructors succeed),
  gpu_cli env hygiene (subprocess-only), CI validation gate soundness.**
- **dump_goldens**: CPU-only, regenerated at runtime — no stale GPU-structure
  goldens.

## Recommended fix order (NOT applied — record only)

1. BH1 (illegal free + pool leak — the failure mode BH19/BH26 route INTO),
   then BH26/BH25/BH30 (same error-path family, one PR).
2. BH2 + BH28 (limits + timing sanity; needs one CI probe [C]).
3. BH7 + BH8 + BH16 + BH34 (test-integrity: vacuous-pass and untested-guard
   classes — cheap, prevents the next regression from hiding).
4. BH3 + BH18 + BH19 (pair-path contract + recalibrated thresholds).
5. BH6 (upload-path observability + CI leg) — same lesson as F26, new surface.
6. BH11 + BH12 + BH20-BH24 + BH30 (wrapper hardening sweep — one style PR).
7. BH9 + BH13 + BH14 + BH15 + BH33 + BH35 (docs/CLI honesty + coverage).
8. BH4/BH5/BH10/BH29/BH31/BH32 (judgment calls; BH5 before any batch mode).

## Caveats of this hunt

- Static analysis only; nothing was executed except read-only greps (GPU
  contention policy §7). BH2's device-limit values and BH6's CI-path claim
  need one runtime confirmation each [C].
- Sub-agent reports were spot-verified at every load-bearing line; [A] items
  are the residual-risk list, not confirmed defects.
- Severity reflects mechanism × reachability today; several MEDIUMs are
  "public API footgun, no current caller hits it."

---

## Resolution (fixes applied, 2026-09-07)

All BH1–BH35 dispositioned across 8 grouped commits; every step verified with
the full workspace green **with validation** (0 Vulkan errors, 15 suites) +
clippy `--no-deps -D warnings` clean. No `.spv` bytes changed (shader edits were
comment-only).

| BH | fix | commit |
|---|---|---|
| BH1 | reset-before-free + PoolReset Drop guard (pool reset on every exit path) | 1ca1748 |
| BH2 | query+enforce workgroup/storage limits; CLI CPU-fallback | 5fd7a2c |
| BH3 | create_image_pair same-size guard (Error::InvalidInput) | ec26776 |
| BH4 | device_wait_idle before fence destroy on wait error | 1ca1748 |
| BH5 | submit_lock mutex; proven by 4-thread concurrency test | f012a2b |
| BH6 | log effective upload path + warn bad override + CI staging leg | 9673f72 |
| BH7 | format check on GPU stdout too | 89d3150 |
| BH8 | assert!(scale>0) floor on identity test | 89d3150 |
| BH9 | 16-bit is full precision; docs corrected + precision test | ab5cdd3 |
| BH10 | VkGuard tears down instance/device/messenger on error | 1ca1748 |
| BH11 | debug_assert required-size formulas per *_into wrapper | 07be109 |
| BH12 | h5_mul_into stride2==stride1 assumption documented + sized | 07be109 |
| BH13 | -o honored in both CPU fallbacks; warning moved post-context | ab5cdd3 |
| BH14 | warn when --gpu given but feature off | ab5cdd3 |
| BH15 | bench cli_ms/cli_ratio column (real CLI shape) | ab5cdd3 |
| BH16 | two #[should_panic] tests for the F29 compare asserts | 89d3150 |
| BH17 | #[ignore]d 2500² CLI split-path test (verified: diff 1.4e-7) | f17d515 |
| BH18 | halved pair threshold (peak comparable to single-image F28) | ec26776 |
| BH19 | MAX_SETS comment: pair doubles worst case to ~80 | 1ca1748 |
| BH20 | blur_v5.comp: dims.z unused, contract made explicit | 07be109 |
| BH21 | pc_bytes_off element-not-byte offsets; h5 padding note fixed | 07be109 |
| BH22 | combine params reordered to bind-order (behavior-preserving) | 07be109 |
| BH23 | channels assert moved into lab_into | 07be109 |
| BH24 | blur()/blur_mul() assert stride >= width | 07be109 |
| BH25 | alloc_buffer frees allocation+buffer on bind failure | 1ca1748 |
| BH26 | record-closure asserts converted to Err returns | 1ca1748 |
| BH27 | mapped-I/O bounds hard asserts; read_bytes checked | 1ca1748 |
| BH28 | gpu_elapsed_ms returns NaN when timestampPeriod==0 | 1ca1748 |
| BH29 | poison-tolerant lock_allocator() at all four sites incl. Drop | f012a2b |
| BH30 | record_pass requires exact, 4-aligned push | 1ca1748 |
| BH31 | constant duplication documented; parity suites are the drift guard; full push-constant migration deferred (judgment call) | f012a2b |
| BH32 | non-finite select divergence documented (note-only, per audit) | f012a2b |
| BH33 | blur.rs/ssim.rs module docs updated to the *_into production path | ab5cdd3 |
| BH34 | create_image_pair x split test | 89d3150 |
| BH35 | CLI identity/1-vs-N/decode-error/gray + tiny-RGB tests | ab5cdd3 |

Deferred by design (audit's own ordering): BH31 full push-constant migration
(refactor risk; drift already caught by parity), BH5 was fixed with a mutex
rather than deferred since it's cheap and makes the Sync claim honest.

---

## Resolution verification (pass 8, 2026-09-07) — fixes `1ca1748..f490b23`

Method: 3 read-only verifier sub-agents (one per fix group) + main-agent
dynamic runs. Verdicts below are OUR re-verification, not the fixer's claims.

**Dynamic (main agent):** workspace green exit 0 with counts matching every
claimed addition (gpu_cli 4→8, phase_e 6→12, 1 ignored = BH17); clippy
`-D warnings` clean; `.spv` files untouched in the range AND 7/7 still match
fresh glslc recompile (comment-only shader edits confirmed); BH6's new log
line observed live (`upload path: zero-copy` ×5 on the ReBAR discrete —
consistent with F34); BH17's ignored 2500² CLI test run: `diff=1.400e-7`
reproduced exactly.

**CLOSED (28):** BH1 (reset-before-free + PoolReset Drop under submit-lock
after fence-wait — ordering proven sound, no double-free), BH2 (limits
queried + CLI pre-check `supports_size` + dispatch-time Err; fallback Cell
checked BEFORE channel-error mapping — decoder "Aborted" can't mask it),
BH3/BH18 (same-size Err before any alloc; threshold halved; seam respected),
BH4, BH5 (submit_lock covers the full critical section; lock order
submit→allocator proven one-way, no deadlock path exists; 4-thread test
real), BH7 (would now catch {:.6} drift — adversarially traced), BH8,
BH13 (both fallbacks write maps), BH14, BH15, BH16 (should_panic with
message-matched expected, fires before any alloc), BH19, BH25 (no
double-free — Allocation moved), BH26 (zero asserts left in record
closures), BH27, BH28, BH29 (all four sites), BH30 (exactness triple-pinned
per pipeline — no legitimate caller broken), BH31/BH32 (notes landed),
BH33, BH34 (non-vacuous: exact f64 equality + CPU parity).

**PARTIAL (4):**
- **BH10** — `create_debug_utils_messenger` failure (`context.rs:216-218`)
  still `?`s BEFORE the VkGuard exists (229): instance leaks on that one
  narrow path. Fix shape: construct guard right after `create_instance`
  with `messenger: None`.
- **BH9** — docs corrected + decode-precision test added, but the audit's
  prescribed 16-bit CPU-vs-GPU *parity* test (CLI or library) does not
  exist; the gap that hid the stale claim is still open.
- **BH17** — test passes and the 1.4e-7 diff reproduces, but nothing makes
  split execution OBSERVABLE in-test (no log/counter): if the threshold
  wiring regressed, it would silently pass in batch mode. The 1.4e-7 number
  lives in an eprintln; the assert is ≤5e-6.
- **BH35** — decode-error exit codes asserted both paths; message TEXT not
  asserted (half the stated gap).

**Honesty nits (wording, not mechanism):** CHECKPOINT says the concurrency
test proves "all-byte-identical" — it asserts 5e-6 parity; BH31's commit
message claims 1.16/255.0 notes landed — those comments predate it. BH1's
guard intentionally swallows pool-reset errors on the success path
(documented trade-off, slightly weaker reporting).

**Test-weakening hunt over `bacc4a8..HEAD`: CLEAN** — every removed line
accounted for, no tolerance constant moved, no new skips beyond the declared
BH17 `#[ignore]`.

**Open residuals:** BH6's CI staging leg has NEVER run (whole range
unpushed — first push is its real test); llvmpipe's actual
`maxComputeWorkGroupCount` still [C]; untracked `PHASE_H_CONSOLIDATED_PLAN.md`
+ `archive/phase-h-addons/` (research consolidation, commit-worthy-looking,
human call); 11 commits ahead of origin awaiting a push decision.

### Pass 8 verdict

**VERIFIED WITH CAVEATS.** 28/32 actionable findings provably closed with no
new bugs introduced (the two riskiest fixes — PoolReset ordering and BH30
push-exactness — were the ones we dug at hardest; both hold); 4 PARTIALs are
small, honestly-close-to-claimed, and listed above with fix shapes. The
resolution table's claims match the code.
