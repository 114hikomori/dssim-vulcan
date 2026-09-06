# Phase H — Techniques Not Found in the Original Research Pass

Date: 2026-09-07. Companion to `PERF_EXPERIMENTS_PHASE_H.md` (pass 4, pinned
`36868da`) and `VULKAN_PERF.md` "Round 2". Research-only document: no code
changed here, and no code was opened for this pass either — see §5. Scope is
deliberately narrow: **only** techniques that do not appear anywhere in the
source document — not in S1–S12, not in its "Research delta this pass" list,
not in the pass-4 addendum, and not already named as one of T1–T11. Every
item below was cross-checked against that document before inclusion; two
things that *are* mentioned there (dedicated transfer queues, specialization
constants) are explicitly excluded for that reason and are not repeated here.
Citations (S13+) continue the source document's numbering scheme but are
scoped to this file only.

## 1. Target profile (inherited, not re-measured)

Per the pass-4 addendum in the source document: GPU-busy time is only ~5% of
`create` wall time at 2048x2048 (93.4 ms wall vs 5.0 ms GPU-busy); the
dominant cost is host-side `prep` (~70–90 ms, ≈1 GB/s for ~89 MB of pyramid
uploads). T5 (shader dispatch geometry) was demoted on this evidence. That
profile — CPU-bound host-side data movement, not GPU compute, not submit
count — is what the search below was aimed at; it is why most of what
follows is about *how data gets from host memory into GPU-visible memory*,
not about shader code.

## 2. New candidates

### T12 — Diagnose the prep loop against write-combining pitfalls, not just memory-type selection

Distinct from T7/S2's question of *which* memory type to allocate
(DEVICE_LOCAL vs HOST_VISIBLE vs both): this is about *how* the prep loop
writes into whichever host-visible pointer it already has, once that memory
is uncached/write-combined (WC) — which any host-visible memory on a
non-UMA discrete GPU normally is.

Regular CPU stores to WC memory are buffered in a small, fixed number of
per-core write-combining/line-fill buffers before being flushed as a single
burst (S14). If a loop touches more distinct cache-line destinations at once
than there are buffers — e.g. writing to several separate plane/channel
buffers in an interleaved, round-robin order, rather than finishing one
destination before starting the next — buffers get evicted before they
fill, and each partial write goes out as its own small transaction instead
of one combined burst (S13, S14). Any *read* of WC memory is far more
expensive still, and can silently appear in code that looks write-only —
e.g. a struct field read-modify-write, or an accumulate-in-place idiom —
which is specifically called out as a common, hard-to-spot cause of this
exact kind of regression (S13). One practical, if dated, data point: a CUDA
forum thread measured host-to-device bandwidth over WC memory that was *no
better or worse* than ordinary pinned memory on some chipsets — i.e. the
common assumption that WC always speeds up PCIe transfer doesn't
universally hold, and is worth measuring rather than assuming (S15).

Concretely testable against your own profile: ~1 GB/s for a supposedly
sequential f32 interleave is roughly in the range this failure mode
produces (normal sequential WC writes should be much closer to full memory
bandwidth). `MOVNTDQA`/`_mm_stream_load_si128`-style non-temporal loads can
read WC memory efficiently where a normal load cannot (S13) — relevant if
`prep` ever reads back anything it just wrote, and non-temporal SIMD stores
(`_mm256_stream_ps` and friends, available via `core::arch::x86_64` in Rust)
are the standard fix for the write side once destinations are kept
sequential per buffer.

Expected: if this is the actual mechanism, fixing it is a pure host-side
loop reorder — finish each destination buffer before touching the next,
avoid any read of the mapped pointer, add non-temporal stores at the hot
loop — with no interaction with GPU-side FP semantics at all.
Risk: none to parity (touches no shader, no per-pixel op order). Risk is
entirely in correctly identifying whether this is actually what your loop
does — see §5.
Exit: a synthetic microbenchmark (sequential non-temporal writes into the
same mapped pointer, same size, outside the real pipeline) either
reproduces ~1 GB/s (mechanism confirmed elsewhere) or hits several GB/s
(mechanism confirmed here, loop rewrite justified). This can be built and
run in isolation before touching the real prep code.

### T13 — Import the prep buffer directly instead of copying through a Vulkan-owned mapped buffer

`VK_EXT_external_memory_host` lets an application-owned host allocation be
imported as a `VkDeviceMemory` object directly — `vkGetMemoryHostPointerPropertiesEXT`
reports which memory types a given pointer qualifies for, then
`VkImportMemoryHostPointerInfoEXT` (handle type `HOST_ALLOCATION_BIT_EXT` or
`HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT`) binds it (S19). The application keeps
writing through its own pointer; the GPU reads the same bytes with no
separate copy step at all.

This is a different mechanism from T7's UMA detection: T7 finds cases where
a *Vulkan-allocated* memory type happens to be both device-local and
host-visible. T13 instead exposes memory *the application already owns and
already wrote into* to the device, which can apply even where T7's
specific memory-type intersection doesn't exist. The pointer must be
aligned to `minImportedHostPointerAlignment` (queried per-device), and
ownership/synchronization stays the application's responsibility (S19) —
so this composes with, rather than replaces, whatever staging-buffer
lifetime logic already exists.
Risk: medium — changes who owns the allocation backing the "staging"
concept; import can fail per-platform (spec explicitly allows
`ERROR_INVALID_EXTERNAL_HANDLE_KHR`), so needs a capability check and a
fallback path, not a hard requirement.
Exit: `vkGetMemoryHostPointerPropertiesEXT` returns a non-empty type mask
for your actual prep-buffer allocation on your own hardware (this alone
answers the applicability question T12 can't); if it does, bench delta at
2048x2048 vs current staged path.

### T14 — VK_EXT_host_image_copy, conditional on a buffer-vs-image inventory

Core-optional in Vulkan 1.4; lets `vkCopyMemoryToImageEXT` write host
memory straight into a `VkImage` with no staging buffer, no command
buffer, and no submit — the CPU does the copy (and any layout swizzling)
directly (S16, S17). Requires the target format to advertise
`VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT`, and the image must be
created with `VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT`; a `MEMCPY` fast path
exists when the source data is pre-swizzled to the driver's own tiling
layout (S17).

Freshly relevant on your exact GPU generation: RADV enabled this by
default for RDNA2 and newer as of a Mesa update this year, citing a new
AVX2-accelerated swizzling path in AMD's ADDRLIB reaching roughly 20 GiB/s
on comparable RDNA hardware (S18) — this is a single commit-message
figure, not independently reproduced, treat it as directional only per
your own document's convention for single-sourced numbers.

The blocking question this document cannot answer: your own T1 write-up
describes GPU-resident **buffers** (tmp, mu, sq, cross), not images, for
the comparison pyramid. This extension only helps where the upload target
is (or could be) a `VkImage`. I have not opened your source tree, so I
cannot say whether any upload target is image-backed today — see §5.
Risk: high relative to T12/T13 — if nothing in the pipeline is image-shaped,
adopting this means introducing images where there were none, which is a
resource-model change, not a drop-in.
Exit: grep your own code for `VkImage`/image-backed uploads first; only if
that turns up a real target, check `vkGetPhysicalDeviceFormatProperties2`
for `HOST_IMAGE_TRANSFER` support on the relevant formats on your device.

### T15 — VK_EXT_descriptor_heap as a fresher answer to T3

The source document's T3 scopes descriptor overhead reduction as either
pre-allocated sets or dynamic storage-buffer offsets (S8). A newer,
more aggressive option exists: `VK_EXT_descriptor_heap`, ratified in
Vulkan 1.4.340 (January 2026), which removes descriptor sets, descriptor
pools, `vkAllocateDescriptorSets`, and `vkUpdateDescriptorSets` entirely —
descriptors become plain data in one sampler heap and one resource heap
that the application writes to directly and the shader indexes by offset
(S20). It was developed specifically to fix portability and performance
issues found in the older `VK_EXT_descriptor_buffer` extension (S20), so
it supersedes rather than complements that one.

It is not vaporware for your stack specifically: AMD's own Windows driver
release notes list `VK_EXT_descriptor_heap` under "Expanded Vulkan
Extension Support" (S21), and RADV (Linux) landed support in Mesa 26.1
behind an experimental flag and made it default in 26.2 (S22) — i.e. both
your Windows dev driver and the Linux/CI side have a path to this today.
A compatibility mode maps existing set/binding decorations onto heap
offsets, so shaders do not strictly need a rewrite to raw heap indexing on
day one (per the extension's own binding-interface option, S20).
Dependencies to plumb: `VK_KHR_maintenance5`, `VK_KHR_buffer_device_address`,
and (for direct heap indexing) `VK_KHR_shader_untyped_pointers`.
Risk: medium — newer extension (ratified this year), so validation-layer
and tooling maturity should be checked before depending on it; the
dependency chain is non-trivial to plumb into an existing pipeline.
Exit: `vkGetPhysicalDeviceFeatures2` reports `descriptorHeap` supported on
your own driver (check this before scoping further); CPU trace shows no
`vkUpdateDescriptorSets` in steady state (same exit bar the source
document already set for T3).

### T16 — Subgroup arithmetic for the eventual fused kernel (T6) — low priority

`GL_KHR_shader_subgroup_arithmetic` (`subgroupAdd` and friends) lets a
compute shader reduce values across a subgroup without shared-memory
writes or a `barrier()` call, where the current shared-memory pattern
(S5) requires both (S23). For a fused blur/reduction kernel where the tap
radius fits inside one subgroup, this removes synchronization stalls that
shared-memory tiling alone doesn't. This is additive to T6, not a
replacement for it, and applies only once T6 itself is attempted.

Given T9-lite's finding that GPU compute is ~5% of wall time right now,
this is explicitly polish, not a priority — listed for completeness only.
Risk: low to FP semantics if reduction order is preserved identically;
still needs the same bitwise-parity gate T6 already carries.
Exit: same as T6's existing exit criterion (bitwise parity vs current
shaders), plus a subgroup-size query since subgroup size is
implementation- and possibly dispatch-dependent (S23).

## 3. Checked, found nothing new

- **gpu-allocator crate**: current release is still 0.28.0 (same version
  the source document already cites as current, S10); no newer arena or
  linear-allocation scheme was found. Nothing to add here.
- **Dedicated transfer queue, concrete implementation status**: the source
  document's S6 already names the abstract Vulkan concept. What's new is
  only that RADV has a concrete SDMA-backed implementation with GFX10.3
  (RDNA2 — your architecture) explicitly listed as supported (S25) — but
  it is Linux/RADV-only, still gated behind `RADV_PERFTEST=transfer_queue`,
  and no equivalent was found for the Windows AMD driver. Not promoted to
  a track: your dev target is Windows, and "experimental flag on the
  platform you don't develop on" isn't actionable evidence. Noted here
  only so a future pass doesn't re-discover it as if it were new.

## 4. Evidence base (sources opened 2026-09-07)

- **S13** Fabian Giesen ("ryg"), *Write combining is not your friend*
  (2013). WC buffering behavior, per-architecture alignment/width rules,
  implicit-read pitfall in ordinary-looking code, `MOVNTDQA` as the
  correct way to read WC memory.
  https://fgiesen.wordpress.com/2013/01/29/write-combining-is-not-your-friend/
- **S14** Mechanical Sympathy blog, *Write Combining* (2011). Fixed,
  small number of per-core write-combining/line-fill buffers; writes to
  more distinct cache lines than there are buffers evict partial combines.
  https://mechanical-sympathy.blogspot.com/2011/07/write-combining.html
- **S15** NVIDIA Developer Forums, *Write-Combining memory can slow down
  your application?* (measured host-side bandwidth over WC vs pinned
  memory; chipset-dependent, not universally faster). Directional only.
  https://forums.developer.nvidia.com/t/write-combining-memory-can-slow-down-your-application/14323
- **S16** Khronos, `VK_EXT_host_image_copy` reference page. Copies host
  memory directly to/from images without a staging buffer; core-optional
  in Vulkan 1.4; requires `HOST_IMAGE_TRANSFER` format feature.
  https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_host_image_copy.html
- **S17** Khronos blog, *Copying Images on the Host in Vulkan* (2023).
  `vkCopyMemoryToImageEXT`; `VK_HOST_IMAGE_COPY_MEMCPY_EXT` fast path for
  pre-swizzled data, expected to match memcpy-class throughput.
  https://www.khronos.org/blog/copying-images-on-the-host-in-vulkan
- **S18** Phoronix, *RADV Driver Enables Host Image Copy By Default For
  RDNA2 & Newer* (April 2026). New AVX2 ADDRLIB swizzling path; ~20 GiB/s
  figure on comparable RDNA hardware (single-sourced, directional).
  https://www.phoronix.com/news/RADV-Default-Host-Image-Copy
- **S19** Khronos Vulkan registry, `VK_EXT_external_memory_host` man page.
  Import an existing host allocation as device memory via
  `vkGetMemoryHostPointerPropertiesEXT` / `VkImportMemoryHostPointerInfoEXT`;
  alignment and ownership rules.
  https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_external_memory_host.html
- **S20** Khronos blog, *Vulkan Introduces Roadmap 2026 and New Descriptor
  Heap Extension* (January 2026). `VK_EXT_descriptor_heap` overview,
  motivation (issues found in `VK_EXT_descriptor_buffer`), binding-interface
  compatibility mode.
  https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension
- **S21** AMD, Adrenalin Windows driver release notes
  (RN-RAD-WIN-25-30-17-02-EXPANDED-VLK-SUPPORT). Confirms
  `VK_EXT_descriptor_heap` support on the Windows AMD driver.
  https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-25-30-17-02-EXPANDED-VLK-SUPPORT.html
- **S22** Phoronix, *Mesa 26.1 RADV Driver Merges Vulkan Descriptor Heap*
  (April 2026) and *RADV Enables Vulkan Descriptor Heap Support By
  Default* (June 2026). RADV timeline: experimental flag → default-on.
  https://www.phoronix.com/news/RADV-Merges-Descriptor-Heap
  https://phoronix.com/news/RADV-Descriptor-Heap-Default
- **S23** Khronos, Vulkan Guide *Subgroups* chapter. Subgroup operations,
  `GL_KHR_shader_subgroup_arithmetic`, dynamic subgroup size caveat.
  https://docs.vulkan.org/guide/latest/subgroups.html
- **S24** Khronos community forum, compute-reduction thread (illustrative
  `subgroupAdd` usage pattern only — forum-tier source, not authoritative;
  paired with S23 for anything load-bearing).
  https://community.khronos.org/t/how-to-efficiently-perform-compute-reductions/106896
- **S25** Mesa GitLab, RADV dedicated transfer-queue merge request (SDMA
  hardware; GFX9/10/10.3/11 support matrix; gated behind
  `RADV_PERFTEST=transfer_queue`).
  https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/25594

## 5. Could-not-verify (this session)

- **No source code was opened for this pass.** Everything above was
  checked against the uploaded planning document only, not against your
  actual Rust/shader source. Every applicability judgment (buffer-vs-image
  for T14, what the prep loop's write pattern actually looks like for T12)
  is an inference from that document's prose, not a confirmed finding.
  Treat §2 as a prioritized list of things to *check against your own
  code*, not as pre-verified conclusions.
- Whether the Windows AMD (Adrenalin) driver exposes
  `VK_EXT_external_memory_host` or `VK_EXT_host_image_copy` specifically
  was not found in AMD's own release notes (only `descriptor_heap` was
  confirmed there, S21) — run `vulkaninfo` on your own machine before
  scoping T13 or T14.
- The exact number of write-combining/line-fill buffers per core (T12) is
  architecture- and generation-specific; S13/S14 are written from an
  Intel-centric vantage point and I do not know your CPU model (only that
  it's a Lenovo laptop pairing an RX 6600M) — the qualitative mechanism
  transfers, the specific buffer count does not.
- Whether your `prep` loop actually exhibits the interleaved-destination
  or implicit-read pattern T12 hypothesizes is unconfirmed — it is a
  candidate diagnosis to test with the synthetic microbenchmark in T12's
  exit criterion, not an established finding.

## 6. How these slot against the existing priority list

Cheapest-to-confirm first, since none of these carry shader/FP-parity risk
except T16 (which is explicitly gated behind T6 anyway): **T12** can be
confirmed or refuted with an isolated microbenchmark before touching real
code, and directly targets the profile T9-lite already measured — it
belongs ahead of T5/T7/T6 in the source document's current re-ranked list.
**T13** and **T15** are next: both are capability-gated (a single API query
each tells you if they're even available on your hardware) before any
real investment. **T14** carries the most rework risk and is gated behind
an inventory question this document couldn't answer, so it sits behind
T6 rather than competing with T12/T13. **T16** stays where T6 already is.

## 7. Bottom line

The original pass's evidence base (S1–S12) and its two research-delta
passes were thorough on submission batching, barriers, memory
sub-allocation, and shared-memory compute — but did not surface five
things that exist and apply here: CPU-side write-combining access-pattern
pitfalls distinct from memory-*type* selection (T12), two host-memory
extensions that remove the staging-buffer copy step itself rather than
just amortizing it (T13, T14), a newer descriptor mechanism that
supersedes T3's original plan (T15), and subgroup arithmetic for the
eventual fused kernel (T16, low priority). None of these were tried, and
none were named anywhere in the source document. T12 is the one worth
acting on first — it's a same-day, zero-parity-risk experiment that
targets the exact bottleneck the last measurement pass already found.
