# RGBA Structural Similarity

This tool computes (dis)similarity between two or more PNG &/or JPEG images using an algorithm approximating human vision. Comparison is done using a variant of [the SSIM algorithm](https://ece.uwaterloo.ca/~z70wang/research/ssim/).

The value returned is 1/SSIM-1, where 0 means identical image, and >0 (unbounded) is amount of difference. Values are not directly comparable with other tools. [See below](#interpreting-the-values) on interpreting the values.

## About this repository

This is **dssim-vulcan** — a fork of [dssim](https://github.com/kornelski/dssim)
that ports the DSSIM computation to a **Vulkan compute backend** (the `--gpu`
flag) without changing the CPU path. Everything above (the algorithm, the values,
plain CPU usage) still applies; the sections below add the GPU backend and this
repo's build/test workflow.

Workspace crates:

- `dssim-core` — the CPU algorithm (AGPL; vendored fork of upstream dssim).
- `dssim-vulkan` — the Vulkan compute backend (`ash` + `gpu-allocator`), the
  port's kernels and orchestration.
- `dssim` — the CLI + library that ties them together and exposes `--gpu`.

## Features

* Improved algorithm
    * Compares at multiple weighed resolutions, and scaling is done in linear-light RGB. It's sensitive to distortions of various sizes and blends colors correctly to detect e.g. chroma subsampling errors.
    * Uses L\*a\*b\* color space for the SSIM algorithm. It measures brightness and color much better than metrics from average of RGB channels.
* Supports alpha channel.
* Supports images with color profiles.
* Takes advantage of multi-core CPUs.
* Can be used as a library in C, Rust, and WASM.
* No OpenCV or MATLAB needed.

## Usage

    dssim file-original.png file-modified.png

Will output something like "0.02341" (smaller is better) followed by a filename.

You can supply multiple filenames to compare them all with the first file:

    dssim file.png modified1.png modified2.png modified3.png

You can save an image visualising the difference between the files:

    dssim -o difference.png file.png file-modified.png

There is an experimental Vulkan compute backend. It gives the same scores
within a small floating-point tolerance (≤0.000005), but requires a Vulkan
driver and ignores `-o` for now. It falls back to the CPU automatically when
Vulkan is unavailable. Color profiles are applied identically to the CPU path
(same decoder). 16-bit inputs are handled at full precision — the u16→linear
conversion uses the same 65536-entry LUT as the CPU path (BH9 corrected an
earlier note that wrongly called this an 8-bit approximation):

    dssim --gpu file.png file-modified.png

The GPU backend also has an opt-in device-side pyramid builder:

    dssim --gpu --gpu-prep=device file.png file-modified.png

`--gpu-prep=device` uploads only the full-resolution image and builds the rest of
the multi-scale pyramid on the GPU (a ~1.6–3.6× faster `create` at large sizes on
a discrete GPU). It is **bitwise-equal** to the default `--gpu-prep=cpu` (same
scores), so it's a drop-in; it's opt-in because GPU-side downsampling is otherwise
a documented non-goal. `--gpu-prep` has no effect without `--gpu`.

It's also usable [as a library](https://docs.rs/dssim).

Please be mindful about color profiles in the images. Different profiles, or lack of support for profiles in other tools, can make images appear different even when the pixels are the same.

### Interpreting the values

The amount of difference goes from 0 to infinity. It's not a percentage.

If you're comparing two different image compression codecs, then ensure you either:

* compress images to the same file size, and then use DSSIM to compare which one is closests to the original, or
* compress images to the same DSSIM value, and compare file sizes to see how much file size gain each option gives.

[More about benchmarking image compression](https://kornel.ski/faircomparison).

When you quote results, please include the DSSIM version. The scale has changed between versions.
The version is printed when you run `dssim -h`.

## Download

[Download from releases page](https://github.com/kornelski/dssim/releases). It's also available in Mac Homebrew and Ubuntu Snaps.

### Build from source

You'll need [Rust 1.90](https://rustup.rs) or later (this repo uses edition 2024).
The Vulkan backend is on by default (the `gpu` feature); for a CPU-only build use
`--no-default-features --features threads`. Clone the repo and run:

    rustup update
    cargo build --release

Will give you `./target/release/dssim`. The compiled shaders (`.spv`) are checked
in, so you do **not** need a Vulkan SDK to build. You only need `glslc` (from the
Vulkan SDK) to recompile a `.comp` after editing it:

    glslc --target-env=vulkan1.3 -O dssim-vulkan/shaders/foo.comp -o dssim-vulkan/shaders/foo.comp.spv

## Development

Run the test suite. The GPU tests need a Vulkan device; run them serially (driver
contention) and in **debug** so the validation layer is active:

    cargo test --workspace -- --test-threads=1

Debug builds enable the Khronos Vulkan validation layer when it's installed and
fail loudly on any `[vulkan ERROR]`; set `DSSIM_VK_NO_VALIDATION=1` to disable it.
`cargo test --release` exercises the shipping code paths (validation and
`debug_assert!`s are off there, but the parity / locked-value checks still run).

Bench CPU vs GPU (release — the numbers are meaningless in debug):

    cargo run -p dssim-vulkan --example bench --release

Environment variables:

- `DSSIM_VK_NO_VALIDATION=1` — disable the validation layer (debug builds).
- `DSSIM_UNIFIED=0|1` — force the staging (`0`) or zero-copy (`1`) upload path
  for A/B testing; unset uses device detection.
- `DSSIM_BENCH_DEVICE=<n>` — pin the bench to candidate GPU `<n>` (e.g. the
  integrated one); the default picks the discrete GPU.

Project docs: `CHECKPOINT.md` (current state), `VULKAN_PERF.md` (measured perf),
`VULKAN_PORT_PLAN.md` / `dssim-vulkan-fable-plan.md` (the port plan), and the
`AUDIT_*.md` files (review passes). `AGENTS.md` governs how the repo is worked on.

## Accuracy

Scores for version 3.2 [measured][2] against [TID2013][1] database:

TID2013  | Spearman | Kendall
---------|----------|--------
Noise    |  -0.9392 | -0.7789
Actual   |  -0.9448 | -0.7913
Simple   |  -0.9499 | -0.8082
Exotic   |  -0.8436 | -0.6574
New      |  -0.8717 | -0.6963
Color    |  -0.8789 | -0.7032
Full     |  -0.8711 | -0.6984

[1]: http://www.ponomarenko.info/tid2013.htm
[2]: https://lib.rs/crates/tid2013stats

## Usage from C

Make sure to build `dssim-core` library project, not the parent `dssim` binary project.

```bash
cd dssim-core
rustup update
cargo build --release
```

This will build `target/release/libdssim_core.a` that you can link with your project. Use `dssim.h` included in the dssim repo. It's up to you where you put these files.

Alternatively, on Linux there is a more involved but slightly more proper method:

```bash
cargo install cargo-c
cargo cinstall --release --destdir=/ --prefix=/usr/lib
```

This will install `libdssim.so` in `/usr/lib` and make `dssim` available to `pkg-config`. See `target/<platform>/release` for all the files built this way.

## License

DSSIM is dual-licensed under [AGPL](LICENSE) or [commercial](https://supso.org/projects/dssim) license.

## The algorithm improvements in DSSIM

* The comparison is done on multiple weighed scales (based on IWSSIM) to measure features of different sizes. A single-scale SSIM is biased towards differences smaller than its gaussian kernel.
* Scaling is done in linear-light RGB to model physical effects of viewing distance/lenses. Scaling in sRGB or Lab would have incorrect gamma and mask distortions caused by chroma subsampling.
* a/b channels of Lab are compared with lower spatial precision to simulate eyes' higher sensitivity to brightness than color changes.
* SSIM score is pooled using mean absolute deviation. You can get per-pixel SSIM from the API to implement custom pooling.

## Compiling for WASM

For compatibility with single-threaded WASM runtimes, disable the `threads` Cargo feature. It's enabled by default, so to disable it, disable default features:

```toml
dssim-core = { version = "3.2", default-features = false }
```
