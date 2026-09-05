//! M0 / Phase A ground-truth harness (VULKAN_PORT_PLAN.md §4 Phase 0,
//! dssim-vulkan-fable-plan.md §6 Phase A).
//!
//! * `dump_goldens` — writes CPU intermediate dumps for every golden fixture
//!   scenario into `$DSSIM_DUMP_DIR` (skipped when the variable is unset).
//! * `dump_reproducible` — generates the same dumps twice and asserts they
//!   are byte-for-byte identical (the M0 exit observation), re-running the
//!   check through the `Dump`/`DiffStats` comparison utility as well.
#![cfg(feature = "dssim-dumps")]

use dssim_core::dumps::{self, DiffStats, Dump};
use dssim_core::{new, Downsample, Dssim, RGBAPLU, ToLABBitmap, ToRGBAPLU};
use imgref::{Img, ImgVec};
use std::fs;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

/// The dump sink is a process-global, so dump-generating tests must run one
/// at a time even though cargo runs them in parallel threads.
static GEN_LOCK: Mutex<()> = Mutex::new(());

fn gen_lock() -> MutexGuard<'static, ()> {
    GEN_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn decode_rgba(path: &str) -> ImgVec<RGBAPLU> {
    let file = lodepng::decode32_file(path).unwrap();
    Img::new(file.buffer.to_rgbaplu(), file.width, file.height)
}

/// Deterministic synthetic grayscale pair (xorshift + gradient) exercising the
/// 1-channel `GBitmap` path: `GBitmap::to_lab` (×1.16 gray variant), 1-channel
/// preprocess, and `compare_scale`.
fn synth_gray(width: usize, height: usize, seed: u64) -> ImgVec<f32> {
    let mut s = seed;
    let mut buf = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = (s & 0xFFFF) as f32 / 65535.0;
            let gradient = ((x + y) % 64) as f32 / 64.0;
            buf.push(0.5 * noise + 0.5 * gradient);
        }
    }
    Img::new(buf, width, height)
}

fn compare_pair<In, Out>(d: &Dssim, name: &str, reference: &In, modified: &In)
where
    In: ToLABBitmap + Send + Sync + Downsample<Output = Out>,
    Out: ToLABBitmap + Send + Sync + Downsample<Output = Out>,
{
    dumps::log_run(&format!("{name}_ref"));
    let ref_image = d.create_image(reference).unwrap();
    dumps::log_run(&format!("{name}_mod"));
    let mod_image = d.create_image(modified).unwrap();
    dumps::log_run(&format!("{name}_cmp"));
    let (score, _) = d.compare(&ref_image, mod_image);
    eprintln!("{name}: dssim = {score:.10}");
}

/// Generate the full dump set for all golden scenarios into `dir`.
pub fn generate_dumps(dir: &Path) {
    let _guard = dumps::enable(dir).unwrap();
    let d = new();

    let img1 = decode_rgba("../tests/test1-sm.png");
    let img2 = decode_rgba("../tests/test2-sm.png");

    // 1. Full-image pair — all 5 scales, 3 channels (the headline fixture).
    compare_pair(&d, "full", &img1, &img2);

    // 2. Sub-image scenarios from `png_compare` — strided ImgRef inputs.
    compare_pair(&d, "sub44x33", &img1.sub_image(2, 3, 44, 33), &img2.sub_image(17, 9, 44, 33));
    compare_pair(&d, "sub61x40", &img1.sub_image(22, 8, 61, 40), &img2.sub_image(22, 8, 61, 40));

    // 3. Grayscale pair — 1-channel path.
    compare_pair(&d, "gray", &synth_gray(96, 80, 0x9E37_79B9_7F4A_7C15), &synth_gray(96, 80, 0x632B_E5AB_2A2D_4D8F));

    // 4. Transparent-alpha pair — exercises the premultiply + alpha dither.
    compare_pair(&d, "alpha", &decode_rgba("../tests/alpha1.png"), &decode_rgba("../tests/alpha2.png"));
}

#[test]
fn dump_goldens() {
    let Some(dir) = std::env::var_os("DSSIM_DUMP_DIR") else {
        eprintln!("DSSIM_DUMP_DIR not set; skipping golden dump generation");
        return;
    };
    let _serial = gen_lock();
    generate_dumps(Path::new(&dir));
    eprintln!("dumps written to {}", Path::new(&dir).display());
}

#[test]
fn dump_reproducible() {
    let _serial = gen_lock();
    let base = std::env::temp_dir().join(format!("dssim-dumps-repro-{}", std::process::id()));
    let run1 = base.join("run1");
    let run2 = base.join("run2");
    generate_dumps(&run1);
    generate_dumps(&run2);

    let files1 = dump_files(&run1);
    let files2 = dump_files(&run2);
    assert_eq!(files1, files2, "dump file sets differ between runs");
    assert!(files1.len() > 100, "expected a substantial dump set, got {}", files1.len());

    for name in &files1 {
        let a = fs::read(run1.join(name)).unwrap();
        let b = fs::read(run2.join(name)).unwrap();
        assert_eq!(a.len(), b.len(), "{name}: size differs between runs");
        if a != b {
            let da = Dump::read(run1.join(name)).unwrap();
            let db = Dump::read(run2.join(name)).unwrap();
            let stats = DiffStats::compare(&da, &db).unwrap();
            panic!("{name}: not byte-identical between runs: {}", stats.report(name));
        }
    }

    // MANIFEST.txt and run.log lines are appended concurrently by parallel
    // compare scales, so their line *order* may vary; content must not.
    for meta in ["MANIFEST.txt", "run.log"] {
        let mut a = fs::read_to_string(run1.join(meta)).unwrap().lines().map(str::to_owned).collect::<Vec<_>>();
        let mut b = fs::read_to_string(run2.join(meta)).unwrap().lines().map(str::to_owned).collect::<Vec<_>>();
        a.sort();
        b.sort();
        assert_eq!(a, b, "{meta}: content differs between runs beyond line order");
    }

    fs::remove_dir_all(&base).ok();
}

#[test]
fn dump_reader_roundtrip() {
    let _serial = gen_lock();
    let dir = std::env::temp_dir().join(format!("dssim-dumps-roundtrip-{}", std::process::id()));
    generate_dumps(&dir);

    let files = dump_files(&dir);
    let mut ssim_maps = 0;
    let mut scale_dims = 0;
    for name in &files {
        let dump = Dump::read(dir.join(name)).unwrap();
        assert_eq!(dump.data.len(), dump.header.count as usize, "{name}: payload size mismatch");
        match dump.header.kind.as_str() {
            "ssim_map" => {
                assert!(dump.data.iter().all(|v| v.is_finite()), "{name}: non-finite SSIM values");
                ssim_maps += 1;
            }
            "scale_dims" => {
                let (w, h) = (dump.data[0], dump.data[1]);
                assert!(w >= 1.0 && h >= 1.0 && w == w.trunc() && h == h.trunc(), "{name}: bad dims {w}x{h}");
                scale_dims += 1;
            }
            _ => {}
        }
    }
    assert!(ssim_maps >= 5, "expected SSIM maps for 5 scenarios, got {ssim_maps}");
    assert!(scale_dims >= 18, "expected scale dims across all scenarios, got {scale_dims}");

    // Comparison utility self-check: identical files must report exact zeros.
    let name = &files[0];
    let stats = dssim_core::dumps::DiffStats::compare_files(dir.join(name), dir.join(name)).unwrap().unwrap();
    assert_eq!(stats.max_abs_error, 0.0, "{}: {}", name, stats.report(name));
    assert_eq!(stats.cpu_value, stats.gpu_value);

    fs::remove_dir_all(&dir).ok();
}

fn dump_files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".bin"))
        .collect();
    v.sort();
    v
}
