//! Test-only intermediate-dump instrumentation for the Vulkan port (plan §6,
//! Phase A / VULKAN_PORT_PLAN.md Phase 0). Feature-gated behind `dssim-dumps`;
//! adds no public API and does not change any production semantics.
//!
//! The dump format is deliberately boring: fixed-size little-endian metadata
//! header + contiguous little-endian f32 payload.
//!
//! ```text
//! magic   [u8; 4]   = b"DSSD"
//! version u32       = 1
//! kind    [u8; 16]  = NUL-padded kind tag (see [`DUMP_KINDS`])
//! scale   u32
//! channel u32       = 0 for L/gray, 1..=2 for a/b planes, 0xFF = n/a
//! width   u32
//! height  u32
//! stride  u32       = buffer stride in elements (>= width)
//! count   u32       = number of f32 payload elements
//! ```
//!
//! A directory of dumps is accompanied by a `MANIFEST.txt` listing every file
//! with a FNV-1a hash of its bytes, so two runs can be compared byte-for-byte
//! via their manifests.
//!
//! Cross-image-run ordering is not encoded in files; the manifest order plus
//! the generation log (`run.log`) carries it. The reproducibility check
//! compares file sets, sizes, hashes, and manifest bytes.

use imgref::ImgVec;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

pub const DUMP_MAGIC: [u8; 4] = *b"DSSD";
pub const DUMP_VERSION: u32 = 1;

/// Every kind tag the instrumentation emits. Kept in one place so the Vulkan
/// port's dump-compare tooling can enumerate them exhaustively.
pub const DUMP_KINDS: [&str; 10] = [
    "input_rgbaplu",
    "lab_plane",
    "scale_dims",
    "chan_img",
    "chan_mu",
    "chan_sq_blur",
    "cross_blur",
    "ssim_map",
    "score_ssim",
    "score_dssim",
];

pub const CHANNEL_NA: u32 = 0xFFFF_FFFF;

const HEADER_SIZE: usize = 4 + 4 + 16 + 4 * 6;

/// Where the current run's dumps go. Set by the generator before any
/// `create_image`/`compare` call; dumps are only written when set.
static SINK: std::sync::Mutex<Option<Sink>> = std::sync::Mutex::new(None);

struct Sink {
    dir: PathBuf,
    /// Monotonic counter over pipeline operations (one `create_image` or one
    /// `compare`), so identical kinds from different images/comparisons don't
    /// collide on file names.
    run_seq: u64,
    /// Dumps recorded before the final scale indexing is known
    /// (`make_scales_recursive` defers input/Lab dumps to the flush point).
    deferred: Vec<Deferred>,
}

struct Deferred {
    kind: &'static str,
    /// Recursion depth: 0 = original image (= post-reverse scale 0),
    /// +1 per downsample. See [`flush_deferred`].
    depth: u32,
    channel: u32,
    width: u32,
    height: u32,
    data: Vec<f32>,
}

/// Enable dumping into `dir` (created on demand). Returns a guard; when the
/// guard (or the whole process) ends, dumping stops. Dumping is disabled while
/// no guard is alive.
///
/// Dump mode is single-threaded at the orchestration level: one
/// `create_image`/`compare` at a time (the rayon parallelism *inside* one
/// operation is fine — the sink is only touched by the operation's own tasks).
pub fn enable(dir: impl AsRef<Path>) -> io::Result<DumpGuard> {
    let dir = dir.as_ref().to_path_buf();
    fs::create_dir_all(&dir)?;
    let mut sink = SINK.lock().expect("dumps sink mutex poisoned");
    *sink = Some(Sink { dir, run_seq: 0, deferred: Vec::new() });
    Ok(DumpGuard)
}

/// RAII guard disabling dumps when dropped.
#[derive(Debug)]
pub struct DumpGuard;

impl Drop for DumpGuard {
    fn drop(&mut self) {
        *SINK.lock().expect("dumps sink mutex poisoned") = None;
    }
}

/// Start a new pipeline operation (`create_image` or `compare`): bumps the
/// run counter used in dump file names and resets deferred dumps.
#[doc(hidden)]
pub fn next_run() {
    if let Some(sink) = SINK.lock().expect("dumps sink mutex poisoned").as_mut() {
        sink.run_seq += 1;
        sink.deferred.clear();
    }
}

/// Queue a dump whose final scale index is not known yet (recursion runs
/// depth-first). Flushed by [`flush_deferred`].
fn defer(kind: &'static str, depth: usize, channel: u32, width: usize, height: usize, data: Vec<f32>) {
    if let Some(sink) = SINK.lock().expect("dumps sink mutex poisoned").as_mut() {
        debug_assert_eq!(data.len(), width * height);
        sink.deferred.push(Deferred {
            kind,
            depth: depth as u32,
            channel,
            width: width as u32,
            height: height as u32,
            data,
        });
    }
}

/// Write all deferred dumps, translating recursion depth into the
/// post-reverse scale index. Scale 0 is the ORIGINAL image
/// (`DssimImage::width()` returns `scale[0]`'s width — the recursion arm of
/// `rayon::join` finishes its pushes before the parent's, so pre-reverse
/// order is deepest-first and reversing puts depth 0 first). Called by
/// `create_image` once the actual number of generated scales is known.
#[doc(hidden)]
pub fn flush_deferred(num_scales: usize) {
    let mut sink_guard = SINK.lock().expect("dumps sink mutex poisoned");
    let Some(sink) = sink_guard.as_mut() else { return };
    let mut deferred = std::mem::take(&mut sink.deferred);
    drop(sink_guard);
    deferred.sort_by_key(|d| (d.depth, d.channel));
    for d in deferred {
        debug_assert!(d.depth < num_scales as u32, "deferred depth {} out of range ({num_scales} scales)", d.depth);
        write_dump(d.kind, d.depth, d.channel, d.width, d.height, d.width, &d.data);
    }
}

fn write_dump(kind: &str, scale: u32, channel: u32, width: u32, height: u32, stride: u32, data: &[f32]) {
    let sink_guard = SINK.lock().expect("dumps sink mutex poisoned");
    let Some(sink) = sink_guard.as_ref() else { return };

    let count = data.len() as u32;
    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(&DUMP_MAGIC);
    header.extend_from_slice(&DUMP_VERSION.to_le_bytes());
    let mut kind_bytes = [0u8; 16];
    let kind_b = kind.as_bytes();
    kind_bytes[..kind_b.len()].copy_from_slice(kind_b);
    header.extend_from_slice(&kind_bytes);
    header.extend_from_slice(&scale.to_le_bytes());
    header.extend_from_slice(&channel.to_le_bytes());
    header.extend_from_slice(&width.to_le_bytes());
    header.extend_from_slice(&height.to_le_bytes());
    header.extend_from_slice(&stride.to_le_bytes());
    header.extend_from_slice(&count.to_le_bytes());

    let mut payload = Vec::with_capacity(header.len() + data.len() * 4);
    payload.extend_from_slice(&header);
    for v in data {
        payload.extend_from_slice(&v.to_le_bytes());
    }

    let file_name = format!("{kind}.s{scale}.c{:#06x}.run{}.bin", channel, sink.run_seq);    let path = sink.dir.join(&file_name);
    // Atomic-ish write: write to a temp name then rename, so an interrupted
    // run can't leave a truncated golden that later looks valid.
    let tmp_path = sink.dir.join(format!("{file_name}.tmp"));
    if fs::write(&tmp_path, &payload).is_ok() {
        let _ = fs::rename(&tmp_path, &path);
    }

    // Manifest + log lines mirror what was written. We rewrite the whole
    // manifest each time (files are few and small); ordering stays stable
    // because the call sequence is deterministic.
    append_manifest(sink, &file_name, &payload);
    append_log(sink, &format!("write {file_name} scale={scale} channel={channel:#x} {width}x{height} stride={stride} count={count}"));
}

fn append_manifest(sink: &Sink, file_name: &str, payload: &[u8]) {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    payload.hash(&mut hasher);
    let hash = hasher.finish();
    let _ = append_line(&sink.dir.join("MANIFEST.txt"), &format!("{hash:016x}  {file_name}"));
}

fn append_log(sink: &Sink, line: &str) {
    let _ = append_line(&sink.dir.join("run.log"), line);
}

fn append_line(path: &Path, line: &str) -> io::Result<()> {
    use std::io::Write;
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")
}

fn sink_active() -> bool {
    SINK.lock().expect("dumps sink mutex poisoned").is_some()
}

/// Dump a planar f32 image (Lab planes, SSIM maps, per-channel planes).
#[doc(hidden)]
pub fn dump_plane(kind: &str, scale: u32, channel: u32, img: &ImgVec<f32>) {
    if !sink_active() {
        return;
    }
    write_dump(
        kind,
        scale,
        channel,
        img.width() as u32,
        img.height() as u32,
        img.stride() as u32,
        img.buf(),
    );
}

/// Dump a contiguous f32 buffer (mu, img_sq_blur, img1_img2_blur vectors).
#[doc(hidden)]
pub fn dump_buf(kind: &str, scale: u32, channel: u32, width: usize, height: usize, data: &[f32]) {
    if !sink_active() {
        return;
    }
    write_dump(kind, scale, channel, width as u32, height as u32, width as u32, data);
}

/// Queue a planar f32 dump whose scale index is decided at flush time.
#[doc(hidden)]
pub fn defer_bitmap(kind: &'static str, depth: usize, channel: u32, width: usize, height: usize, data: &[f32]) {
    if !sink_active() {
        return;
    }
    defer(kind, depth, channel, width, height, data.to_vec());
}

/// Record Lab planes (deferred; one payload per plane).
#[doc(hidden)]
pub fn defer_lab(depth: usize, lab: &[ImgVec<f32>]) {
    if !sink_active() {
        return;
    }
    for (c, plane) in lab.iter().enumerate() {
        let data: Vec<f32> = plane.pixels().collect();
        defer("lab_plane", depth, c as u32, plane.width(), plane.height(), data);
    }
}

/// Dump scalar values (scores, run metadata) as one-element f32 payloads.
#[doc(hidden)]
pub fn dump_scalar(kind: &'static str, scale: u32, channel: u32, value: f32) {
    if !sink_active() {
        return;
    }
    write_dump(kind, scale, channel, 1, 1, 1, &[value]);
}

/// Record the scale dimensions for one scale as a 2-element payload
/// (width, height), using kind `scale_dims`.
#[doc(hidden)]
pub fn dump_dims(scale: u32, width: usize, height: usize) {
    if !sink_active() {
        return;
    }
    write_dump("scale_dims", scale, CHANNEL_NA, 2, 1, 2, &[width as f32, height as f32]);
}

/// Note a run boundary in the log (which fixture is being materialized).
#[doc(hidden)]
pub fn log_run(label: &str) {
    if let Some(sink) = SINK.lock().expect("dumps sink mutex poisoned").as_ref() {
        append_log(sink, &format!("run {label}"));
    }
}

// ---------------------------------------------------------------------------
// Reader + comparison utility
// ---------------------------------------------------------------------------

/// Parsed header of a dump file.
#[derive(Debug, Clone, PartialEq)]
pub struct DumpHeader {
    pub kind: String,
    pub scale: u32,
    pub channel: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub count: u32,
}

/// A fully parsed dump file: header plus payload.
#[derive(Debug, Clone)]
pub struct Dump {
    pub header: DumpHeader,
    pub data: Vec<f32>,
}

impl Dump {
    /// Parse a dump file (header validation included).
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a dump from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "dump truncated (header)"));
        }
        if bytes[0..4] != DUMP_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != DUMP_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported dump version {version}")));
        }
        let kind_raw = &bytes[8..24];
        let kind = String::from_utf8_lossy(kind_raw)
            .trim_end_matches('\0')
            .to_owned();
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let header = DumpHeader {
            kind,
            scale: u32_at(24),
            channel: u32_at(28),
            width: u32_at(32),
            height: u32_at(36),
            stride: u32_at(40),
            count: u32_at(44),
        };
        let expected = header.count as usize * 4;
        if bytes.len() < HEADER_SIZE + expected {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "dump truncated (payload)"));
        }
        let payload = &bytes[HEADER_SIZE..HEADER_SIZE + expected];
        let mut data = Vec::with_capacity(header.count as usize);
        let mut rest = payload;
        while rest.len() >= 4 {
            data.push(f32::from_le_bytes(rest[..4].try_into().unwrap()));
            rest = &rest[4..];
        }
        Ok(Dump { header, data })
    }
}

/// What two dumps differ by.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffStats {
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub rmse: f64,
    /// Index (payload element) and coordinates of the worst pixel.
    pub worst_index: usize,
    pub worst_x: u32,
    pub worst_y: u32,
    pub cpu_value: f32,
    pub gpu_value: f32,
}

impl DiffStats {
    /// Compare two dumps of the same shape; `a` is the CPU (reference) side,
    /// `b` the GPU side. Returns `Err` with a reason if shapes differ.
    pub fn compare(a: &Dump, b: &Dump) -> Result<DiffStats, String> {
        let ha = &a.header;
        let hb = &b.header;
        for (label, va, vb) in [
            ("kind", ha.kind.as_str(), hb.kind.as_str()),
            ("scale", &ha.scale.to_string(), &hb.scale.to_string()),
            ("channel", &format!("{:#x}", ha.channel), &format!("{:#x}", hb.channel)),
            ("width", &ha.width.to_string(), &hb.width.to_string()),
            ("height", &ha.height.to_string(), &hb.height.to_string()),
            ("count", &ha.count.to_string(), &hb.count.to_string()),
        ] {
            if va != vb {
                return Err(format!("shape mismatch on {label}: {va} vs {vb}"));
            }
        }
        let n = a.data.len();
        if n == 0 {
            return Ok(DiffStats {
                max_abs_error: 0.0,
                mean_abs_error: 0.0,
                rmse: 0.0,
                worst_index: 0,
                worst_x: 0,
                worst_y: 0,
                cpu_value: 0.0,
                gpu_value: 0.0,
            });
        }
        let width = ha.width.max(1) as usize;
        let mut max_abs = f64::MIN;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut worst = 0usize;
        for i in 0..n {
            let d = (f64::from(a.data[i]) - f64::from(b.data[i])).abs();
            sum_abs += d;
            sum_sq += d * d;
            if d > max_abs {
                max_abs = d;
                worst = i;
            }
        }
        Ok(DiffStats {
            max_abs_error: max_abs,
            mean_abs_error: sum_abs / n as f64,
            rmse: (sum_sq / n as f64).sqrt(),
            worst_index: worst,
            worst_x: (worst % width) as u32,
            worst_y: (worst / width) as u32,
            cpu_value: a.data[worst],
            gpu_value: b.data[worst],
        })
    }

    /// One-line report in the format Phase A prescribes.
    pub fn report(&self, name: &str) -> String {
        format!(
            "{name}: max_abs={:.3e} mean_abs={:.3e} rmse={:.3e} worst @ ({}, {}) cpu={} gpu={}",
            self.max_abs_error,
            self.mean_abs_error,
            self.rmse,
            self.worst_x,
            self.worst_y,
            self.cpu_value,
            self.gpu_value
        )
    }

    /// Read two dump files and compare them; `Ok(Err(..))` on shape mismatch.
    pub fn compare_files(
        a: impl AsRef<Path>,
        b: impl AsRef<Path>,
    ) -> io::Result<Result<DiffStats, String>> {
        let da = Dump::read(a)?;
        let db = Dump::read(b)?;
        Ok(DiffStats::compare(&da, &db))
    }
}
