# Changelog

All notable changes to petrify are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
petrify adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-04-10

### Added

#### Core pipeline
- Two-pass architecture: **Pass 1** classifies every frame; **Pass 2** composites
- **MOG2** adaptive background subtraction for motion detection
- **HOG** person detector (`cv2.HOGDescriptor` with `getDefaultPeopleDetector()`) for stopped subjects
- **BackgroundDiffer** — static diff against the locked clean reference frame; catches subjects that MOG2 adapts away (standing still) and tiny/partial subjects HOG misses (far away, cut off at frame edge)
- Temporal **gap fill** (`--gap-fill`): expand every detected region ±N frames to bridge missed detections mid-sequence
- **Cold-start freeze**: unconditionally freeze the first 50 frames to cover the MOG2 warmup window
- **MOG2 pre-warmup**: feed the locked background frame ×history times before classification to establish a trained baseline at frame 0

#### Auto-detection
- **Auto ROI detection** (two-pass: frequency anchor + edge-based X extension): locates the burned-in timestamp rectangle automatically; prints a `--roi` hint so users can skip the scan on future runs
- **Auto background selection**: scans every 5th frame with HOG, locks the latest clean frame (no person) as the reference background

#### CLI
- `petrify INPUT OUTPUT` — zero-config entry point
- `--roi X Y W H` / `--no-roi` — manual or disabled timestamp overlay
- `--detector {motion,person,both}` — select active detectors
- `--threshold`, `--history`, `--gap-fill`, `--downsample` — motion tuning
- `--hog-min-confidence`, `--hog-min-box-area` — HOG tuning
- `--bg-diff-pixel-threshold`, `--bg-diff-area-threshold`, `--no-bg-diff` — background differ tuning
- `--background-frame N` — manual background frame override
- `--roi-scan-samples`, `--roi-scan-min-freq`, `--roi-scan-edge-threshold`, `--roi-scan-edge-gap-tolerance` — ROI scan tuning
- `--codec`, `--crf`, `--preset`, `--hwaccel` — FFmpeg encoding options
- `--verbose` — DEBUG-level log output
- ASCII art banner at startup with `Made with ❤️ by starside.io`

#### Infrastructure
- `pipx install petrify` single-command install; `petrify` available globally on PATH
- Homebrew FFmpeg fallback in encoder (`/opt/homebrew/bin/ffmpeg`)
- macOS quarantine strip (`xattr -d com.apple.quarantine`) applied automatically to output
- `pyproject.toml` with hatchling build backend, Apache 2.0 classifier

[Unreleased]: https://github.com/starside-io/petrify/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/starside-io/petrify/releases/tag/v0.1.0
