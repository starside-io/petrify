# Changelog

All notable changes to petrify are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
petrify adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Single-pass streaming engine (`petrify/core/engine.py`). Per frame it runs the detectors lazily and in order (MOG2 always runs because it is stateful; the others stop once a subject is found), delays output by `--gap-fill` frames in a lookahead buffer to apply the gap fill, applies the cold-start freeze of the first 50 frames, renders the erase mode (locked background when a subject is present, live frame otherwise), pastes the live timestamp ROI, and writes to the file sink
- Package layout under `petrify/`: `core/` (frame, detections, config, plugin protocols, engine), `sources/file.py`, `detectors/` (`motion.py`, `bgdiff.py`, `person_hog.py`), `background/locked.py`, `modes/erase.py`, `stages/roi_overlay.py`, `sinks/file.py`
- Golden parity tests (`pytest -m golden`) that compare the old pipeline, the engine, and the CLI against fixtures committed in `tests/golden/fixtures/`. They need the demo clips `data/camera.mp4` and `data/camera2.mp4` (not in the repository) and are skipped without them. `python -m tests.golden.regenerate` records fixtures

### Changed
- `petrify INPUT OUTPUT` now runs through the streaming engine (`petrify.pipeline.run_file`) instead of the two-pass pipeline. Output for the same input and flags is unchanged, verified frame for frame against recorded fixtures. Background selection and timestamp ROI auto-detection still pre-scan the file before the processing pass
- The modules `petrify.motion`, `petrify.person`, `petrify.bgdiff`, and `petrify.compositor` moved to `petrify.detectors.motion`, `petrify.detectors.person_hog`, `petrify.detectors.bgdiff`, and `petrify.stages.roi_overlay`; the old module paths no longer exist. `petrify.Compositor`, `petrify.MotionDetector`, and `petrify.run` are still exported. `petrify.run` is still the old two-pass function, kept as the reference implementation for the golden tests; the CLI no longer calls it
- Log output: one progress bar and an `Engine done: frozen=... pass-through=...` summary replace the "Pass 1" and "Pass 2" blocks
- HOG downsampling never shrinks a frame below the 64x128 detection window and keeps the aspect ratio
- If a detector, the mode, or a stage fails on a frame, that frame is written as a black frame and counted (fail closed), a warning is logged, and the run continues

### Removed
- MOG2 pre-warmup: it never had an effect, because its model was reset before use
- `--adaptive-background` flag: it was parsed but never read

### Fixed
- Frames smaller than the HOG detection window (64x128) now produce a clear error instead of crashing the process
- A negative `--gap-fill` is rejected
- A failed run never leaves an encoded output from partial frames

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
