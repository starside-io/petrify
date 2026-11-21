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

#### `petrify run`
- New command form `petrify run --source SPEC` next to the unchanged `petrify INPUT OUTPUT`. It processes any source in one streaming pass and writes to one or more outputs. Options may precede `run` (`petrify --verbose run --source ...`), and a file named `run` still works in the file form (`petrify run out.mp4`)
- `--lookahead` (the run form's `--gap-fill`, at most 600), `--frame-budget-ms`, `--realtime`, `--mode erase`, and `--fps` for image folders. Defaults that depend on the source kind: background `locked` for a video file or image folder and `running` for a live source; lookahead 10 for a file, image folder, or stream URL and 3 for `camera:N`; timestamp ROI auto-detected except on `camera:N`
- Background options `--background {locked,running,calibrate}`, `--calibrate-seconds` (default 3), and `--no-background-check`
- ROI options `--roi-auto`, `--roi-warmup` (default 10 seconds), and `--roi-max-area` (default 0.15)
- Exit codes: 0 when the run completes (frames that failed closed and a live run ended with Ctrl-C included), 1 for a fatal error (a missing FFmpeg included), 2 for a missing input or an invalid option, 130 for a Ctrl-C that aborts a file or image folder run

#### Sources
- Image folder or glob source (`petrify.sources.ImageFolderSource`), in natural name order, hidden files ignored
- Webcam source `camera:N` (`CameraSource`) and network stream source for `rtsp://`, `rtsps://`, `http://`, and `https://` URLs (`StreamSource`). A capture thread keeps only the newest frame and counts the dropped ones
- Push API for library use (`petrify.sources.PushSession`): feed in-memory frames one by one and get the processed frames back after the lookahead delay
- `petrify.sources.open_source` and `source_kind` pick a source from a command-line spec

#### Outputs
- `--output`: FFmpeg encode in one pass through a pipe (`FfmpegPipeSink`), with no intermediate file. FFmpeg writes to `<name>.partial<suffix>`, which is moved into place only when the encode succeeds. The audio of a video file source is copied. Live sources keep real frame timestamps
- `--virtualcam` and `--virtualcam-device` (`VirtualCamSink`, optional extra `petrify[virtualcam]`; on macOS it also needs OBS 30 or newer). A black frame is sent before the camera closes
- `--preview` (`PreviewSink`): a local OpenCV window that `q` or Escape closes; needs an OpenCV build with GUI support
- `ThreadedSink`, which forwards frames to a wrapped sink on a worker thread

#### Background
- `running` (rolling median of recent clean frames) and `calibrate` (median of the first seconds) background models, with a person check (HOG) on the frames they learn from (`petrify.background.person_free`). A warning about the remaining limitation is logged at startup
- Frames are written black until an online background is ready

#### ROI
- Warmup ROI detection for sources that cannot be scanned in advance (live sources and image folders): detection on the first `--roi-warmup` seconds, then locked, with nothing pasted before it locks and a box over `--roi-max-area` of the frame discarded. A manual `--roi` over that cap is accepted with a warning

#### Engine
- Step API (`open`, `step`, `flush`, `close`, `abort`); `run()` is built from it
- `Engine.reconfigure` swaps detectors, mode, stages, or config at a frame boundary. The buffered frames are written with the old components first, and the frames around the switch are frozen. `Engine.timings` reports rolling detect, render, and write times and the output frame rate
- Sink isolation: sinks receive only the output image and frame metadata, never the input frame. A failing non-critical sink is dropped and counted while the run continues; a failing critical sink aborts the run
- Frame time budget (`frame_budget_ms`): a frame over budget fails closed to black and is counted
- The engine refuses to start when its mode requires detections and no detector provides any, and rejects re-entrant calls

#### Web page
- `--web`, `--web-port` (default 8765), and `--web-raw-preview` (optional extra `petrify[web]`): a local page that shows the processed feed, the run status and timings, and a stop button, and changes the detector, detector thresholds, background differ, lookahead, frame budget, and ROI while the run is in progress. Each change is applied at the next frame boundary and briefly freezes the output
- A "Weakened protections" list on the page and in the log: the background differ off, the person detector alone with the differ off, a threshold more than twice its startup value, a lookahead below its startup value or 0, a manual ROI over 5 percent of the frame, and the raw preview
- `petrify.server.control.RunControl`, the thread-safe link between the processing loop and the server

### Changed
- `petrify INPUT OUTPUT` now runs through the streaming engine (`petrify.pipeline.run_file`) instead of the two-pass pipeline. Output for the same input and flags is unchanged, verified frame for frame against recorded fixtures. Background selection and timestamp ROI auto-detection still pre-scan the file before the processing pass
- The modules `petrify.motion`, `petrify.person`, `petrify.bgdiff`, and `petrify.compositor` moved to `petrify.detectors.motion`, `petrify.detectors.person_hog`, `petrify.detectors.bgdiff`, and `petrify.stages.roi_overlay`; the old module paths no longer exist. `petrify.Compositor`, `petrify.MotionDetector`, and `petrify.run` are still exported. `petrify.run` is still the old two-pass function, kept as the reference implementation for the golden tests; the CLI no longer calls it
- Log output: one progress bar and an `Engine done: frozen=... pass-through=...` summary replace the "Pass 1" and "Pass 2" blocks
- HOG downsampling never shrinks a frame below the 64x128 detection window and keeps the aspect ratio
- If a detector, the mode, or a stage fails on a frame, that frame is written as a black frame and counted (fail closed), a warning is logged, and the run continues
- Optional extras `web` (FastAPI, uvicorn) and `virtualcam` (pyvirtualcam); `httpx` added to the `dev` extra for the web tests
- In `petrify run`, `--hwaccel` is passed to the FFmpeg process that encodes `--output` but does not speed anything up, since petrify decodes the source itself; the help says so
- A live run ended with Ctrl-C finalizes its outputs; a second Ctrl-C while they are finalized aborts. A live run that fails keeps the recording written up to that point and logs the error. FFmpeg runs in its own session, so a Ctrl-C reaches petrify only

### Removed
- MOG2 pre-warmup: it never had an effect, because its model was reset before use
- `--adaptive-background` flag: it was parsed but never read

### Fixed
- Frames smaller than the HOG detection window (64x128) now produce a clear error instead of crashing the process
- A negative `--gap-fill` is rejected
- A failed run never leaves an encoded output from partial frames

### Security
- Stream URLs are redacted in logs and error messages: at most the scheme, host, and port are shown, credentials become `***`, the path and query are dropped, and an ambiguous URL hides the host. A mistyped stream spec is not echoed with its credentials
- The web page listens on `127.0.0.1` only and accepts only `127.0.0.1:PORT` as `Host` and `Origin` (DNS rebinding); `localhost` is refused because the server does not listen on `[::1]`, where another local user could. Every route except the page and its assets needs the per-run token printed in the link (`http://127.0.0.1:PORT/#token=...`); the token is never logged and the access log is off. No redirects, no CORS headers, a strict Content-Security-Policy, and bounded connections
- The unprocessed feed is served only with `--web-raw-preview`, and is then listed as a weakened protection
- Values set from the web page are bounded so every detector can still fire, and a manual ROI set there may not exceed the `--roi-max-area` cap, since its region is shown live
- `--output` equal to the input file, or to one of the images of an image folder source, is rejected

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
