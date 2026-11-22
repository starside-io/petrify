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
- Per-mode detection policy (`Mode.detect_policy`): `"first_hit"` (erase) stops the stateless detectors at the first hit, `"all"` runs every active detector on every frame. `Mode.mandatory` names detection kinds that must have an active detector or refiner, or the engine refuses to start or to reconfigure
- Refiners (`petrify.core.plugin.Refiner`) run after the detectors and add detections; they require the `"all"` policy and do not run on a frame whose detection failed

#### Web page
- `--web`, `--web-port` (default 8765), and `--web-raw-preview` (optional extra `petrify[web]`): a local page that shows the processed feed, the run status and timings, and a stop button, and changes the detector, detector thresholds, background differ, lookahead, frame budget, and ROI while the run is in progress. Each change is applied at the next frame boundary and briefly freezes the output
- A "Weakened protections" list on the page and in the log: the background differ off, the person detector alone with the differ off, a threshold more than twice its startup value, a lookahead below its startup value or 0, a manual ROI over 5 percent of the frame, and the raw preview
- `petrify.server.control.RunControl`, the thread-safe link between the processing loop and the server
- A Mode section on the page: the active mode, its origin, description, and options, and the installed modes. Options the mode marks as `runtime` can be changed while the run is in progress; the others are shown as fixed. A plugin mode is marked with a notice that it runs in the petrify process and receives the unprocessed frames. The mode cannot be switched during a run
- The "Weakened protections" list also holds the sentences the active mode reports for its options (`ModeFactory.weakened`), or a sentence saying the mode could not report them

#### Modes and plugins
- Mode plugins: an installed package registers a mode under the `petrify.modes` entry point group, and `petrify run --mode NAME` selects it (`petrify.core.registry`). The built-in `erase` cannot be shadowed. A plugin that is not selected is never imported. A distribution whose metadata cannot be read is skipped with a warning, and a mode name that is invalid, registered more than once, or not `module:attr` is listed as not selectable with the reason
- `petrify run --list-modes` lists the modes without a `--source`; with `--mode NAME` it also prints that mode's options. `--mode-option KEY=VALUE` sets an option of the selected mode
- The mode contract in `petrify.core.plugin`: `ModeFactory` (`name`, `summary`, `options`, `needs_background`, `roi_overlay`, `requires`, `mandatory`, `create`, `weakened`), `ModeContext`, and `OptionSpec` with the kinds `bool`, `int`, `float`, `choice`, and `choices`, each stored in one canonical type. A `runtime` number option must declare a minimum and a maximum. A plugin mode must declare `detect_policy`
- When the selected mode comes from a plugin, the run logs a warning that it runs in the petrify process with petrify's permissions and receives the unprocessed frames
- `docs/PLUGINS.md`, a guide to writing a mode plugin

#### Tracking and models
- `--track` and `--track-hold N` (default 15) in `petrify run`: a greedy IoU box tracker (`petrify.detectors.tracker`) that holds person boxes and faces for N frames after a miss, without moving them. The erase mode does not accept it
- `--model-dir` and `petrify.models`: model files are looked up in an explicit path, `--model-dir`, `PETRIFY_MODEL_DIR`, then the per-user data directory, and loaded only when their size and SHA-256 match a pinned value. petrify never downloads anything
- Config sections for the YuNet face, DNN person, and segmentation detectors
- `petrify.background.any_person` combines several background sample checks into one

#### Model-based detectors
- YuNet face detector (`petrify.detectors.face_yunet.FaceYuNetDetector`, `--faces`, `--face-score-threshold`, default 0.6): face boxes with five landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner). Boxes are rounded outward. Frames with a side of 32 px or less are refused and fail closed
- NanoDet-Plus person detector (`petrify.detectors.person_dnn.PersonDnnDetector`, `--person-detector {hog,dnn}`, default `hog`, and `--dnn-score-threshold`, default 0.4). With `dnn` it replaces HOG for `--detector person` and `both`; the locked background pre-scan still uses HOG. Overlapping boxes are merged into their union, then padded by 10 percent of their width and 7 percent of their height on each side. It misses people smaller than about 110 px tall on a 1080p frame, and on the 640x480 demo clip takes about 34 ms per frame (about 77 ms at 1080p)
- PP-HumanSeg person segmenter (`petrify.detectors.segment.PersonSegmenter`, `--segment`, `--segment-threshold`, default 0.5): a full-resolution person mask that keeps every pixel the model flagged when it is resized to the frame. It misses small, distant people and marks parts of empty scenes
- The three model files are pinned in `petrify.models.KNOWN_MODELS` with their size, SHA-256, source (the OpenCV model zoo), and license: `face_detection_yunet_2023mar.onnx` (MIT), `object_detection_nanodet_2022nov.onnx` (Apache-2.0), and `human_segmentation_pphumanseg_2023mar.onnx` (Apache-2.0). `--model-dir` is now used. The log names the file each detector was loaded from
- `--background-check NAMES`: the detectors that check the samples of a `running` or `calibrate` background, from `hog`, `yunet`, and `dnn` (default `hog`). A warning names them when they reject 50 samples in a row and the background is still not ready
- `petrify.background.detector_check` turns a frame detector into a background sample check
- Warnings when `--person-detector dnn` is given with `--detector motion`, when `--faces`, `--segment`, or `--background-check` has no effect for the mode, and when the DNN person detector runs without the background differ
- The web page shows which person detector the run uses

### Changed
- `petrify INPUT OUTPUT` now runs through the streaming engine (`petrify.pipeline.run_file`) instead of the two-pass pipeline. Output for the same input and flags is unchanged, verified frame for frame against recorded fixtures. Background selection and timestamp ROI auto-detection still pre-scan the file before the processing pass
- The modules `petrify.motion`, `petrify.person`, `petrify.bgdiff`, and `petrify.compositor` moved to `petrify.detectors.motion`, `petrify.detectors.person_hog`, `petrify.detectors.bgdiff`, and `petrify.stages.roi_overlay`; the old module paths no longer exist. `petrify.Compositor`, `petrify.MotionDetector`, and `petrify.run` are still exported. `petrify.run` is still the old two-pass function, kept as the reference implementation for the golden tests; the CLI no longer calls it
- Log output: one progress bar and an `Engine done: frozen=... pass-through=...` summary replace the "Pass 1" and "Pass 2" blocks
- HOG downsampling never shrinks a frame below the 64x128 detection window and keeps the aspect ratio
- If a detector, the mode, or a stage fails on a frame, that frame is written as a black frame and counted (fail closed), a warning is logged, and the run continues
- Optional extras `web` (FastAPI, uvicorn) and `virtualcam` (pyvirtualcam); `httpx` added to the `dev` extra for the web tests
- In `petrify run`, `--hwaccel` is passed to the FFmpeg process that encodes `--output` but does not speed anything up, since petrify decodes the source itself; the help says so
- A live run ended with Ctrl-C finalizes its outputs; a second Ctrl-C while they are finalized aborts. A live run that fails keeps the recording written up to that point and logs the error. FFmpeg runs in its own session, so a Ctrl-C reaches petrify only
- Detector results are validated and normalized before they are merged (`petrify.core.detections.check_detections`). A detector that returns a malformed result now gives a black frame, counted as failed closed, instead of aborting the run. A detector's `provides` must hold `DetectionKind` members. Boxes reach the mode clipped to the frame, and the mask as a read-only copy
- `Detections.merged` returns the union of both results (flags OR-ed, boxes and faces concatenated, masks OR-ed) instead of letting the other side's fields override this one's
- `Detections.subject_present` also counts a face or a mask pixel as a subject
- A mode or stage that returns something other than a `numpy.ndarray` with the input frame's shape and dtype, or, on a frame where a subject is present, an array that may share memory with the input image, now fails that frame closed
- `petrify run` builds the mode through the registry: a detector is built only when the mode requires what it provides, the background only when the mode needs one, and the ROI stage only when the mode allows it
- `SegmentConfig.min_area_share` now accepts 0 and defaults to 0 (off). With the previous default a frame where the model flagged only part of a visible person counted as having no person
- `BackgroundConfig.sample_checkers` no longer accepts `segment`: the segmenter marks parts of empty scenes, so as a check it would reject nearly every sample
- Every model file a run needs, for its detectors and its background checks, is loaded and verified at startup, before any frame is read. A missing or corrupt model stops the run with one `Fatal error:` line (exit code 1); during a run, a settings change that needs one is rejected and the run continues
- `petrify run` refuses to start when the mode requires faces, landmarks, or a mask and the detector that provides it is not enabled (`--faces`, `--segment`), and when a `first_hit` mode requires faces, landmarks, or a mask together with another kind (it must declare `detect_policy = "all"`). Before, a mode that required them together with motion or person boxes received `None` in those fields
- `petrify run` rejects `--dnn-score-threshold`, `--face-score-threshold`, and `--segment-threshold` without their detector flag, and `--background-check` with `--no-background-check` or a `locked` background (usage error, exit code 2)
- With `--person-detector dnn` the web page rejects changes to the HOG settings, and a raised HOG threshold is not listed as a weakened protection while the DNN is in use

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
- A mode plugin is trusted code with petrify's permissions and the unprocessed frames; there is no sandbox. The checks on a plugin (factory and mode shape, names, option specs, `requires` and `mandatory`, output images) guard against mistakes, not against a hostile plugin
- Text from a plugin is shown on the web page as plain text with a length cap, and text from package metadata as printable ASCII, so a listed name cannot pass for another mode's. The ROI overlay, which pastes live pixels, runs only for a mode that allows it (`roi_overlay`), so a mode that obscures regions does not get them pasted back

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
