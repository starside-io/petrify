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
- Exit codes: 0 when the run completes (frames that failed closed and a live run ended with Ctrl-C or SIGTERM included), 1 for a fatal error (a missing FFmpeg and a `--source-timeout` included), 2 for a missing input or an invalid option, 130 for a Ctrl-C or SIGTERM that aborts a run (a file or image folder run, or a second signal)
- `--source-timeout SECONDS`: a live source that delivers no frame for that long, the first frame included, ends the run with an error (exit code 1) and the recording made so far is kept. Refused for a file or image folder source
- `--profile cctv`: sets `--detector both`, `--person-detector dnn`, `--lookahead 10`, `--faces` when the mode uses faces or landmarks, `--background-check hog,dnn` when the mode uses a `running` or `calibrate` background, and `--source-timeout 10` on a live source, each only where the flag was not given (also abbreviated or in `--flag=value` form). The run logs what it set. There is no webcam profile
- `--threads N` (1 to 64): limits OpenCV's worker threads, for several petrify processes on one machine. The run logs the count OpenCV reports and warns when it differs from the request (some builds, the macOS ones, honor only 1). FFmpeg's encoder threads are not limited
- README: a "Running several cameras" section (one process per camera, a systemd unit template, what a stop does to the recording, measured capacity) and a table of measured detector costs

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
- `Box.held`: the number of refined frames a box has been carried without a matching detection (0 for a box not carried over). The tracker refiner sets it, cumulatively across chained refiners; frames that failed closed are not counted. Only refiner results may carry it above 0: a detector result that does fails the frame closed. It must be a Python `int` from 0 to `MAX_HELD`, and it takes part in box equality and hashing
- Per-detector and per-refiner timings: `Engine.timings.detectors` holds a rolling call time for each active detector and refiner, labelled by class name (and detection kind for the trackers). The web page shows one line per component, and the run logs `Detector and refiner times (moving average): ...` after the summary line
- `tests/faults.py`, a fault injection harness, and `tests/test_fail_closed_property.py`, a property test of 300 seeded scenarios that checks at the sink that every written frame is black or the mode's output for that frame, that every faulted frame is black and counted, and that the lookahead and reconfiguration windows hold

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
- `--track` and `--track-hold N` (default 15) in `petrify run`: a greedy IoU box tracker (`petrify.detectors.tracker`) that holds person boxes and faces for N frames after a miss, without moving them. The erase mode does not accept it: the run stops with a `Fatal error:` line (exit code 1) before any frame is read
- `--model-dir` and `petrify.models`: model files are looked up in an explicit path, `--model-dir`, `PETRIFY_MODEL_DIR`, then the per-user data directory, and loaded only when their size and SHA-256 match a pinned value. petrify never downloads anything
- Config sections for the YuNet face, DNN person, and segmentation detectors
- `petrify.background.any_person` combines several background sample checks into one

#### Model-based detectors
- YuNet face detector (`petrify.detectors.face_yunet.FaceYuNetDetector`, `--faces`, `--face-score-threshold`, default 0.6): face boxes with five landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner). Boxes are rounded outward. Frames with a side of 32 px or less are refused and fail closed
- NanoDet-Plus person detector (`petrify.detectors.person_dnn.PersonDnnDetector`, `--person-detector {hog,dnn}`, default `hog`, and `--dnn-score-threshold`, default 0.4). With `dnn` it replaces HOG for `--detector person` and `both`; the locked background pre-scan still uses HOG. Overlapping boxes are merged into their union, then padded by 10 percent of their width and 7 percent of their height on each side. It misses people smaller than about 110 px tall on a 1080p frame, and on an idle machine takes a median of about 33 ms per frame on the 640x480 demo clip (about 45 ms at 1080p)
- PP-HumanSeg person segmenter (`petrify.detectors.segment.PersonSegmenter`, `--segment`, `--segment-threshold`, default 0.5): a full-resolution person mask that keeps every pixel the model flagged when it is resized to the frame. It misses small, distant people and marks parts of empty scenes
- The three model files are pinned in `petrify.models.KNOWN_MODELS` with their size, SHA-256, source (the OpenCV model zoo), and license: `face_detection_yunet_2023mar.onnx` (MIT), `object_detection_nanodet_2022nov.onnx` (Apache-2.0), and `human_segmentation_pphumanseg_2023mar.onnx` (Apache-2.0). `--model-dir` is now used. The log names the file each detector was loaded from
- `petrify.models.load_model` returns a `LoadedModel` with the verified bytes and the path they were read from (`load_model_bytes` returns only the bytes). The model-based detectors expose that path as the read-only `model_path` (None when built directly from bytes)
- `--background-check NAMES`: the detectors that check the samples of a `running` or `calibrate` background, from `hog`, `yunet`, and `dnn` (default `hog`). A warning names them when they reject 50 samples in a row and the background is still not ready
- `petrify.background.detector_check` turns a frame detector into a background sample check
- Warnings when `--person-detector dnn` is given with `--detector motion`, when `--faces`, `--segment`, or `--background-check` has no effect for the mode, and when the DNN person detector runs without the background differ. `--faces` and `--person-detector dnn` are not reported as having no effect when their settings are used by the `yunet` or `dnn` background sample check
- The web page shows which person detector the run uses

#### Shadow mode plugin
- `petrify-shadow` (`plugins/petrify-shadow`), a separate package that registers the mode `shadow` through the `petrify.modes` entry point. It keeps the live frame and covers the people in it while a subject is present. Install it from the repository root with `python -m pip install --no-deps -e plugins/petrify-shadow`. See `plugins/petrify-shadow/README.md` for its coverage rules and measured limits
- Nine styles, stackable and always drawn in a fixed order: face covers `gaussian`, `pixelate`, `noise`, `facebox`, and `eyebar` (need `--faces`), mask styles `silhouette`, `silhouette_color`, and `outline` (need `--segment`), and `bodybox` (needs `--detector person` or `both`). `outline` on its own, an empty style list, and `bodybox` with `--detector motion` are refused
- Six options, all changeable while a run is in progress: `styles` (default `facebox`), `padding` (0.3, 0 to 1), `strength` (`high`, `medium`, `low`), `color` (`black`, `white`, `gray`, `red`, `green`, `blue`), `dilate` (0.01, 0 to 0.1), and `orphan` (`box` or `head`, default `box`)
- A frame with a subject but no region to draw on is written black, and a person box that the selected styles do not cover (no face in its top third, or too little mask) is filled black. With `--track`, the mode reads `Box.held`: a held box grows its cover by 25 percent per held frame up to 3 times its size, and a box held 8 frames or more is drawn but no longer counts as a region
- `orphan=head` fills only the top half of an orphan person box that is fresh, at least 150 px tall, and at least 1.4 times as tall as wide; every other box, and under the mask styles a box the segmenter missed as a whole, keeps the whole-box fill. It is opt-in and listed as a weakened protection. On 400 frames of the two demo clips (200 per clip) no real head or face was left visible; bending, sitting low, and two people in one box were not in the footage and remain a risk
- Weakened protections reported by the mode: no body style, the reversible `gaussian` and `pixelate` covers and a lower `strength`, `eyebar` as the only face style, `padding` or `dilate` below the default, a silhouette without `bodybox`, and `orphan=head` without `bodybox`. A warning is logged for a mask style without `bodybox` and with `--detector motion`
- `plugins/petrify-shadow/tools/padding_sheet.py` writes contact sheets of padded face crops, to judge the face padding by eye, and `plugins/petrify-shadow/tools/head_sheet.py` measures where heads sit in person boxes and writes contact sheets, to check the `orphan=head` rule. The plugin's tests are part of the root test run

### Changed
- `petrify INPUT OUTPUT` now runs through the streaming engine (`petrify.pipeline.run_file`) instead of the two-pass pipeline. Output for the same input and flags is unchanged, verified frame for frame against recorded fixtures. Background selection and timestamp ROI auto-detection still pre-scan the file before the processing pass
- The modules `petrify.motion`, `petrify.person`, `petrify.bgdiff`, and `petrify.compositor` moved to `petrify.detectors.motion`, `petrify.detectors.person_hog`, `petrify.detectors.bgdiff`, and `petrify.stages.roi_overlay`; the old module paths no longer exist. `petrify.Compositor`, `petrify.MotionDetector`, and `petrify.run` are still exported. `petrify.run` is still the old two-pass function, kept as the reference implementation for the golden tests; the CLI no longer calls it
- Log output: one progress bar and an `Engine done: frozen=... pass-through=...` summary replace the "Pass 1" and "Pass 2" blocks
- HOG downsampling never shrinks a frame below the 64x128 detection window and keeps the aspect ratio
- If a detector, the mode, or a stage fails on a frame, that frame is written as a black frame and counted (fail closed), a warning is logged, and the run continues
- Optional extras `web` (FastAPI, uvicorn) and `virtualcam` (pyvirtualcam); `httpx` added to the `dev` extra for the web tests
- In `petrify run`, `--hwaccel` is passed to the FFmpeg process that encodes `--output` but does not speed anything up, since petrify decodes the source itself; the help says so
- A live run ended with Ctrl-C finalizes its outputs; a second Ctrl-C while they are finalized aborts. A live run that fails keeps the recording written up to that point and logs the error. FFmpeg runs in its own session, so a Ctrl-C reaches petrify only
- `petrify run` handles SIGTERM like Ctrl-C, so `systemctl stop` and `docker stop` finalize a live recording. One handler covers the whole run: on a live source a first signal during the open, a frame, or the close takes effect once that step is done, and a second signal aborts. A file source aborts
- The shadow mode reads the held count from `Box.held` instead of taking a box equal to one on the previous frame for a held one, and keeps no per-frame state apart from its noise generator. This covers less in three stated cases: a fresh detection equal to the previous frame's box is not grown and its face clears its person box again; a still subject is no longer cut off (black) after 10 identical frames; a held box that follows a still period grows from the held frames only. On the 640x480 demo clip with the tracker, 27 or 28 frames per run were covered less, and on none of them was a detected face or person box left uncovered
- A plugin that compares boxes across frames, or unpacks a `Box` into five values, is affected by `Box.held`: a held box is no longer equal to the box it was carried over from, and `Box` has six fields
- The DNN person detector resizes a large frame for the model one axis at a time where that was shown to give byte-identical output (both axes shrink at least 3 times and neither by a whole factor), which saves about 9.5 ms per 1080p frame. Other frame sizes keep the single resize; detections on the demo clips are unchanged
- Detector results are validated and normalized before they are merged (`petrify.core.detections.check_detections`). A detector that returns a malformed result now gives a black frame, counted as failed closed, instead of aborting the run. A detector's `provides` must hold `DetectionKind` members. Boxes reach the mode clipped to the frame, and the mask as a read-only copy
- `Detections.merged` returns the union of both results (flags OR-ed, boxes and faces concatenated, masks OR-ed) instead of letting the other side's fields override this one's
- `Detections.subject_present` also counts a face or a mask pixel as a subject
- A mode or stage that returns something other than a `numpy.ndarray` with the input frame's shape and dtype, or, on a frame where a subject is present, an array that may share memory with the input image, now fails that frame closed
- `petrify run` builds the mode through the registry: a detector is built only when the mode requires what it provides, the background only when the mode needs one, and the ROI stage only when the mode allows it. The mode is loaded once; `run_stream` accepts the loaded factory as `factory`
- The `petrify run` startup line `Detector mode: <setting>` says when the mode does not require the motion or person detector the setting selects, and then lists the detectors built, for example `Detector mode: motion (not used: the mode 'facebox' does not require motion). Detectors built: FaceYuNetDetector`
- `SegmentConfig.min_area_share` now accepts 0 and defaults to 0 (off). With the previous default a frame where the model flagged only part of a visible person counted as having no person
- `BackgroundConfig.sample_checkers` no longer accepts `segment`: the segmenter marks parts of empty scenes, so as a check it would reject nearly every sample
- Every model file a run needs, for its detectors and its background checks, is loaded and verified at startup, before any frame is read. A missing or corrupt model stops the run with one `Fatal error:` line (exit code 1); during a run, a settings change that needs one is rejected and the run continues
- `petrify run` refuses to start when the mode requires faces, landmarks, or a mask and the detector that provides it is not enabled (`--faces`, `--segment`), and when a `first_hit` mode requires faces, landmarks, or a mask together with another kind (it must declare `detect_policy = "all"`). Before, a mode that required them together with motion or person boxes received `None` in those fields
- `petrify run` rejects `--dnn-score-threshold`, `--face-score-threshold`, and `--segment-threshold` without their detector flag, and `--background-check` with `--no-background-check` or a `locked` background (usage error, exit code 2)
- With `--person-detector dnn` the web page rejects changes to the HOG settings, and a raised HOG threshold is not listed as a weakened protection while the DNN is in use
- `--track` is refused with every mode that uses the `first_hit` detection policy, at startup and on a live change, whatever trackers would be built
- When `--track` is on and cannot hold a kind the mode requires (for example person boxes with `--detector motion`), a warning says so and the web page lists it as a weakened protection; a mode that requires neither person boxes nor faces gets a warning that the tracker has no effect

### Removed
- MOG2 pre-warmup: it never had an effect, because its model was reset before use
- `--adaptive-background` flag: it was parsed but never read

### Fixed
- Frames smaller than the HOG detection window (64x128) now produce a clear error instead of crashing the process
- A negative `--gap-fill` is rejected
- `petrify run` without `--source` but with an option only the run form has (for example `petrify run --output out.mp4`) is a usage error that says `petrify run` requires `--source` (exit code 2), instead of an "unrecognized arguments" error from the file form
- A failed run never leaves an encoded output from partial frames
- A mode that requires person boxes, run with `--detector motion` and `--track`, no longer makes the engine refuse to start: a tracker refiner is now built only for a kind that a detector of the run provides, a live detector switch rebuilds the refiners, and a mode that requires landmarks without faces gets the faces refiner
- A live recording is no longer lost on Ctrl-C: the signal handler logged, and a signal that arrived during another write to stderr raised a reentrant call error inside the frame step, which aborted the engine. The handler no longer logs
- A stop signal sent to the whole process group (the default of `systemctl stop`) no longer deletes the recording. FFmpeg finalizes its own file on that signal; the sink now keeps it after reading it back, with a warning that it ends early and how many frames it holds. Any other FFmpeg failure fails the run as before. 10 of 10 group stops kept the recording, against 0 of 10 before. The older form `petrify INPUT OUTPUT` still writes to the final path directly, so a stop there can leave a truncated file

### Security
- Stream URLs are redacted in logs and error messages: at most the scheme, host, and port are shown, credentials become `***`, the path and query are dropped, and an ambiguous URL hides the host. A mistyped stream spec is not echoed with its credentials
- The web page listens on `127.0.0.1` only and accepts only `127.0.0.1:PORT` as `Host` and `Origin` (DNS rebinding); `localhost` is refused because the server does not listen on `[::1]`, where another local user could. Every route except the page and its assets needs the per-run token printed in the link (`http://127.0.0.1:PORT/#token=...`); the token is never logged and the access log is off. No redirects, no CORS headers, a strict Content-Security-Policy, and bounded connections
- The unprocessed feed is served only with `--web-raw-preview`, and is then listed as a weakened protection
- Values set from the web page are bounded so every detector can still fire, and a manual ROI set there may not exceed the `--roi-max-area` cap, since its region is shown live
- `--output` equal to the input file, or to one of the images of an image folder source, is rejected
- A mode plugin is trusted code with petrify's permissions and the unprocessed frames; there is no sandbox. The checks on a plugin (factory and mode shape, names, option specs, `requires` and `mandatory`, output images) guard against mistakes, not against a hostile plugin
- Text from a plugin is shown on the web page as plain text with a length cap, and text from package metadata as printable ASCII, so a listed name cannot pass for another mode's. The ROI overlay, which pastes live pixels, runs only for a mode that allows it (`roi_overlay`), so a mode that obscures regions does not get them pasted back

---

## [0.1.0] - 2026-04-10

### Added

#### Core pipeline
- Two-pass architecture: **Pass 1** classifies every frame; **Pass 2** composites
- **MOG2** adaptive background subtraction for motion detection
- **HOG** person detector (`cv2.HOGDescriptor` with `getDefaultPeopleDetector()`) for stopped subjects
- **BackgroundDiffer**: static diff against the locked clean reference frame; catches subjects that MOG2 adapts away (standing still) and tiny/partial subjects HOG misses (far away, cut off at frame edge)
- Temporal **gap fill** (`--gap-fill`): expand every detected region ±N frames to bridge missed detections mid-sequence
- **Cold-start freeze**: unconditionally freeze the first 50 frames to cover the MOG2 warmup window
- **MOG2 pre-warmup**: feed the locked background frame ×history times before classification to establish a trained baseline at frame 0

#### Auto-detection
- **Auto ROI detection** (two-pass: frequency anchor + edge-based X extension): locates the burned-in timestamp rectangle automatically; prints a `--roi` hint so users can skip the scan on future runs
- **Auto background selection**: scans every 5th frame with HOG, locks the latest clean frame (no person) as the reference background

#### CLI
- `petrify INPUT OUTPUT`: zero-config entry point
- `--roi X Y W H` / `--no-roi`: manual or disabled timestamp overlay
- `--detector {motion,person,both}`: select active detectors
- `--threshold`, `--history`, `--gap-fill`, `--downsample`: motion tuning
- `--hog-min-confidence`, `--hog-min-box-area`: HOG tuning
- `--bg-diff-pixel-threshold`, `--bg-diff-area-threshold`, `--no-bg-diff`: background differ tuning
- `--background-frame N`: manual background frame override
- `--roi-scan-samples`, `--roi-scan-min-freq`, `--roi-scan-edge-threshold`, `--roi-scan-edge-gap-tolerance`: ROI scan tuning
- `--codec`, `--crf`, `--preset`, `--hwaccel`: FFmpeg encoding options
- `--verbose`: DEBUG-level log output
- ASCII art banner at startup with `Made with ❤️ by starside.io`

#### Infrastructure
- `pipx install petrify` single-command install; `petrify` available globally on PATH
- Homebrew FFmpeg fallback in encoder (`/opt/homebrew/bin/ffmpeg`)
- macOS quarantine strip (`xattr -d com.apple.quarantine`) applied automatically to output
- `pyproject.toml` with hatchling build backend, Apache 2.0 classifier

[Unreleased]: https://github.com/starside-io/petrify/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/starside-io/petrify/releases/tag/v0.1.0
