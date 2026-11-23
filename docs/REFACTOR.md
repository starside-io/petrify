# petrify refactor plan: real-time core + plugin modes

Status: phases 0 to 3 are done. Phase 4 is done, pending the user's sign-off: the plugin API, the mode registry, the tracker, the model file lookup, and the YuNet face, DNN person, and segmentation detectors have landed. Phase 5 is done, pending the user's sign-off: the `petrify-shadow` plugin (mode `shadow`) with its nine styles and five options, and the tracker fixes it needed in core. Phase 6 is done, pending the owner's sign-off: the held count on boxes, the opt-in head fill of the shadow mode, `--threads`, stop signal handling that keeps the recording, `--source-timeout`, `--profile cctv`, per-detector timings, a faster DNN resize, and fault injection tests. With it all phases are complete; the [findings from phase 6](#findings-from-phase-6) list what the owner still has to decide.

## Goals

1. Make petrify work in real time, not only on finished video files.
2. Accept input from many sources: files, webcams, small IP cameras (RTSP / HTTP MJPEG), image folders, and frames pushed one by one through a library API.
3. Support both webcam and CCTV use cases from one codebase.
4. Turn the processing behavior into pluggable modes:
   - `erase` (built in): today's petrify behavior, the person is frozen out of the scene.
   - `shadow` (plugin): the person stays visible but is anonymized (face blur, eye bar, silhouette, and more).
5. Output to a virtual camera so Zoom / Meet / Teams see the processed feed.
6. Provide a small local webapp to watch the result and tweak the configuration live.

## What blocks real time today

This table describes the pre-refactor two-pass pipeline (still kept as `petrify.run`, no longer used by the CLI), which assumes a complete, seekable file in several places:

| Location | Problem |
|---|---|
| `pipeline._find_clean_background` | Scans the whole video and seeks to the latest clean frame. Streams cannot seek or see the future. |
| `roi_detect.auto_detect_roi` | Samples frames across the whole clip. |
| `pipeline._classify_frames` + `_fill_gaps` | Two passes; gap fill looks both backward and forward in time. |
| `PersonDetector`, `BackgroundDiffer`, `MotionDetector` | Return `bool` only. No boxes, landmarks, or masks. |
| `encoder.ffmpeg_encode` | Re-encodes a temp file after processing. No live output. |

The detectors themselves already process one frame at a time, so they survive the refactor. Only the orchestration has to change.

## Core idea: one streaming engine

```
Source ──▶ Detectors (lazy) ──▶ Lookahead buffer ──▶ Mode (stage chain) ──▶ Sinks (in order)
                                   (K frames)          erase | shadow | ...
```

### Lookahead buffer

- The engine holds K frames of delay before a frame is emitted.
- File mode sets K = `gap_fill`, which reproduces today's `_fill_gaps` exactly (the fill is local, plus or minus `gap` frames). No separate two-pass code path is needed.
- Live mode uses a small K. Example: K=3 at 30 fps adds about 100 ms of latency, acceptable for video calls. CCTV can afford K=10 or more.

### Background model (online)

Three strategies:

- `locked`: pre-scan for a clean frame (file sources only, today's behavior).
- `running`: rolling median of recent frames classified as clean.
- `calibrate`: the user steps out of frame for a few seconds at startup.

### ROI auto-detect

Runs on the warmup window (first N seconds) instead of the whole clip, then locks.

### Lazy detectors

Each mode declares the detections it needs (`motion`, `persons`, `faces`, `landmarks`, `mask`). The engine only runs those detectors.

### Fail closed

If a detector raises, or the per-frame time budget is exceeded, the output frame is replaced with a black or fully blurred frame. Raw passthrough on error is never allowed. Fail-closed events are counted and reported.

### Threading

- Live sources: a capture thread keeps only the latest frame (older frames are dropped) so latency never accumulates.
- File sources: no frames are dropped.
- Sinks are written one after another, in order, on the engine thread. `petrify run` wraps none of them. A library caller can isolate a slow sink by wrapping it in `ThreadedSink`, which writes on a worker thread through a bounded queue (a full queue drops the oldest frame for a non-critical sink and blocks, with a timeout, for a critical one).
- A non-critical sink that fails is aborted, dropped, and counted in `sink_failures` while the run continues; if no sink is left the engine aborts. A critical sink that fails aborts the engine: every active sink is aborted, pending frames are discarded, and the error propagates.

## Proposed structure

Files marked "(not yet)" do not exist.

```
petrify/
  core/
    frame.py         # Frame: image, timestamp, index, source metadata
    detections.py    # Detections: motion flag, person boxes, faces + landmarks, mask; check_detections
    engine.py        # source -> detectors -> refiners -> lookahead -> mode -> stages -> sinks
    config.py        # typed config, hot-swappable at a frame boundary
    plugin.py        # protocols (Detector, Refiner, Mode, Stage, Sink, Source) + mode contract
                     # (ModeFactory, ModeContext, OptionSpec)
    registry.py      # petrify.modes entry point discovery, load_mode, build_mode
  models.py          # local model file lookup with pinned SHA-256; never downloads
  sources/
    file.py          # video files (supports pre-scan)
    camera.py        # webcam by index (AVFoundation / V4L2 / DirectShow)
    stream.py        # RTSP, HTTP MJPEG (IP cameras, ESP32-CAM, Raspberry Pi)
    images.py        # folder or glob of still images
    push.py          # library API: PushSession(engine, w, h).push(image) -> image | None
  detectors/
    motion.py        # existing MOG2
    bgdiff.py        # existing; returns a flag only (a mask output is not implemented yet)
    person_hog.py    # existing HOG, now returns boxes
    person_dnn.py    # NanoDet-Plus person boxes via cv2.dnn (--person-detector dnn)
    face_yunet.py    # faces + 5 landmarks (cv2.FaceDetectorYN, --faces)
    segment.py       # PP-HumanSeg person segmentation mask via cv2.dnn (--segment)
    tracker.py       # greedy IoU tracker + TrackerRefiner: holds person boxes and faces
                     # for N frames after a miss, without motion extrapolation
  background/        # locked | running | calibrate; vetting.py: person_free, any_person
  modes/
    erase.py         # today's petrify: frame freeze (later: mask-level erase); ERASE factory
  stages/
    roi_overlay.py   # live timestamp paste (existing Compositor)
  sinks/
    file.py          # ffmpeg stdin pipe, audio mux for file sources
    virtualcam.py    # pyvirtualcam
    mjpeg.py         # localhost stream for the webapp
    preview.py       # cv2 window (requires non-headless OpenCV)
    threaded.py      # ThreadedSink: writes a wrapped sink on a worker thread
    rtsp.py          # (not yet) re-publish for an NVR
  server/
    app.py           # FastAPI: config API + MJPEG stream + stats
    control.py       # RunControl: thread-safe hand-off between the loop and the server,
                     # runtime bounds, weakened protections, the mode for the page
    static/          # index.html, app.js, app.css
  cli.py             # `petrify run --source ... --mode ...`; legacy `petrify in out` kept

plugins/petrify-shadow/      # separate package with its own pyproject (mode `shadow`)
  pyproject.toml     # petrify.modes entry point: shadow = petrify_shadow:SHADOW
  README.md          # styles, options, coverage rules, measured limits
  src/
    petrify_shadow/
      __init__.py    # exports SHADOW
      options.py     # the six options, style groups, fixed drawing order, colors
      factory.py     # ShadowFactory: requires / mandatory per style, refusals, weakened
      mode.py        # plan (pure decision: pass, black, draw) and ShadowMode (render;
                     # held counts read from Box.held)
      regions.py     # face and person padding, held growth, mask preparation, orphan rule,
                     # head rect (orphan=head)
      styles/
        face.py      # gaussian, pixelate, noise, facebox (fill), eyebar
        body.py      # silhouette, silhouette_color, outline, bodybox
  tests/             # unit, contract, end-to-end (models marker), and timing tests
  tools/
    padding_sheet.py # contact sheets of padded face crops, to judge the padding
    head_sheet.py    # where heads sit in person boxes, to check the orphan=head rule
```

### Plugin mechanism

Implemented in phase 4. [PLUGINS.md](PLUGINS.md) is the guide for plugin authors; this is the summary.

- A plugin publishes a `ModeFactory` instance under the `petrify.modes` entry point group (`name = "module:attr"`). `petrify run --mode NAME` selects it, `--mode-option KEY=VALUE` sets its options, and `--list-modes` lists the modes.
- Built-in modes are declared in code (`registry.BUILTIN_MODES`), so `erase` works from a source checkout and an installed package can neither remove nor shadow it. A name registered more than once, an invalid name, or a value that is not `module:attr` is listed as not selectable with the reason. Discovery reads metadata only and never raises: a distribution with unreadable metadata is skipped with a warning. Only the selected mode is imported.
- The factory declares `name`, `summary`, `options` (`OptionSpec`), `needs_background`, `roi_overlay`, and the methods `requires(options)`, `mandatory(options)`, `create(ctx, previous)`, and `weakened(options)`. `create` receives a `ModeContext` (source info, the whole config, the resolved options, and the background when the factory needs one) and returns `previous` when nothing the mode depends on changed.
- The mode has `name`, `requires`, `render(frame, detections, subject_present)`, a `detect_policy` (`"first_hit"` or `"all"`, required for a plugin mode), and optionally `mandatory`. When `subject_present` is True the mode must cover the subject, cover the whole frame when it has no regions to draw on, and return an array that does not share memory with the input image; the engine checks the type, shape, dtype, and memory and fails the frame closed otherwise.
- Option values have one canonical type per kind (`bool`, `int`, `float`, `choice`, `choices`), because the pipeline detects a settings change with `!=`. An option marked `runtime` can be changed from the web page; a runtime number option must declare both bounds.
- `build_mode` checks what `create` returns. A `ValueError` from the factory means the settings are not acceptable: at startup the run stops with the message, and during a run the change is rejected and the previous settings stay. Any other exception, or a broken contract, raises `ModeLoadError`: the plugin is faulty. At startup that also stops the run; during a run it is reported as a rejected change.
- Installing `petrify-shadow` (phase 5) makes `--mode shadow` available in the CLI and lists it on the web page without changes to the plugin API. Phase 5 changed only the pipeline's tracker wiring (see [Findings from phase 5](#findings-from-phase-5)).
- Keeping shadow as its own package proves the plugin API is real and keeps heavy dependencies (e.g. `mediapipe`) out of core.

## Shadow mode styles

| Style | Needs |
|---|---|
| `gaussian` | Face box |
| `pixelate` | Face box |
| `noise` | Face box (random noise refilled every frame) |
| `facebox` | Face box, filled with a solid color. A person box whose face is missed is filled as an orphan; `orphan=head` (phase 6, opt-in) fills only its top half |
| `eyebar` | Eye landmarks (YuNet 5 points); bar rotates with the head |
| `silhouette` | Person mask, filled black |
| `silhouette_color` | Person mask, filled with a solid color (no blur) |
| `outline` | Person mask edges; refused on its own, since it hides no one |
| `bodybox` | Person box only; cheap fallback for CCTV angles where segmentation struggles |

Styles are stackable (e.g. `silhouette` + `eyebar`) and always drawn in a fixed order: body fills, orphan person boxes, outline, face covers, eye bar. The [plugin README](../plugins/petrify-shadow/README.md) describes each style and option as built.

Note: plain Gaussian blur and pixelation have been partially reversed by ML models. `noise`, `facebox`, and `silhouette` are the irreversible options and should be the recommended defaults for compliance use.

## Webcam vs CCTV profiles

Profiles are presets of defaults, not code forks. Every value can be overridden.

| | `--profile webcam` | `--profile cctv` |
|---|---|---|
| Person | segmentation mask (as the only person signal it lets a person the segmenter misses through: the shadow mode's orphan rule needs person boxes, see the findings from phase 5) | motion + bgdiff + DNN boxes |
| Face | YuNet (large, frontal faces) | YuNet when faces are large enough, else head-from-body box |
| Background | calibrate / running | locked / running |
| Lookahead | 0 to 3 frames | 10+ frames |
| Typical sink | virtual cam + webapp | file / RTSP + webapp |

Built in phase 6: `--profile cctv` only. It sets `--detector both`, `--person-detector dnn`, `--lookahead 10`, `--faces` when the mode uses faces or landmarks, `--background-check hog,dnn` when the mode uses a `running` or `calibrate` background, and `--source-timeout 10` on a live source, each only where the flag was not given. It does not estimate heads from bodies in core and never turns on `--track` or `--segment`. `--profile webcam` was not built, since nothing was tested on a webcam.

## Webapp

- FastAPI + uvicorn, one static HTML page with vanilla JS. No build step.
- Output preview as an MJPEG `<img>` stream.
- Raw side-by-side preview is opt-in, so the unprocessed feed is not exposed by default.
- Controls: mode, style, strength, box padding, hold frames, lookahead, debug overlay (boxes and landmarks). Changes are sent over REST and applied live at the next frame boundary.
- Stats: fps, per-stage timing, dropped frames, fail-closed events.
- Binds to `127.0.0.1` only.

Implemented in phase 3 (`petrify run --web`): the processed MJPEG feed, the opt-in raw preview (`--web-raw-preview`), run stats and per-stage timings, a stop button, and controls for the erase settings only (detector, detector thresholds, background differ, lookahead, frame budget, ROI), with a list of weakened protections. Changes are sent over REST only (the page polls the state); there is no WebSocket. Mode, style, strength, box padding, hold frames, and the debug overlay belong to later phases.

Added in phase 4: a Mode section that shows the active mode, its origin and options, and the installed modes, with controls for the options the mode marks `runtime`, the mode's own weakened protections in the weakened list, and a trust notice for a plugin mode. The mode cannot be switched during a run. Added in phase 5: style, strength, and box padding are options of the shadow mode (`styles`, `strength`, `padding`, with `color` and `dilate`), changeable from the page. The debug overlay is not implemented. Hold frames are set with `--track-hold` and are not changeable from the page. Added in phase 6: the shadow option `orphan`, and one line per detector and refiner with its mean and last call time.

## Findings from phases 2 and 3

Design points that later phases must address:

- The lookahead smooths a boolean (subject present or not), so gap-filled frames carry no boxes. Modes that need boxes on those frames (shadow, tracker hold) need another way to fill them.
- The detector short-circuit (the detectors after MOG2 stop at the first hit) is an erase policy. Modes that need every detection, such as boxes from all detectors, cannot use it.
- Online backgrounds (`running`, `calibrate`) need a stronger person or face detector for their sample check. HOG often misses a seated person framed from the chest up, the usual webcam view.
- A lasting lighting change freezes the output of an online background, which keeps showing the old background.
- The live paths (webcams, streams, virtual camera, preview) were developed against test doubles and still need verification on real hardware.

## Findings from phase 4

Design points that later phases (the detectors, `petrify-shadow`, CCTV hardening) must respect:

- The ROI overlay pastes the live pixels of the timestamp region over the mode's output. It is a per-mode choice (`ModeFactory.roi_overlay`): right for erase, wrong for any mode that obscures regions, which must set it to False, since a face inside the ROI would be pasted back unobscured.
- Frames frozen by the lookahead (gap fill, cold start, the frames around a reconfiguration) carry no regions: only the frame's own detections, usually none. A region mode must cover the whole frame on such frames. Holding regions across those frames needs another mechanism than the boolean lookahead.
- A mode cannot tell a cold-start or gap-filled frame from a frame with a detected subject: it only receives `subject_present`.
- Refiners do not run on a frame whose detection failed (that frame is black). Tracks do not age on such frames, so a hold lasts more wall-clock frames.
- `Engine.reconfigure(detectors=...)` alone does not reset a refiner; only refiners that are passed or newly activated are reset. The pipeline therefore hands the refiners over whenever the detectors change (`pipeline._rebuild_components`).
- Plugin code is trusted. There is no sandbox, and the plugin receives the unprocessed frames. The registry's and the engine's checks guard against mistakes only.
- Text from package metadata (names, versions, problems) is untrusted even for a plugin that is not selected, and is reduced to printable ASCII on the page. Text from the loaded plugin is reduced to plain text and capped.
- The DNN person detector replaces HOG for frame detection; it does not add to it. It misses far subjects: no person smaller than about 110 px tall on a 1080p frame was found (the frame is scaled to 416 px), so on CCTV footage the background differ is what covers distant people. It is also slower: about 34 ms per frame on the 640x480 demo clip and about 77 ms at 1080p, so a 25 fps 1080p live source drops frames.
- The segmenter marks parts of empty scenes (false positives) and misses small, distant people. It is therefore not offered as a background sample checker, and it should not be the only person signal on CCTV footage.
- Under `"first_hit"` the engine skips later detectors once one fires, so a mode that requires `faces`, `landmarks`, or `mask` together with another kind would get `None` in those fields on most frames with a subject. `registry.build_mode` refuses such a mode; it must declare `detect_policy = "all"`.
- The locked background pre-scan still uses HOG only, whatever `--person-detector` and `--background-check` say.
- The `--background-check` thresholds are tied to the detector flags: the `dnn` and `yunet` checkers use `--dnn-score-threshold` and `--face-score-threshold`, which are accepted only with `--person-detector dnn` and `--faces`. A checker's threshold can only be set by also enabling that detector: for `dnn` that replaces HOG on the frames, and for `yunet` with `erase` the face detector does not run on the frames, and the run does not warn about it, since `--faces` then takes effect through the checker.

## Findings from phase 5

Design points from building `petrify-shadow` that phase 6 and later plugins must respect:

- The tracker needed a fix in the pipeline. A tracker refiner was built for every kind the mode required, so a mode that requires person boxes with `--detector motion` and `--track` made the engine refuse to start (a persons refiner with no detector to feed it). Refiners are now built only for kinds a detector the run builds provides, a live detector switch rebuilds them, and a mode that requires landmarks without faces gets the faces refiner. When the tracker cannot hold a kind the mode requires, a warning says so and the web page lists it as a weakened protection. `--track` with a `first_hit` mode is refused for every mode, at startup and on a live change.
- A held box cannot be told from a fresh one: `Detections` carries no held marker. The plugin treats a box exactly equal to one on the previous frame as held, grows its cover by 10 percent per held frame up to twice its size, and stops trusting it as a region after 10 frames (the growth cap). A box that truly stays still, or a fresh detection with the same box, is also seen as held, which only covers more. A core marker (held flag or age per box) would replace this heuristic.
- The orphan rule (a person box the primary styles do not cover is filled black) is what catches a person the face detector or the segmenter misses. With the mask styles a box is an orphan below 30 percent mask coverage of the box or of its head zone. At those thresholds no person box the segmenter missed escaped on the demo clips, at the cost of turning 77 percent (640x480) and 86 percent (1080p) of correctly segmented persons into black boxes.
- Mask styles need person boxes. Without them the orphan rule cannot run, and the segmenter marks something on almost every frame: `silhouette` with `--detector motion` left a fully visible head on 94 of 655 frames of the 640x480 clip, against 0 with `--detector both --person-detector dnn`. The plugin logs a warning for that setup.
- The blur and pixelation styles leak: with two different real faces, about 25 percent (gaussian high) to 83 percent (pixelate low) of the difference between them survives the cover, nearly independent of face size, and averaging pixelated frames of a moving face recovers detail. They are reported as weakened protections; `noise`, `facebox`, `silhouette`, `silhouette_color`, and `bodybox` replace the pixels.
- `ModeFactory.requires(options)` and `mandatory(options)` see only the options, not the config. A rule that depends on the detector settings (`bodybox` with `--detector motion`) can only be enforced in `create`, which receives the config through `ModeContext`.
- `needs_background` is a fixed attribute and cannot follow the mode's options. The shadow mode sets it to False, so the background differ never runs for it and cannot serve as a presence signal; with the DNN person detector the run warns that small, distant people can be shown.
- A `choices` option arrives in typed order from the command line and in option order from the web page. A mode that compares options to decide whether to rebuild must not depend on that order; the plugin stores its styles as a set and draws in a fixed order.
- The plugin was tested on the two demo clips (CCTV-style footage) only, not on a real webcam or any live hardware.

## Findings from phase 6

What a maintainer must know after CCTV hardening. Every number was measured on the two demo clips on one development machine; none is a guarantee.

**Held boxes.**

- `Box.held` (`petrify/core/detections.py`) is the number of refined frames a box has been carried without a matching detection, 0 for a box not carried over. The tracker refiner sets it, cumulatively across chained refiners. Frames that failed closed are not counted. Only refiner results may carry it above 0: a detector result that does fails the frame closed, so a detector cannot make a fresh box look stale. It must be a Python `int`, at most `MAX_HELD` (1000000).
- `held` takes part in box equality and hashing, and `Box` now has six fields. A plugin that compared a box with the previous frame's box to guess that it was held no longer sees a held box as equal, and code that unpacks a box into five values breaks. [PLUGINS.md](PLUGINS.md#what-the-mode-receives) says so.
- `petrify-shadow` reads `held` and keeps no per-frame state apart from its noise generator. A held cover grows 25 percent per held frame up to 3 times, and a box held 8 frames or more is drawn but is not a region. This replaces the phase 5 rule (10 percent per frame, twice the size, 10 frames).
- Three stated coverage reductions against that rule, all from one cause (a fresh detection equal to the previous frame's box is no longer taken for a held one): such a box is not grown and its face clears its person box again; a still subject is no longer cut off (black) after 10 identical frames; a held box that follows a still period grows from the held frames only. On the 640x480 clip with the tracker, 27 or 28 frames per run were covered less than before, and on none of them was a detected face or person box left uncovered.

**Shadow `orphan` option.**

- `orphan=box` (default, unchanged) fills the whole orphan box. `orphan=head` fills only the top half of an orphan box that is fresh, at least 150 px tall, and at least 1.4 times as tall as wide; every other box, and under the mask styles a box the segmenter missed as a whole, keeps the whole fill. It is listed as a weakened protection unless `bodybox` is selected.
- On 400 frames (two ranges of 200 frames, one per clip) no real head or face was left visible. The tightest pose was arms raised over the head (face at 0.40 of the box height, about 6 percent of the box height to spare). Bending, sitting low, and two people in one box were not in the footage and stay a risk.
- An earlier figure (the padded face inside the top half in every case) was measured on boxes a face had cleared, which have the face in their top third by construction; it is not evidence for orphan boxes. The 400 frame result is.
- On those ranges, with `silhouette` 94 of 168 orphan boxes got the head rect and 74 kept the whole-box fill (1080p), 95 of 141 and 46 (640x480); with `gaussian` alone 97 of 97 (1080p) and 108 of 116 (640x480) got the head rect. Under the mask styles a large share keeps the whole fill, so `head` changes the most for face-only stacks.

**Several cameras and stops.**

- One process per camera, each with its own `--output` and `--web-port`. `--threads N` limits OpenCV's worker threads and logs the count OpenCV reports, with a warning when it differs: the macOS builds honor only 1. FFmpeg's encoder threads are not limited.
- Capacity with the DNN detector on the 1080p clip, 12 core machine: 1 process about 23 to 25 fps, 2 processes about 8.5 to 10 fps each, 4 processes about 6.5 to 7.5 fps each. Two processes together deliver less than one alone, because the decoder and the encoder also use threads.
- SIGTERM is handled like Ctrl-C in `petrify run`. On a live source the first signal is deferred during the open, every frame, and the close, and the recording is finalized; a second signal aborts; a file source aborts. A crash on Ctrl-C that lost the recording is fixed: the handler logged, and a signal during another write to stderr raised a reentrant call error inside the frame step.
- When the stop signal reaches the whole process group (the default of `systemctl stop`), FFmpeg finalizes its own file. The sink keeps it only after reading it back, with a warning that it ends early and how many frames it holds; any other FFmpeg failure fails the run as before. 10 of 10 group stops kept the recording, against 0 of 10 before the fix.
- Known limit: the older form `petrify INPUT OUTPUT` writes its final encode directly to the output path, so a stop can leave a truncated file there.
- `--source-timeout SECONDS` ends a stalled live run with an error (exit code 1) and keeps the recording. It covers the first frame and applies to live sources only. It does not cover a detector call that never returns: the frame budget is checked after the call, so a hung detector stalls the output.

**Profile, timings, speed, tests.**

- `--profile cctv` sets `--detector both`, `--person-detector dnn`, `--lookahead 10`, `--faces` when the mode uses faces or landmarks, `--background-check hog,dnn` for a mode with a `running` or `calibrate` background, and `--source-timeout 10` on a live source; explicit flags always win, also abbreviated or in `--flag=value` form. The run logs `Profile cctv: set ...; explicit flags win`. There is no webcam profile, because nothing was tested on a webcam.
- The engine times each detector and refiner (label: class name, plus the detection kind for the trackers). The settings page shows one line per component, and the run logs `Detector and refiner times (moving average): ...` after the summary line. The values are exponentially weighted moving averages (alpha 0.1), not means over the whole run.
- The DNN person detector resizes its letterbox one axis at a time on large frames: byte-identical output, about 9.5 ms saved per 1080p frame, used only for frame sizes where it was shown identical (both axes shrink at least 3 times and neither by a whole factor).
- Detector cost per frame, median, idle machine: DNN person detector about 33 ms at 640x480 and 45 ms at 1080p; YuNet about 7 ms and 29 ms; segmenter about 10 ms and 15 ms; motion 1 to 2 ms. A shadow run with the DNN and faces at 1080p runs at about 12 to 13 fps.
- `tests/faults.py` is a fault injection harness (detectors, refiners, stages, and a mode that inject scheduled faults, a deterministic clock, a recording sink). `tests/test_fail_closed_property.py` runs 300 seeded scenarios and checks at the sink that every written frame is black or the mode's output for that frame, that every faulted frame is black and counted, and that the lookahead and reconfiguration windows hold.
- `plugins/petrify-shadow/tools/head_sheet.py` measures where heads sit in person boxes and writes contact sheets, using the mode's own `head_rect`.

**Not built, by decision.**

- One process with several cameras, and a launcher for several processes.
- Detecting on every Nth frame only (a detection stride).
- A `--person-detector both` option (HOG and the DNN together).
- Running YuNet at a reduced resolution: it lost 20 percent of the faces.
- Tiling the DNN for distant people: 3.3 times the cost, and it found nobody under 110 px.

**Open decisions for the owner.**

1. Whether to add `--person-detector both`.
2. Whether to delete the old two-pass `petrify.run()`, which the golden tests still use as their reference.
3. The exit code when frames failed closed: 0 today, so a service manager or script does not see them.
4. Whether to narrow the head zone of `orphan=head`.
5. A webcam profile, once something has been measured on a webcam.
6. Whether the older form `petrify INPUT OUTPUT` should write a partial file and rename it, as `petrify run --output` does.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | Golden tests (done) | Per-frame classification of the current pipeline on the demo clips is captured as test fixtures. |
| 1 | Core engine refactor (done) | `petrify in.mp4 out.mp4` matches the golden output through the new engine. |
| 2 | Live sources + sinks (done) | Webcam, RTSP, image folder, and push API inputs; preview, ffmpeg pipe, and virtual cam outputs. Erase mode runs live. |
| 3 | Webapp (done) | Live feed visible in the browser; erase config tweakable live. |
| 4 | Plugin API + new detectors (done, pending sign-off) | Entry-point loading, the mode contract and registry, `--mode`, `--mode-option`, `--list-modes`, detection policies and refiners in the engine, the tracker (`--track`), model file lookup with pinned SHA-256 (`--model-dir`), the Mode section of the web page, and the YuNet face (`--faces`), DNN person (`--person-detector dnn`), and segmentation (`--segment`) detectors with their three pinned model files, a startup check of every model file a run needs, and `--background-check`. |
| 5 | `petrify-shadow` plugin (done, pending sign-off) | All styles implemented and stackable: the nine styles and five options of the mode `shadow`, the orphan rule, held-box growth with the tracker, weakened protections, and the tracker fixes in the pipeline. |
| 6 | CCTV hardening (done, pending sign-off) | Held count on boxes (`Box.held`) read by the shadow mode, the opt-in head fill `orphan=head`, one process per camera with `--threads`, SIGTERM handled like Ctrl-C and the recording kept on a process group stop, `--source-timeout`, `--profile cctv`, per-detector and per-refiner timings, a faster DNN letterbox resize, and the fault injection harness with a fail-closed property test. See [Findings from phase 6](#findings-from-phase-6). |

## Open decisions

1. **Plugin packaging.** Decided: a separate package in the same repo at `plugins/petrify-shadow`, loaded through the `petrify.modes` entry point.
2. **Preview transport.** Decided: MJPEG first. No WebRTC.
3. **Virtual camera on macOS.** Decided: `pyvirtualcam`. Requiring OBS 30+ on macOS is acceptable, as a dependency of the virtual camera sink only.

## Compliance notes (not legal advice)

- GDPR does not prohibit faces on camera. CCTV footage is personal data with or without blurring and needs a lawful basis either way.
- What is restricted is biometric identification (GDPR Art. 9, and the EU AI Act ban on real-time remote biometric identification in public spaces). The AI Act also bans emotion recognition in workplaces and schools, so any "suspicious activity" analytics need their own review.
- Face detection used only to obfuscate is generally not identification, and supports data minimisation and privacy by design (GDPR Art. 5 and 25).
- This only holds if obfuscation happens on the device before storage. If the raw stream is stored anywhere, the benefit is largely lost.
