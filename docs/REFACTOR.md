# petrify refactor plan: real-time core + plugin modes

Status: phases 0 and 1 are done. Phases 2 to 6 are planned and not implemented yet.

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
Source ──▶ Detectors (lazy) ──▶ Lookahead buffer ──▶ Mode (stage chain) ──▶ Sinks (fan-out)
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
- Sinks run as a fan-out so a slow sink does not stall processing.

## Proposed structure

```
petrify/
  core/
    frame.py         # Frame: image, timestamp, index, source metadata
    detections.py    # Detections: motion flag, person boxes, faces + landmarks, mask
    engine.py        # source -> detectors -> lookahead -> mode -> sinks
    config.py        # typed config, hot-swappable at a frame boundary
    plugin.py        # Stage / Mode protocol + entry-point registry
  sources/
    file.py          # video files (supports pre-scan)
    camera.py        # webcam by index (AVFoundation / V4L2 / DirectShow)
    stream.py        # RTSP, HTTP MJPEG (IP cameras, ESP32-CAM, Raspberry Pi)
    images.py        # folder or glob of still images
    push.py          # library API: engine.push(frame) -> frame
  detectors/
    motion.py        # existing MOG2
    bgdiff.py        # existing, extended to return a mask
    person_hog.py    # existing HOG, now returns boxes
    person_dnn.py    # fast person boxes via cv2.dnn for real-time CCTV
    face_yunet.py    # faces + 5 landmarks (cv2.FaceDetectorYN)
    segment.py       # person segmentation mask (optional extra)
    tracker.py       # IoU tracker + hold boxes for N frames after a miss
  background/        # locked | running | calibrate
  modes/
    erase.py         # today's petrify: frame freeze (later: mask-level erase)
  stages/
    roi_overlay.py   # live timestamp paste (existing Compositor)
  sinks/
    file.py          # ffmpeg stdin pipe, audio mux for file sources
    virtualcam.py    # pyvirtualcam
    mjpeg.py         # localhost stream for the webapp
    preview.py       # cv2 window (requires non-headless OpenCV)
    rtsp.py          # later: re-publish for an NVR
  server/
    app.py           # FastAPI: config API + MJPEG stream + stats
    static/index.html
  cli.py             # `petrify run --source ... --mode ...`; legacy `petrify in out` kept

plugins/petrify-shadow/      # separate package with its own pyproject
  petrify_shadow/
    mode.py          # ShadowMode: declares required detections per style
    styles/          # gaussian, pixelate, noise, eyebar, facebox,
                     # silhouette, silhouette_color, outline, bodybox
```

### Plugin mechanism

- Modes register through the `petrify.modes` Python entry point.
- Installing `petrify-shadow` makes `--mode shadow` available in the CLI and lists it in the webapp automatically.
- Keeping shadow as its own package proves the plugin API is real and keeps heavy dependencies (e.g. `mediapipe`) out of core.

## Shadow mode styles

| Style | Needs |
|---|---|
| `gaussian` | Face box |
| `pixelate` | Face box |
| `noise` | Face box (random noise refilled every frame) |
| `facebox` | Face box, with head-from-body fallback |
| `eyebar` | Eye landmarks (YuNet 5 points); bar rotates with the head |
| `silhouette` | Person mask, filled black |
| `silhouette_color` | Person mask, filled with a color or blur |
| `outline` | Person mask edges |
| `bodybox` | Person box only; cheap fallback for CCTV angles where segmentation struggles |

Styles are stackable (e.g. `silhouette` + `eyebar`).

Note: plain Gaussian blur and pixelation have been partially reversed by ML models. `noise`, `facebox`, and `silhouette` are the irreversible options and should be the recommended defaults for compliance use.

## Webcam vs CCTV profiles

Profiles are presets of defaults, not code forks. Every value can be overridden.

| | `--profile webcam` | `--profile cctv` |
|---|---|---|
| Person | segmentation mask | motion + bgdiff + DNN boxes |
| Face | YuNet (large, frontal faces) | YuNet when faces are large enough, else head-from-body box |
| Background | calibrate / running | locked / running |
| Lookahead | 0 to 3 frames | 10+ frames |
| Typical sink | virtual cam + webapp | file / RTSP + webapp |

## Webapp

- FastAPI + uvicorn, one static HTML page with vanilla JS. No build step.
- Output preview as an MJPEG `<img>` stream.
- Raw side-by-side preview is opt-in, so the unprocessed feed is not exposed by default.
- Controls: mode, style, strength, box padding, hold frames, lookahead, debug overlay (boxes and landmarks). Changes are sent over REST or WebSocket and applied live at the next frame boundary.
- Stats: fps, per-stage timing, dropped frames, fail-closed events.
- Binds to `127.0.0.1` only.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | Golden tests (done) | Per-frame classification of the current pipeline on the demo clips is captured as test fixtures. |
| 1 | Core engine refactor (done) | `petrify in.mp4 out.mp4` matches the golden output through the new engine. |
| 2 | Live sources + sinks | Webcam, RTSP, image folder, and push API inputs; preview, ffmpeg pipe, and virtual cam outputs. Erase mode runs live. |
| 3 | Webapp | Live feed visible in the browser; erase config tweakable live. |
| 4 | Plugin API + new detectors | YuNet, segmentation, DNN person detector, tracker. Entry-point loading works. |
| 5 | `petrify-shadow` plugin | All styles implemented and stackable. |
| 6 | CCTV hardening | Head-from-body fallback, multi-camera, performance budgets, fail-closed tests. |

## Open decisions

1. **Plugin packaging.** Decided: a separate package in the same repo at `plugins/petrify-shadow`, loaded through the `petrify.modes` entry point.
2. **Preview transport.** Decided: MJPEG first. No WebRTC.
3. **Virtual camera on macOS.** Decided: `pyvirtualcam`. Requiring OBS 30+ on macOS is acceptable, as a dependency of the virtual camera sink only.

## Compliance notes (not legal advice)

- GDPR does not prohibit faces on camera. CCTV footage is personal data with or without blurring and needs a lawful basis either way.
- What is restricted is biometric identification (GDPR Art. 9, and the EU AI Act ban on real-time remote biometric identification in public spaces). The AI Act also bans emotion recognition in workplaces and schools, so any "suspicious activity" analytics need their own review.
- Face detection used only to obfuscate is generally not identification, and supports data minimisation and privacy by design (GDPR Art. 5 and 25).
- This only holds if obfuscation happens on the device before storage. If the raw stream is stored anywhere, the benefit is largely lost.
