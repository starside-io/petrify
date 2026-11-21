<p align="center">
  <img src="docs/logo.svg" alt="petrify" width="320" />
</p>

# petrify

[![GitHub](https://img.shields.io/badge/GitHub-starside--io%2Fpetrify-181717?logo=github)](https://github.com/starside-io/petrify)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/starside-io/petrify)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/powered%20by-OpenCV-5C3EE8?logo=opencv)](https://opencv.org)
[![FFmpeg](https://img.shields.io/badge/powered%20by-FFmpeg-007808?logo=ffmpeg)](https://ffmpeg.org)

**Freeze the world. Keep the clock ticking.**

Built by [starside.io](https://starside.io). We build tools at the intersection of computer vision and human experience.

---

### Why *petrify*?

Security footage is relentlessly honest. It records everything: every arrival, every departure, every face crossing a threshold it shouldn't. But that honesty is exactly what makes it sensitive.

**petrify** takes a different approach. The moment a person enters the scene, the world freezes into a perfect, clean still of the empty space. Only the burned-in clock keeps running, composited live onto every frozen frame. The scene says *nothing happened here*. The timestamp says *time kept moving.*

`petrify` comes from the idea of turning a scene to stone: immovable, unchanging, as if the world held its breath. The motion is real. The stillness is manufactured. Only the time is true.

It uses [OpenCV's MOG2](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) background subtractor, a [HOG person detector](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html), and a static background differ. Three complementary signals, each aimed at a different kind of subject: one moving fast, one standing still, one barely visible at the edge of the frame.

---

When a security camera records someone entering a building, does the output need to show *who* was there? **petrify** answers that question with a short pre-scan and a single processing pass. The pre-scan picks the clean background. The processing pass composites every frame so only the timestamp survives.

---

## Demo

**Left: original footage &nbsp;·&nbsp; Right: petrify output**

![Driveway camera. Person frozen out, timestamp ticking](docs/demo_camera1.gif)
*Driveway camera · person enters from street · petrify shows only the empty driveway with the live clock*

![School corridor. Standing person frozen, timestamp preserved](docs/demo_camera2.gif)
*School corridor · person walks and lingers · petrify holds the clean background while the timestamp keeps running*

---

## Use cases

| Scenario | What petrify removes | What survives |
|---|---|---|
| **Privacy compliance (PDPA / GDPR)** | Identifiable individuals in public spaces | Scene context · verified timeline |
| **Footage compression** | Sparse motion (bird, leaf, flag) | Static background · motion-free stream |
| **Space & layout analysis** | People occluding floor plans | Clean geometry for heatmaps / CAD |
| **Physical hazard detection** | Human presence masking static hazards | Exposed obstacles, spills, trip points |
| **AI training data** | Subject identity from annotated scenes | Background environment · ground truth |
| **Any fixed camera** | All foreground subjects | Clean background · live timestamp |

---

> **Disclaimer:** petrify is provided for educational and research purposes. Always ensure you have the legal right to process any footage you use, including compliance with applicable privacy laws (PDPA, GDPR, CCPA, and equivalents). Do not use petrify to process footage of individuals without proper authorisation, consent, or a lawful basis.

---

## Installation

### 1. System FFmpeg (required)

petrify calls `ffmpeg` via subprocess. Install it first:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows: download from https://ffmpeg.org/download.html and add to PATH
```

### 2. petrify

The recommended install method is [pipx](https://pipx.pypa.io). It puts `petrify` on your PATH without touching your system Python:

```bash
# Install pipx if you don't have it
brew install pipx        # macOS
# pip install --user pipx  # Linux / Windows

# Install petrify
pipx install petrify
```

Or install from source:

```bash
git clone https://github.com/starside-io/petrify.git
cd petrify
pipx install -e .
```

Or with plain pip into a virtualenv:

```bash
pip install petrify
```

### 3. Optional extras (for `petrify run`)

Some outputs of [`petrify run`](#petrify-run-cameras-streams-and-image-folders) need extra packages. The `petrify INPUT OUTPUT` form needs none of them.

| Output | Install | Notes |
|---|---|---|
| `--web` (local web page) | `pip install 'petrify[web]'` | Installs FastAPI and uvicorn |
| `--virtualcam` (virtual camera) | `pip install 'petrify[virtualcam]'` | Installs pyvirtualcam. On macOS it also needs OBS 30 or newer, whose virtual camera is the device pyvirtualcam drives |
| `--preview` (local window) | `pip uninstall opencv-python-headless && pip install opencv-python` | petrify depends on the headless OpenCV build, which has no GUI support |

With pipx, add the extras to the install: `pipx install 'petrify[web,virtualcam]'`.

---

## Quick start

```bash
# Zero config. Auto-detects background + timestamp ROI
petrify input.mp4 output.mp4

# Explicit ROI (skip auto-detection for speed)
petrify input.mp4 output.mp4 --roi 0 55 700 60

# More sensitive. Catches slower or more distant subjects
petrify input.mp4 output.mp4 --threshold 200 --bg-diff-area-threshold 100

# Disable timestamp overlay entirely
petrify input.mp4 output.mp4 --no-roi

# Higher quality encode
petrify input.mp4 output.mp4 --crf 18 --preset slow

# Apple Silicon GPU encode
petrify input.mp4 output.mp4 --hwaccel videotoolbox --codec h264_videotoolbox

# NVIDIA GPU encode (requires ffmpeg with NVENC)
petrify input.mp4 output.mp4 --hwaccel cuda --codec h264_nvenc

# Debug. Print every detection decision
petrify input.mp4 output.mp4 --verbose
```

### Example output

`petrify camera2.mp4 output.mp4` on the 640x480, 6 fps corridor clip:

```
██████╗ ███████╗████████╗██████╗ ██╗███████╗██╗   ██╗
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝╚██╗ ██╔╝
██████╔╝█████╗     ██║   ██████╔╝██║█████╗   ╚████╔╝
██╔═══╝ ██╔══╝     ██║   ██╔══██╗██║██╔══╝    ╚██╔╝
██║     ███████╗   ██║   ██║  ██║██║██║        ██║
╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝

  Freeze the world. Keep the clock ticking.
  Made with ❤️  by starside.io  ·  https://starside.io

16:55:00 [INFO] petrify.sources.file: Input: camera2.mp4  |  640x480 @ 6.00 fps  |  ~655 frames
16:55:00 [INFO] petrify.pipeline: Detector mode: motion
16:55:00 [INFO] petrify.roi_detect: Auto ROI scan: diffing 60 frame pairs (gap=6 fr) across 655 frames...
16:55:01 [INFO] petrify.roi_detect: Pass 1 anchor: x=151 y=0 w=8 h=13  freq=0.41  cx_rel=0.24 cy_rel=0.01
16:55:01 [INFO] petrify.roi_detect: Auto ROI detected: x=0 y=0 w=440 h=29
16:55:01 [INFO] petrify.roi_detect:   → To skip auto-detection next time: --roi 0 0 440 29
16:55:01 [INFO] petrify.background.locked: Auto-background scan: checking every 5 frames for a clean frame...
16:55:02 [INFO] petrify.background.locked: Auto background: frame 650  (108.3s into video)
16:55:02 [INFO] petrify.pipeline: BackgroundDiffer enabled: pixel_threshold=25 area_threshold=200
16:55:02 [INFO] petrify.sinks.file: Compositing frames -> /tmp/tmpj6z7l5hi.mp4
16:55:04 [INFO] petrify.sinks.file: Re-encoding with FFmpeg -> output.mp4
16:55:04 [INFO] petrify.encoder: FFmpeg command: /opt/homebrew/bin/ffmpeg -y -i /tmp/tmpj6z7l5hi.mp4 -i camera2.mp4 -map 0:v:0 -map 1:a? -c:v libx264 -crf 23 -preset medium -c:a copy output.mp4
16:55:04 [INFO] petrify.sinks.file: Done. Output written to: output.mp4
Petrify: 100%|██████████| 655/655 [00:02<00:00, 308.86fr/s]
16:55:04 [INFO] petrify.pipeline: Engine done: frozen=494 (75.4%)  pass-through=161 (24.6%)
```

---

## `petrify run`: cameras, streams, and image folders

`petrify INPUT OUTPUT` is unchanged. A second command form, `petrify run --source SPEC`, reads from more kinds of source and writes to one or more outputs in one streaming pass. Run `petrify run --help` for every option.

> The live paths (webcams, stream URLs, the virtual camera, and the preview window) have not been verified on real hardware yet. Read [Before you rely on it](#before-you-rely-on-it) before you depend on them.

```bash
# Same result as: petrify input.mp4 output.mp4
petrify run --source input.mp4 --output output.mp4

# Webcam to a virtual camera for video calls, with a preview window
petrify run --source camera:0 --virtualcam --preview

# IP camera to a file; step out of view for the first 5 seconds
petrify run --source rtsp://camera.local/stream --output out.mp4 \
    --background calibrate --calibrate-seconds 5

# Image folder at 10 fps
petrify run --source 'frames/*.png' --fps 10 --output out.mp4
```

The run form is selected when the first argument that is not an option is `run` and `--source` is given. Otherwise the arguments are `INPUT OUTPUT`, so a file named `run` still works: `petrify run out.mp4`.

### Sources

| `--source` | Reads |
|---|---|
| A file path, such as `input.mp4` | A video file. A bare number such as `0` is a file name, not a camera |
| A directory, or a glob such as `'frames/*.png'` | Still images in natural name order (`frame2.png` before `frame10.png`), at `--fps` frames per second (default 25) |
| `camera:N` | Webcam number N |
| `rtsp://`, `rtsps://`, `http://`, `https://` URL | A network stream (RTSP, HTTP MJPEG) |

On a live source (`camera:N` or a stream URL) a capture thread keeps only the newest frame, so a slow run drops frames instead of falling behind. Stream URLs are redacted in logs and error messages: at most the scheme, host, and port are shown. Credentials are replaced with `***`, and the path and query are dropped.

For library use, `petrify.sources.PushSession` feeds in-memory frames to an engine one by one and returns the processed frames.

### Outputs

At least one is required, and several can be combined.

| Flag | Output |
|---|---|
| `--output PATH` | A video file, encoded by FFmpeg in one pass through a pipe (no intermediate file). FFmpeg writes to `<name>.partial<suffix>`, which is moved onto `PATH` only when the encode succeeds, so a failed run never leaves a file at `PATH` and never replaces an existing one. For a video file source its audio is copied |
| `--virtualcam` | A virtual camera that video call apps can select. `--virtualcam-device NAME` picks the device (default: the first one). Needs `petrify[virtualcam]`, and OBS 30 or newer on macOS |
| `--preview` | A local window. `q` or Escape ends the run. Needs an OpenCV build with GUI support |
| `--web` | A local web page to watch the processed feed and change settings while the run is in progress. Needs `petrify[web]`. See [The web page](#the-web-page) |

On a live source the run ends with Ctrl-C (or `q` in the preview window), and the output file is still finalized. A second Ctrl-C while the outputs are being finalized aborts. If a live run fails, the recording up to that point is kept and the error is logged.

### Defaults by source kind

Some defaults depend on the source kind. Every value can be overridden with the flag named in the column.

| Source | `--background` | `--lookahead` | Timestamp ROI |
|---|---|---|---|
| Video file | `locked` | 10 | Auto-detected on the whole clip before processing |
| Image folder or glob | `locked` | 10 | Auto-detected on the first `--roi-warmup` seconds (default 10), then locked |
| Stream URL | `running` | 10 | Auto-detected on the first `--roi-warmup` seconds (default 10), then locked |
| `camera:N` | `running` | 3 | Disabled. `--roi-auto` turns warmup detection on |

`--lookahead` is the run form's name for `--gap-fill`: every frame is delayed by this many frames, and every detection is expanded by as many frames on both sides. A smaller value lowers the latency but lets more of a slowly entering subject through before it is detected. It is capped at 600, because the buffered frames are kept uncompressed. `--gap-fill` itself is not accepted by `petrify run`.

During warmup ROI detection nothing is pasted until the region locks. A detected box larger than `--roi-max-area` of the frame (default 0.15) is discarded as too large to be a timestamp, so nothing is pasted. A webcam has no burned-in timestamp and detection there can lock onto a person, which is why the ROI is off for `camera:N` unless `--roi` or `--roi-auto` is given. A manual `--roi` larger than that cap is accepted, with a warning that the region is shown live.

### Background strategies

| `--background` | How the clean background is found |
|---|---|
| `locked` | Pre-scan the source for one clean frame (video files and image folders only). `--background-frame N` picks the frame |
| `running` | Rolling median of recent frames with no subject |
| `calibrate` | Median of the first `--calibrate-seconds` (default 3), while the scene is empty |

Frames are written black until the background is ready. With `running` or `calibrate`, every frame the background is learned from is first checked with the HOG person detector, and a frame in which it finds a person is not used. `--no-background-check` turns that check off and is unsafe. At startup petrify logs a warning that a person in view while the background is learned can be absorbed into it. The check reduces this but does not eliminate it: start with the scene empty either way. See [Before you rely on it](#before-you-rely-on-it).

### Fail closed

A frame that cannot be processed safely is written black and counted: a detector error, the background not ready yet, or a frame over `--frame-budget-ms` (the per-frame time budget for detection plus rendering; no budget by default). The end-of-run summary reports how many frames failed closed.

### The web page

`--web` serves a page on `127.0.0.1` at `--web-port` (default 8765; 0 picks a free port). At start petrify prints the link to the terminal:

```
Web page (this link gives control of this run to whoever has it): http://127.0.0.1:PORT/#token=...
```

The page shows the processed feed (what the outputs receive), the run's status (source, background state, frame counters, frames failed closed and dropped, output frame rate, detect, render, and write timings, the ROI in use, the last error), a settings form, and a button that stops the run and finalizes the outputs. While the run is in progress the page can change:

- the detector (`motion`, `person`, `both`)
- the motion detector's area and variance thresholds and shadow detection
- the person detector's minimum confidence, minimum box area, and pyramid scale
- the background differ (on or off) and its pixel and area thresholds
- the lookahead and the frame budget
- the ROI: disabled, or a manual box no larger than the run's `--roi-max-area` of the frame

The values are bounded more tightly than on the command line, so every detector can still fire. The background strategy, the source, the outputs, the mode, the encoding, the MOG2 history, and the downsample factors are fixed for the run. Changes are applied at the next frame boundary, and each one briefly freezes the output: the frames buffered at that moment and at least one lookahead window after them show the background. A change of only the frame budget freezes nothing.

A "Weakened protections" list on the page names every setting in effect that can let a person be shown: the background differ off, the person detector alone with the differ off, a detector threshold more than twice its startup value, a lookahead below its startup value or 0, a manual ROI over 5 percent of the frame, and the raw preview. Each newly weakened protection is also logged as a warning.

`--realtime` processes a video file or image folder no faster than its frame rate, so it can be watched on the page. It changes no output frame and is ignored for live sources.

Security model:

- The server listens on `127.0.0.1` only, and every request must carry a `Host` of exactly `127.0.0.1:PORT`. Use the printed `http://127.0.0.1:PORT/` link: `localhost` is refused on purpose, because the server does not listen on `[::1]`, where another local user could.
- The link holds a per-run token. Whoever has the link controls the run, so do not share it. The token is never logged, and the server's access log is off.
- The unprocessed input feed is never served unless the run was started with `--web-raw-preview`, which shows it next to the processed feed and lists it as a weakened protection.

---

## Before you rely on it

Known limitations, in particular for live use:

1. **Live paths are unverified on real hardware.** Webcams (`camera:N`), stream URLs, `--virtualcam`, and `--preview` were developed and tested against test doubles. They have not been verified with real cameras, streams, virtual camera drivers, or windows.
2. **An online background can absorb a person.** With `--background running` or `calibrate`, a person in view while the background is learned can be absorbed into it and then shown on frozen frames after they leave. The person check on background samples reduces this, but it uses the HOG pedestrian detector, which often misses a seated person framed from the chest up (the usual webcam view). Start with the scene empty. Do not rely on erase mode on a webcam for privacy yet; stronger detectors are planned.
3. **The timestamp ROI is shown live.** Anything inside the ROI region is not anonymized, by design.
4. **A slow entry at the frame edge can show through.** A subject entering slowly at the edge of the frame can be partly visible for a few frames before it is detected, more so with a small `--lookahead` (the `camera:N` default is 3).
5. **A lasting lighting change freezes an online background.** With `running` or `calibrate` and the background differ on (the default), a lasting change in lighting makes every frame differ from the learned background, so the output freezes and keeps showing the old background.
6. **Golden parity tests need demo clips that are not in the repository.** Without `data/camera.mp4` and `data/camera2.mp4` they are skipped.

---

## How it works: three-detector pipeline

Both command forms run the same streaming engine. `petrify INPUT OUTPUT` writes an intermediate file and re-encodes it with FFmpeg, as shown below; `petrify run --output` pipes frames to FFmpeg in one pass instead. petrify uses three complementary signals, each aimed at a different kind of subject:

```
input video
    │
    ├── Pre-scan: Auto ROI detection (frame-diff + Laplacian edges)
    │          Finds the burned-in timestamp region automatically.
    │
    ├── Pre-scan: Auto background selection (HOG scan → latest clean frame)
    │          Locks the reference background before any processing.
    │
    └── Single streaming pass, frame by frame
         │
         ├── Detectors (in order; MOG2 always runs, the others stop at the first hit)
         │    ├── MOG2 (motion)        catches moving subjects
         │    ├── HOG  (person)        catches stopped subjects MOG2 adapted away
         │    │                        (only with --detector person or both)
         │    └── BackgroundDiffer     catches tiny / partial / edge subjects
         │              ↓                both detectors miss
         │         subject present = True / False
         │
         ├── Lookahead buffer: output delayed by --gap-fill frames (--lookahead in petrify run)
         │         gap-fill ±10 frames  (bridges missed detections mid-sequence)
         │         cold-start freeze    (first 50 frames, MOG2 warmup window)
         │
         ├── Erase: locked clean frame when a subject is present, live frame otherwise
         │
         ├── ROI overlay: live timestamp pasted from the real frame (always)
         │            timestamp keeps ticking regardless of freeze state
         │
         └── File sink → intermediate file → FFmpeg re-encode + audio mux
                                                     ↓
                                                output.mp4
```

### Why three detectors?

| Detector | Catches | Misses |
|---|---|---|
| **MOG2** (adaptive BG) | Fast-moving subjects | Subject who stands still → model adapts them in |
| **HOG** (person shape) | Stopped subjects with visible full body | Partial bodies, small silhouettes, unusual poses |
| **BackgroundDiffer** (static diff) | *Everything* different from the empty-corridor reference: moving or still, full or partial, near or far | Nothing, as long as the background frame is clean |

The static background differ is the safety net. Because its reference never adapts, a person standing perfectly still for 60 seconds is still detected on every single frame. They are simply always different from the known-empty scene.

---

## Auto-detection

### Timestamp ROI

When `--roi` is not specified, petrify automatically finds the timestamp rectangle using a two-pass approach:

1. **Frequency pass.** Sample 60 frame pairs spaced 1 second apart. Find pixels that change in at least 60% of pairs (the seconds digits).
2. **Edge pass.** Run Laplacian edge detection in the detected Y strip. Walk column-by-column from the nearest frame corner and stop at the first gap greater than 30 px. Captures the full text width including static hours and camera labels.

The detected ROI is printed at startup so you can hard-code it on future runs to skip the ~10-second scan:

```
→ To skip auto-detection next time: --roi 0 0 642 129
```

### Background frame

petrify scans every 5th frame with the HOG person detector, collects all clean frames (no person detected), and locks the **latest** one as the reference background. For typical security footage the subject leaves at the end of the clip, so the cleanest frame is usually in the final few seconds. Use `--background-frame N` to override.

---

## CLI reference

```
usage: petrify [options] INPUT OUTPUT
```

This section lists the options of the `petrify INPUT OUTPUT` form. `petrify run` accepts the same ROI, detector, background, motion, and encoding options except `--gap-fill` (use `--lookahead`), plus the options described in [`petrify run`](#petrify-run-cameras-streams-and-image-folders). In the run form `--hwaccel` does not speed anything up, because petrify decodes the source itself; choose a hardware encoder with `--codec` instead.

### Positional

| Argument | Description |
|---|---|
| `INPUT` | Path to the input video |
| `OUTPUT` | Path for the output video |

### ROI / Compositor

| Flag | Default | Description |
|---|---|---|
| `--roi X Y W H` | auto | Bounding box of the live timestamp region (px, top-left origin). Auto-detected if omitted. |
| `--no-roi` | off | Disable timestamp overlay and skip auto-detection entirely. |
| `--roi-scan-samples N` | 60 | Frame pairs to diff during auto ROI detection. |
| `--roi-scan-min-freq F` | 0.6 | Min change-frequency for a pixel to be part of the anchor. |
| `--roi-scan-edge-threshold F` | 80 | Laplacian column energy threshold for the ROI edge pass (text vs. background). |
| `--roi-scan-edge-gap-tolerance PX` | 30 | Max inter-character gap before the edge scan stops. |

### Background / Reference

| Flag | Default | Description |
|---|---|---|
| `--background-frame N` | auto | Frame index to use as the locked reference. Auto-scanned if omitted. |
| `--no-bg-diff` | off | Disable the static background differ (reduces recall). |
| `--bg-diff-pixel-threshold INT` | 25 | Per-pixel grayscale difference (0 to 255) above which a pixel is counted as changed. |
| `--bg-diff-area-threshold PX` | 200 | Changed-pixel area (px² in downsampled space) to fire the differ. |

### Detector mode

| Flag | Default | Description |
|---|---|---|
| `--detector {motion,person,both}` | motion | Freeze trigger: MOG2 (`motion`), HOG (`person`), or either (`both`). The background differ runs in every mode unless `--no-bg-diff` is set. |
| `--hog-min-confidence F` | 0.3 | HOG detection confidence floor. Raise to 1.2 if false positives on architectural structures. |
| `--hog-min-box-area PX` | 1500 | Minimum HOG detection bounding-box area. |

### Motion (MOG2)

| Flag | Default | Description |
|---|---|---|
| `--threshold PX` | 500 | Foreground contour area (downsampled px²) to classify a frame as motion. Lower = more sensitive. |
| `--history N` | 200 | MOG2 background model window (frames). |
| `--gap-fill N` | 10 | Expand every detected region ±N frames to bridge gaps. Output is delayed by N frames. Must be 0 or more. |
| `--downsample F` | 0.5 | Resize factor before detection. 0.5 = 4× speedup. |

### Encoding

| Flag | Default | Description |
|---|---|---|
| `--codec CODEC` | `libx264` | FFmpeg video codec. |
| `--crf INT` | 23 | Constant Rate Factor (lower = better quality). |
| `--preset PRESET` | `medium` | Encoding speed/quality tradeoff. |
| `--hwaccel ACCEL` | none | FFmpeg `-hwaccel` for hardware-accelerated decoding. |
| `--verbose` | off | Enable DEBUG-level log output. |

---

## System requirements

| Dependency | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | Minimum supported version (`requires-python >= 3.10`) |
| opencv-python-headless | 4.8 | MOG2, HOG, Laplacian edge detection |
| numpy | 1.24 | Frame array operations |
| tqdm | 4.65 | Progress bars |
| FFmpeg | Any recent | System binary. `brew install ffmpeg` / `apt install ffmpeg` |
| RAM | 2 GB | 4 GB recommended for 1080p |
| Storage | Source file size × 2 | `petrify INPUT OUTPUT` writes a temporary intermediate video file before the FFmpeg re-encode; `petrify run --output` does not |

No GPU required. All detection runs on CPU via OpenCV.

---

## Performance

### Benchmarks

Measured on **MacBook Pro · Apple M2 Pro · 12 cores (8P + 4E) · 16 GB RAM · macOS 26.3**.

| Video | Resolution | FPS | Duration | Frames | Wall time | Speed |
|---|---|---|---|---|---|---|
| camera.mp4 | 1920x1080 | 25 | 46.8s | 1 169 | **45.7s** | 1.0x real-time |
| camera2.mp4 | 640x480 | 6 | 109.2s | 655 | **4.95s** | 22x real-time |

Speed scales with resolution and source frame rate. Low-FPS cameras (6fps) process in a fraction of real-time even at longer durations. High-FPS 1080p tracks close to real-time on Apple Silicon.

### General guidance

| Resolution | CPU (libx264 medium) | Apple M-series | NVIDIA NVENC |
|---|---|---|---|
| 480p | Real-time | Real-time | Real-time |
| 720p | Real-time | Real-time | Real-time |
| 1080p | 0.5–1x real-time | Real-time | Real-time |
| 4K | Needs GPU | Recommended | Recommended |

- Use `--downsample 0.25` for 4K to reduce MOG2 cost.
- With `petrify INPUT OUTPUT`, the bottleneck for large resolutions is OpenCV's `VideoWriter` (intermediate file I/O). Use a fast SSD or a RAM-backed `/tmp`.

---

## Docker

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install petrify
ENTRYPOINT ["petrify"]
```

```bash
docker run --rm \
  -v "$(pwd)/data:/data" \
  petrify \
  /data/input.mp4 /data/output.mp4
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment, run the test suite, and open a pull request.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

Made with ❤️ by [starside.io](https://starside.io) · [github.com/starside-io/petrify](https://github.com/starside-io/petrify)
