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

It uses [OpenCV's MOG2](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) background subtractor, a [HOG person detector](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html), and a static background differ. Three complementary signals that together catch every subject: one moving fast, one standing still, one barely visible at the edge of the frame.

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

## How it works: three-detector pipeline

petrify uses three complementary signals that together cover every failure mode:

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
         ├── Lookahead buffer: output delayed by --gap-fill frames
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
| Storage | Source file size × 2 | Temporary intermediate video file written before the FFmpeg re-encode |

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
- The bottleneck for large resolutions is OpenCV's `VideoWriter` (intermediate file I/O). Use a fast SSD or a RAM-backed `/tmp`.

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
