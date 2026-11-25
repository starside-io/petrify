# Usage: `petrify INPUT OUTPUT`

Back to [README](../README.md)

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

## CLI reference

```
usage: petrify [options] INPUT OUTPUT
```

This section lists the options of the `petrify INPUT OUTPUT` form. `petrify run` accepts the same ROI, detector, background, motion, and encoding options except `--gap-fill` (use `--lookahead`), plus the options described in [`petrify run`](run.md). In the run form `--hwaccel` does not speed anything up, because petrify decodes the source itself; choose a hardware encoder with `--codec` instead.

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
| 1080p | 0.5 to 1x real-time | Real-time | Real-time |
| 4K | Needs GPU | Recommended | Recommended |

- Use `--downsample 0.25` for 4K to reduce MOG2 cost.
- With `petrify INPUT OUTPUT`, the bottleneck for large resolutions is OpenCV's `VideoWriter` (intermediate file I/O). Use a fast SSD or a RAM-backed `/tmp`.

### Detector cost in `petrify run`

Median time per frame of each detector, measured on the two demo clips on one development machine while it was idle. Another machine, another camera, or a loaded machine gives other numbers.

| Detector | 640x480 | 1920x1080 |
|---|---|---|
| DNN person boxes (`--person-detector dnn`) | about 33 ms | about 45 ms |
| YuNet faces (`--faces`) | about 7 ms | about 29 ms |
| PP-HumanSeg mask (`--segment`) | about 10 ms | about 15 ms |
| MOG2 motion | 1 to 2 ms | 1 to 2 ms |

A 25 fps source leaves 40 ms per frame for every detector together, plus rendering and writing. A `shadow` run with the DNN person detector and faces at 1080p ran at about 12 to 13 frames per second. For several processes on one machine, see [Capacity](deployment.md#capacity).

The DNN person detector resizes a large frame for the model one axis at a time, which gives the same bytes as a single resize and saved about 9.5 ms per 1080p frame. It is used only for the frame sizes where the two were shown to be byte-identical; every other size keeps the single resize.

To see where the time goes on your own setup, the web page lists the time of each detector and refiner, and every `petrify run` logs one line after the summary, in this form (each `N` a time to one decimal):

```
[INFO] petrify.pipeline: Detector and refiner times (moving average): MotionDetector N ms, PersonDnnDetector N ms, FaceYuNetDetector N ms, TrackerRefiner (persons) N ms, TrackerRefiner (faces) N ms
```

Each label is the class name, with the detection kind for the two tracker refiners, in run order; a component never sampled shows `n/a`. The times are moving averages that weight recent frames most, so on a long run they describe its end. The line is left out when the run had no detector or refiner.
