# petrify — Usage Examples

This directory contains example commands and notes for common petrify use cases.

---

## Basic motion freeze (no timestamp overlay)

Freeze the background whenever motion exceeds 500 px² of foreground area
(the default threshold):

```bash
petrify input.mp4 output.mp4 --no-roi
```

Without `--no-roi`, petrify auto-detects the timestamp region and keeps it live.

---

## Timestamp ROI overlay

Keep a 320 × 80 px region in the upper-left corner live while freezing the
rest of the frame on motion.  The `--roi` argument takes `X Y W H` in pixels
measured from the top-left of the frame.

```bash
petrify input.mp4 output.mp4 --roi 10 10 320 80
```

If you don't know the exact coordinates of your timestamp, use FFmpeg to grab
a reference frame and inspect it in any image editor:

```bash
ffmpeg -i input.mp4 -vframes 1 frame.png
```

---

## Tuning motion sensitivity

**Lower threshold** = freezes more easily (detects smaller movements):

```bash
petrify input.mp4 output.mp4 --threshold 150
```

**Higher threshold** = only freezes on large, obvious motion:

```bash
petrify input.mp4 output.mp4 --threshold 2000
```

Increase `--history` for a more stable background model in scenes with
gradual lighting changes (e.g. sunrise/sunset timelapses):

```bash
petrify input.mp4 output.mp4 --history 500 --threshold 300
```

Increase `--var-threshold` to ignore minor pixel-level noise:

```bash
petrify input.mp4 output.mp4 --var-threshold 32
```

---

## Processing at half resolution (faster)

The default `--downsample 0.5` already halves the frame before motion
detection.  For 4K sources you may want `0.25`:

```bash
petrify 4k_input.mp4 output.mp4 --downsample 0.25 --threshold 200
```

Note: `--threshold` is evaluated in the downsampled coordinate space, so a
lower absolute value is needed when downsampling more aggressively.

---

## High-quality output

```bash
petrify input.mp4 output.mp4 --crf 18 --preset slow
```

---

## HEVC / H.265 output

```bash
petrify input.mp4 output_hevc.mp4 --codec libx265 --crf 24 --preset medium
```

---

## Hardware-accelerated encoding

### NVIDIA NVENC (Linux / Windows)

```bash
petrify input.mp4 output.mp4 --hwaccel cuda --codec h264_nvenc
```

### Apple Silicon / VideoToolbox (macOS)

```bash
petrify input.mp4 output.mp4 --hwaccel videotoolbox --codec h264_videotoolbox
```

### Intel Quick Sync (Linux)

```bash
petrify input.mp4 output.mp4 --hwaccel qsv --codec h264_qsv
```

---

## Debugging

Enable verbose logging to see per-frame detector decisions:

```bash
petrify input.mp4 output.mp4 --verbose 2>petrify.log
```

The log will show:
- `is_motion: total_contour_area=... threshold=... -> True` (MOG2 decision for a frame)
- `BackgroundDiffer: area=... threshold=... -> True` (static background differ decision)
- `PersonDetector: person detected box=(...) confidence=... area=...` or `PersonDetector: no detections` (HOG)
- `Engine done: frozen=... pass-through=...` at the end of the run
- The FFmpeg command line, and FFmpeg's own output (its exit code is logged if it fails)

---

## `petrify run`: other sources and outputs

`petrify run --source SPEC` reads a video file, an image folder or glob, a
webcam (`camera:N`), or an RTSP or HTTP stream, and writes to one or more
outputs. See `petrify run --help` and the README for the options and the
defaults per source kind.

Note: the live paths (`camera:N`, stream URLs, `--virtualcam`, `--preview`)
have not been verified on real hardware yet. See "Before you rely on it" in
the README.

### Video file

```bash
petrify run --source input.mp4 --output output.mp4
```

Processes the same frames as `petrify input.mp4 output.mp4`, but FFmpeg
encodes them in one pass through a pipe instead of from an intermediate file.

### Image folder

```bash
petrify run --source 'frames/*.png' --fps 10 --output out.mp4
```

Reads the images in natural name order (`frame2.png` before `frame10.png`) as
a 10 fps video; the timestamp ROI is detected on the first 10 seconds, then locked.

### Webcam to a virtual camera

```bash
petrify run --source camera:0 --virtualcam --preview
```

Needs `pip install 'petrify[virtualcam]'` (and OBS 30 or newer on macOS) and,
for `--preview`, an OpenCV build with GUI support. Frames are black until the
running background is ready; start with the scene empty. Ctrl-C, or `q` in
the preview window, ends the run.

### IP camera stream to a file

```bash
petrify run --source rtsp://camera.local/stream --output out.mp4 \
    --background calibrate --calibrate-seconds 5
```

The first seconds are black while the background is calibrated: stay out of
view. Ctrl-C ends the run and `out.mp4` is still finalized. The URL is shown
redacted in the log.

### Watch and tune on the local web page

```bash
petrify run --source input.mp4 --output output.mp4 --web --realtime
```

Needs `pip install 'petrify[web]'`. Prints a `http://127.0.0.1:8765/#token=...`
link: open that exact link (not `localhost`), and do not share it, since it
gives control of the run. `--realtime` plays the file at its own frame rate.

---

## Docker / cloud batch processing

```bash
docker run --rm \
  -v /path/to/videos:/data \
  petrify:latest \
  /data/input.mp4 /data/output.mp4 \
  --roi 10 10 320 80 \
  --threshold 300 \
  --crf 20 \
  --preset fast
```
