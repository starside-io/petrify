# petrify — Usage Examples

This directory contains example commands and notes for common petrify use cases.

---

## Basic motion freeze (no timestamp overlay)

Freeze the background whenever motion exceeds 500 px² of foreground area
(the default threshold):

```bash
petrify input.mp4 output.mp4
```

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

Enable verbose logging to see per-frame motion decisions:

```bash
petrify input.mp4 output.mp4 --verbose 2>petrify.log
```

The log will show:
- `Frame: MOTION  -> using frozen still_frame` — background was frozen
- `Frame: STILL   -> updated still_frame` — new clean frame captured
- FFmpeg command line and exit status

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
