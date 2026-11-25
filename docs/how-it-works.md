# How it works: three-detector pipeline

Back to [README](../README.md)

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
         ├── ROI overlay: live timestamp pasted from the real frame on every frame
         │            (erase allows it; a mode that does not allow the overlay
         │            skips this stage), so the timestamp keeps ticking
         │            regardless of freeze state
         │
         └── File sink → intermediate file → FFmpeg re-encode + audio mux
                                                     ↓
                                                output.mp4
```

## Why three detectors?

| Detector | Catches | Misses |
|---|---|---|
| **MOG2** (adaptive BG) | Fast-moving subjects | Subject who stands still → model adapts them in |
| **HOG** (person shape) | Stopped subjects with visible full body | Partial bodies, small silhouettes, unusual poses |
| **BackgroundDiffer** (static diff) | *Everything* different from the empty-corridor reference: moving or still, full or partial, near or far | Nothing, as long as the background frame is clean |

The static background differ is the safety net. Because its reference never adapts, a person standing perfectly still for 60 seconds is still detected on every single frame. They are simply always different from the known-empty scene.

In `petrify run`, `--person-detector dnn` puts the NanoDet-Plus person detector in the HOG slot. It is slower and misses small, distant people. See [Model-based detectors](run.md#model-based-detectors).
