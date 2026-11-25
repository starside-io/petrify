# Before you rely on it

Back to [README](../README.md)

Known limitations, in particular for live use:

1. **Live paths are unverified on real hardware.** Webcams (`camera:N`), stream URLs, `--virtualcam`, and `--preview` were developed and tested against test doubles. They have not been verified with real cameras, streams, virtual camera drivers, or windows.
2. **An online background can absorb a person.** With `--background running` or `calibrate`, a person in view while the background is learned can be absorbed into it and then shown on frozen frames after they leave. The person check on background samples reduces this, but by default it uses the HOG pedestrian detector, which often misses a seated person framed from the chest up (the usual webcam view). `--background-check` can add the face and DNN person checks. Start with the scene empty. Do not rely on erase mode on a webcam for privacy yet.
3. **The model-based detectors have measured limits.** The DNN person detector misses people smaller than about 110 px tall on a 1080p frame and replaces HOG rather than adding to it. The segmenter misses small, distant people and marks parts of empty scenes. No built-in mode uses faces or a mask; the `shadow` plugin does, and its [README](../plugins/petrify-shadow/README.md) lists its own measured limits. See [Limits, before you choose one](run.md#limits-before-you-choose-one).
4. **A plugin mode is trusted code.** It runs in the petrify process and receives the unprocessed frames. petrify does not sandbox or vet it.
5. **The timestamp ROI is shown live.** Anything inside the ROI region is not anonymized, by design.
6. **A slow entry at the frame edge can show through.** A subject entering slowly at the edge of the frame can be partly visible for a few frames before it is detected, more so with a small `--lookahead` (the `camera:N` default is 3).
7. **A lasting lighting change freezes an online background.** With `running` or `calibrate` and the background differ on (the default), a lasting change in lighting makes every frame differ from the learned background, so the output freezes and keeps showing the old background.
8. **Golden parity tests need demo clips that are not in the repository.** Without `data/camera.mp4` and `data/camera2.mp4` they are skipped.
9. **The shadow mode can show distant people.** It does not use the background differ, and the DNN person detector found no person smaller than about 110 px tall on a 1080p frame. Running the DNN on tiles of the frame was tried and not kept: it cost 3.3 times as much and still found nobody under 110 px. A distant person that no detector localizes is shown whenever the frame has another region to draw (see the [plugin README](../plugins/petrify-shadow/README.md#coverage-rules)).
10. **Every detector runs on every processed frame.** There is no option to detect on every Nth frame only. When the detectors are slower than a live source, frames are dropped (see [Performance](usage.md#performance)), so the output frame rate falls.
11. **A hung detector stalls the output.** `--frame-budget-ms` is checked after a detector call returns, and `--source-timeout` watches the source only, so a detector call that never returns stops the output until the process is stopped.
12. **A stop can truncate the older form's output.** `petrify INPUT OUTPUT` writes its final encode directly to `OUTPUT`, so a stop during that encode can leave a truncated file there. `petrify run --output` writes to a partial file and moves it into place only when the encode succeeds.
