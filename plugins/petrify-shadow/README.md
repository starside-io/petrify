# petrify-shadow

A [petrify](../../README.md) mode plugin that registers the mode `shadow`. Where the built-in `erase` mode freezes the subject out of the scene, `shadow` keeps the live frame and covers the people in it: their faces (blur, pixelation, noise, a solid box, or an eye bar), their bodies (a filled silhouette or a filled person box), or both. While no subject is present the frame passes through unchanged.

The mode draws on regions that the model-based detectors find (face boxes, person boxes, a person mask). Each detector misses people, and the weaker styles can be partly undone. Read [Limits, before you rely on it](#limits-before-you-rely-on-it) before you depend on this mode for privacy.

> **A mode plugin is ordinary Python code.** `petrify-shadow` runs in the petrify process with petrify's permissions and receives the unprocessed frames, people included. When it is the selected mode, the run logs a warning that names the package and its version.

## Install

The plugin lives in this repository and is not published. From the repository root, install it into the environment that has petrify:

```bash
python -m pip install --no-deps -e plugins/petrify-shadow
petrify run --list-modes
```

The listing then shows `shadow (plugin petrify-shadow 0.1.0)`. `petrify run --list-modes --mode shadow` prints its options. `--no-deps` keeps pip from reinstalling petrify; the plugin's other dependency, `numpy`, already comes with petrify.

The styles need the model-based detectors and their model files. petrify never downloads a model: see [Model files](../../README.md#model-files) for the three files and where petrify looks for them.

## Styles

Select one or more with `--mode-option styles=NAME,NAME`. The default is `facebox`.

| Style | What it draws | Needs | Can the cover be undone? |
|---|---|---|---|
| `gaussian` | Blurs the padded face box. The blur sigma is 0.20, 0.10, or 0.05 of the box's longer side for `strength` high, medium, or low | Faces: `--faces` | Partly. See the measured leakage under [Limits](#limits-before-you-rely-on-it) |
| `pixelate` | Replaces the padded face box with blocks of their mean color: about 6, 10, or 16 blocks across the longer side (high, medium, low), at least 4 px each, on a grid anchored to the frame | Faces: `--faces` | Partly, and averaging the frames of a moving face recovers detail |
| `noise` | Replaces the padded face box with new random pixels on every frame | Faces: `--faces` | No: the output does not depend on the input |
| `facebox` | Fills the padded face box with `color` | Faces: `--faces` | No: the output does not depend on the input |
| `eyebar` | A solid bar of `color` over the eyes, placed and rotated from the two eye landmarks. Where the landmarks cannot place a bar, the whole padded face box is filled | Faces and landmarks: `--faces` | The bar cannot be undone, but the rest of the face stays visible |
| `silhouette` | Fills the dilated person mask in black, whatever `color` says | Mask: `--segment` | No: the output does not depend on the input |
| `silhouette_color` | Fills the dilated person mask with `color` (a solid color, no blur) | Mask: `--segment` | No: the output does not depend on the input |
| `outline` | Draws the inner edge of the dilated mask in `color`, `max(2, round(0.004 * frame height))` px wide. Refused on its own, since it hides no one | Mask: `--segment` | It covers only the edge: the person inside stays visible |
| `bodybox` | Fills every person box, padded by 5 percent of its size on each side, with `color` | Person boxes: `--detector person` or `both`, recommended with `--person-detector dnn` | No: the output does not depend on the input |

A style's detector is mandatory: without `--faces` or `--segment` the run stops at startup with the error described in the main README, and `bodybox` with `--detector motion` stops it with `the bodybox style needs person boxes: use --detector person or both, with --person-detector dnn for the recommended detector`. An empty style list stops it with `the shadow mode needs at least one style in 'styles'`.

Besides what the styles need, the mode always asks for motion and person boxes as presence signals. They are optional: only the ones `--detector` builds run, so `--detector motion` gives motion only, `person` gives person boxes only, and `both` gives both. The mode does not use the clean background, so the background differ does not run for it and is not a presence signal.

## Options

All five can be changed while a run is in progress, from the web page.

| Option | Kind | Default | Range or choices | Meaning |
|---|---|---|---|---|
| `styles` | choices | `facebox` | `gaussian`, `pixelate`, `noise`, `facebox`, `eyebar`, `silhouette`, `silhouette_color`, `outline`, `bodybox` | The styles to draw, at least one, comma-separated |
| `padding` | float | 0.3 | 0.0 to 1.0 | Margin around each face box, as a share of its size: `padding` times the box width on the left and right, 1.5 times `padding` times the box height on top (hair), 0.5 times on the bottom (chin), and at least 2 px on each side |
| `strength` | choice | `high` | `high`, `medium`, `low` | Strength of `gaussian` and `pixelate`. Lower values are easier to reverse |
| `color` | choice | `black` | `black`, `white`, `gray`, `red`, `green`, `blue` | Fill of `facebox`, `eyebar`, `bodybox`, `silhouette_color`, and `outline` |
| `dilate` | float | 0.01 | 0.0 to 0.1 | How far the mask styles grow the person mask, as a share of the frame height, rounded up to whole pixels (11 px at 1080p, 5 px at 480p with the default) |

```bash
petrify run --source input.mp4 --output out.mp4 --mode shadow --faces \
    --mode-option styles=pixelate,eyebar --mode-option strength=high
```

`pixelate` without `noise` or `facebox` is listed as a [weakened protection](#weakened-protections), since it can be partly reversed.

## How styles stack

Styles are always drawn in this fixed order, whatever order they were given in, and a later one wins where regions overlap:

1. `silhouette` (black)
2. `silhouette_color`
3. `bodybox`
4. orphan person boxes, filled black (see [Coverage rules](#coverage-rules))
5. `outline`
6. `gaussian`, `pixelate`, `noise`, `facebox`, each over every face before the next starts
7. `eyebar`

So a face cover sits on top of a body fill, and an eye bar sits on top of a face cover. `styles=facebox,bodybox` and `styles=bodybox,facebox` draw the same frame.

## Coverage rules

Every frame gets one of three outcomes:

- **Passed through.** No subject is present: the frame is written unchanged.
- **Black.** A subject is present but the mode has nothing to draw it on. This includes the frames the lookahead fills around a detection and the cold-start frames at the beginning of the run, which carry no detections of their own, and, for example, a face-only setup with `--detector motion` on a frame where motion fired and no face was found.
- **Drawn.** The selected styles are drawn on their regions, and orphan person boxes are filled black.

Which regions decide between black and drawn depends on the styles. When any body style is selected (`silhouette`, `silhouette_color`, `bodybox`), the body styles decide: the frame is drawn when there is a person box (with `bodybox`), or a mask that marks at least 0.1 percent of the frame (with `silhouette` or `silhouette_color`). Otherwise the face styles decide: the frame is drawn when there is a face. An orphan person box also counts. Once a frame is drawn, every selected style draws everything it has, so a face style draws its faces even when the body styles decided.

**Orphan rule.** With person boxes (`--detector person` or `both`), a person box that the deciding styles do not cover is filled black as a whole:

- Face styles decide: each face clears at most one person box, the smallest one whose top third holds the face's center. A person box no face cleared is an orphan. So a person whose face the face detector misses is covered by a black box, as long as the person detector finds them.
- Mask styles decide: a person box is an orphan when the mask covers less than 30 percent of its area, or less than 30 percent of its head zone (its top quarter). So a person the segmenter misses, or segments without the head, is covered by a black box.
- `bodybox` decides: there are no orphans, since `bodybox` fills every person box itself.

**Mask specks.** A mask that marks less than 0.1 percent of the frame's pixels does not count as a region, so a few stray pixels from the segmenter cannot make a frame "drawn" on their own. Such a small mask is still drawn when the frame is drawn for another region.

**Held boxes with the tracker.** With `--track`, the tracker keeps the last box of a lost person or face for `--track-hold` frames (default 15) without moving it. The mode sees the box but cannot be told it is held, so it treats a box exactly equal to one on the previous frame as held:

- its cover grows by 10 percent of its size per held frame, around its center, up to twice its size;
- a held face no longer clears its person box, so that box is filled black as an orphan;
- after 10 held frames, when the growth reaches its cap, the box is still drawn but no longer counts as a region: a frame with nothing else turns black.

A box that truly stays still is also seen as held, which only covers more. A held cover can still miss a subject that moves faster than the cover grows.

**Left uncovered by design.** Once a frame has at least one region, every other presence signal (motion) is taken to belong to it:

- a second person whom no detector localizes is shown;
- when the mask styles decide, a real mask anywhere makes the frame drawn, and a person the segmenter misses is shown unless a person box makes them an orphan;
- when the face styles decide, a false face inside the top third of a person box whose real face was missed clears that box.

A frame on which a detector or the mode fails is written black, as for every mode. The mode turns the live timestamp ROI overlay off, so a face inside the timestamp region is not pasted back: that region is covered like the rest of the frame.

## Limits, before you rely on it

The numbers below were measured on the two demo clips (CCTV-style footage, 1920x1080 and 640x480). They describe those clips, not every camera, and they are not guarantees.

1. **Not tested on live hardware.** The demo footage is CCTV style. The plugin has not been tested on a real webcam or any live source.
2. **gaussian and pixelate leak the face.** With two visibly different real faces, the share of the difference between them that survives the cover was about 25 percent for `gaussian` at high strength, 41 to 46 percent at medium, and 56 to 60 percent at low; for `pixelate`, 50 to 57 percent at high, 61 to 72 percent at medium, and 75 to 83 percent at low. The share is nearly the same for faces from 16 to 400 px. Averaging the pixelated frames of a moving face recovers detail with any pixelation. Use `noise` or `facebox` for a face cover that cannot be undone.
3. **The eyebar leaves most of the face visible.** It covered the eye fissure on all 226 measured faces, but only 66 to 85 percent of the eye and brow region, and the rest of the face stays visible.
4. **The face padding does not cover everything.** With the default 0.3, the padded box covered a median of 94 percent of the head zone on the 640x480 clip (faces of 10 to 44 px, seen from above) and 100 percent on the 1080p clip (faces of 24 to 214 px). Hair, ears, and chin were covered on the contact sheets, except the top of a hair bun on the 640x480 clip, which padding 0.5 does not cover either.
5. **The orphan rule over-covers.** At its thresholds (30 percent of the box and of its head zone), no person box that the segmenter missed escaped on either clip. The cost: 77 percent (640x480) and 86 percent (1080p) of correctly segmented persons were also turned into black boxes.
6. **Mask styles need person boxes.** `silhouette` with `--detector motion` left a fully visible head on 94 of 655 frames of the 640x480 clip, because the segmenter marks something on almost every frame and misses the person, and the orphan rule needs person boxes. With `--detector both --person-detector dnn` that count was 0. Without `bodybox`, the run logs a warning for mask styles with `--detector motion`.
7. **A false face can clear a person box.** YuNet sometimes reports a face on a torso (one case in the clips). A false face in the top third of a person box whose real face was missed clears that box, and that person is then shown.
8. **A person no detector localizes is shown** when another region is on the frame (see [Coverage rules](#coverage-rules)).
9. **Without `--track`, nothing escaped on the clips.** On every tested stack and both clips, no independently detected face, mask pixel, or person box was left uncovered, and every frame the engine marked as having a subject without detections of its own (lookahead gap fill, cold start) was black. That covers what a detector found; it says nothing about people every detector missed.
10. **The detectors set the speed.** Rendering takes under 1 ms per frame at 640x480 and about 0.2 to 6 ms at 1080p depending on the styles (about 2 to 6 ms with all nine). The detectors dominate: with `--detector both --person-detector dnn --faces`, about 60 ms per frame at 640x480 and about 100 to 130 ms at 1080p on the development machine under load, so a 25 fps live source drops frames.
11. **No background differ.** The mode does not use the clean background, so the background differ, which covers small and distant people in `erase`, never runs. With `--person-detector dnn` the run warns that people smaller than about 110 px tall on a 1080p frame can be shown: the DNN does not find them, and on a frame that has another region nothing covers them.
12. **Not implemented yet** (planned for phase 6 of the [refactor plan](../../docs/REFACTOR.md)): a head estimated from a person box when no face is found, webcam and CCTV profiles, and multi-camera use.

The detectors' own limits apply too: see [Limits, before you choose one](../../README.md#limits-before-you-choose-one) in the main README. The DNN person detector misses people smaller than about 110 px tall on a 1080p frame, and the segmenter misses small, distant people.

## Recommended command lines

Each line below was run on a demo clip (`camera.mp4` is the 1920x1080 driveway clip, `camera2.mp4` the 640x480 corridor clip), with the model files in `PETRIFY_MODEL_DIR`. The log excerpt is the end of the run. `frozen` counts the frames with a subject, drawn or black; `pass-through` the frames written unchanged.

**CCTV: body boxes and face boxes.** Every person box is filled, every face is covered on top, and the tracker holds lost boxes for a few frames.

```bash
petrify run --source camera.mp4 --output out.mp4 --mode shadow \
    --mode-option styles=bodybox,facebox \
    --detector both --person-detector dnn --faces --track
```

```
[WARNING] petrify.pipeline: The DNN person detector runs without the background differ (--no-bg-diff, or a mode that does not use it): the DNN misses people smaller than about 110 px tall on a 1080p frame, and nothing else covers that gap. Small or distant people can be shown.
[INFO] petrify.pipeline: Run done: frames=1169 frozen=1129 pass-through=40 failed-closed=0 (over budget 0) source-dropped=0 sink-failures=0
```

The warning is expected: the shadow mode does not use the background differ (see limit 11).

**Silhouette with a person box detector.** The mask is filled black, and a person the segmenter misses is filled as an orphan box. Keep a person box detector with the mask styles (see limit 6).

```bash
petrify run --source camera2.mp4 --output out.mp4 --mode shadow \
    --mode-option styles=silhouette \
    --segment --detector both --person-detector dnn
```

```
[INFO] petrify.pipeline: Run done: frames=655 frozen=655 pass-through=0 failed-closed=0 (over budget 0) source-dropped=0 sink-failures=0
```

On this clip every frame had a subject (the segmenter marks something on almost every frame), so none passed through unchanged.

**Faces only, minimal.** The default style (`facebox`) with the face detector and motion. Bodies stay visible, and a frame with motion but no face is black.

```bash
petrify run --source camera2.mp4 --output out.mp4 --mode shadow --faces
```

```
[INFO] petrify.pipeline: Run done: frames=655 frozen=428 pass-through=227 failed-closed=0 (over budget 0) source-dropped=0 sink-failures=0
```

## Weakened protections

With `--web`, the settings page lists these sentences under "Weakened protections" for the options in effect, and each newly weakened one is logged as a warning.

| Listed when | Sentence | What it means |
|---|---|---|
| No body style is selected | Only faces are covered: bodies, clothing and posture stay visible, and a face the face detector misses is shown when another face is found on the same frame, unless the person detector finds that person. | Only the face cover protects the person, and only the person detector's orphan rule catches a missed face |
| `gaussian` or `pixelate` without `noise` or `facebox` | The gaussian and pixelate styles can be partially reversed by machine learning models, and a face no body style covers is protected only by them; add noise or facebox for a cover that cannot be undone. | The face cover leaks (limit 2), with or without a body style, since a person the body style misses keeps only the face cover |
| The above, with `strength` medium or low (VALUE) | strength is VALUE: the blur or pixelation is weaker than the default and easier to reverse. | More of the face survives (limit 2) |
| `eyebar` is the only face style | The eyebar style covers only the eyes: a face no body style covers stays visible apart from the eyes and can be recognized. | See limit 3 |
| A face style with `padding` below 0.3 | padding is VALUE, below the default of 0.3: hair, ears and chin around the face box can be shown. | Less of the head is covered (limit 4) |
| `silhouette` or `silhouette_color` with `dilate` below 0.01 | dilate is VALUE, below the default of 0.01: the edge of the person can show around the silhouette. | The mask edge is not grown enough to cover the outline of the person |
| `silhouette` or `silhouette_color` without `bodybox` | The body of a person the segmenter misses is shown when another person is segmented on the same frame, unless the person detector finds that person; add bodybox to cover every person box. | Only the orphan rule catches a person the segmenter misses, and only with person boxes |

petrify itself adds `The tracker is on but does not hold persons: no detector in use provides them, so persons are not held between detections.` when `--track` is on and no person box detector runs (`--detector motion`), since the mode requires person boxes.

At startup, a mask style without `bodybox` and with `--detector motion` logs this warning:

```
shadow: the mask styles run without person boxes (--detector motion). A person the segmenter misses is shown whenever the segmenter marks anything else on the frame, and the orphan rule that would cover that person needs person boxes. Use --detector person or both, with --person-detector dnn.
```

## Tests and tools

The tests load the plugin through an entry point written to a temporary directory, so they run without installing it. From the repository root, with petrify's development dependencies installed:

```bash
python -m pytest plugins/petrify-shadow/tests
```

They are also part of the root `python -m pytest` run. The end-to-end tests are marked `models`: they need the model files in `PETRIFY_MODEL_DIR` and the demo clips in `data/`, and are skipped without them. `-m "not models"` leaves them out.

`tools/padding_sheet.py` writes contact sheets to judge the face padding by eye. It runs YuNet over a clip and, for each padding value, writes `<clip>_padding_<value>.png` with tiles of sampled faces: the face box in green, the padded cover in red. The sheets show frames of the clip, so write them outside the repository:

```bash
PYTHONPATH=. python plugins/petrify-shadow/tools/padding_sheet.py \
    data/camera.mp4 MODEL_DIR /tmp/sheets --paddings 0.2,0.3,0.4
```

`MODEL_DIR` is the directory that holds `face_detection_yunet_2023mar.onnx`. `--paddings` defaults to `0.1,0.2,0.3,0.4,0.5`, and `--faces` (default 48) caps the faces per sheet.
