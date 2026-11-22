# Writing a petrify mode plugin

A mode decides what petrify writes for each frame. The built-in mode, `erase`, shows the clean background while a subject is present and the live frame otherwise. A plugin is an installed Python package that adds another mode, selected with `petrify run --mode NAME`.

This guide describes the contract as it is implemented today. Read [Which detector provides what](#which-detector-provides-what) before you plan a mode: faces, landmarks, and the person mask come from model-based detectors that the user must enable with a flag and supply model files for, and each has measured limits.

## Trust model

Read this before you install or write a plugin.

- A mode plugin is ordinary Python code. It runs inside the petrify process with petrify's permissions: it can read and write files, open network connections, and do anything else petrify can.
- A mode receives the unprocessed frames, people included. A plugin can store or send them anywhere.
- Installing a plugin means trusting it as much as petrify itself. There is no sandbox, and petrify does not vet plugin code.
- The checks petrify applies to a plugin (described below) catch mistakes in an honest plugin. They do not protect against a hostile one.

petrify repeats this at run time. When the selected mode comes from a plugin, the run logs a warning that names the plugin's distribution and version, and the web page shows a notice next to the mode.

A plugin that is not selected is never imported: `petrify run --list-modes` and the mode listing on the web page read package metadata only.

## What a mode is and when the engine calls it

For each frame the engine:

1. runs the detectors the mode requires, then any refiners (the box tracker), and merges their results into one `Detections`;
2. pushes the frame into the lookahead buffer, which delays it by `--lookahead` frames and decides whether a subject is present (gap fill around every detection, and a cold-start window at the beginning of the run, are both treated as "subject present");
3. when the frame leaves the buffer, calls `mode.render(frame, detections, subject_present)`;
4. applies the stages (the live timestamp ROI overlay, when the mode allows it) and writes the result to every output.

`render` is not called for a frame whose detection failed or went over the frame budget: that frame is written black. If `render` raises, or returns an image that breaks the output rules below, the frame is written black too and the run continues.

`subject_present` is the engine's final decision after gap fill and cold start, not the raw detection. A mode cannot tell a cold-start frame or a gap-filled frame from a frame where a detector fired. On those frames `detections` holds whatever the detectors found on that frame itself, which is usually nothing.

A plugin publishes a **factory** under the `petrify.modes` entry point group. The factory describes the mode (name, options, what it needs) and builds the mode object, which does the rendering.

## A minimal working example

This plugin adds a `blackout` mode: while a subject is present it covers the whole frame with one flat gray level, otherwise it passes the frame through. It requires only motion, so it needs no model file, no background, and no person detector.

Layout:

```
petrify-blackout/
  pyproject.toml
  src/
    petrify_blackout/
      __init__.py
```

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "petrify-blackout"
version = "0.1.0"
dependencies = ["petrify", "numpy"]

[project.entry-points."petrify.modes"]
blackout = "petrify_blackout:BLACKOUT"
```

The entry point name (`blackout`) must equal the factory's `name` and the mode's `name`. Its value is `module:attr` and must name a factory instance, not a class.

`src/petrify_blackout/__init__.py`:

```python
"""A petrify mode that covers the whole frame while a subject is present."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import Mode, ModeContext, OptionSpec

_REQUIRES = frozenset({DetectionKind.MOTION})


class BlackoutMode:
    name = "blackout"
    requires = _REQUIRES
    detect_policy = "first_hit"  # A yes or no is enough: no regions are drawn.
    mandatory: frozenset[DetectionKind] = frozenset()

    def __init__(self, level: int) -> None:
        self.level = level

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        if subject_present:
            # A new array: it must not share memory with frame.image.
            return np.full_like(frame.image, self.level)
        return frame.image


class BlackoutFactory:
    name = "blackout"
    summary = "Cover the whole frame with a flat gray level while a subject is present."
    options = (
        OptionSpec(
            name="level",
            kind="int",
            default=0,
            help="Gray level of the cover, 0 (black) to 255 (white).",
            minimum=0,
            maximum=255,
            runtime=True,
        ),
    )
    needs_background = False
    roi_overlay = False  # Nothing live is pasted over the cover.

    def requires(self, options: Mapping[str, object]) -> frozenset[DetectionKind]:
        return _REQUIRES

    def mandatory(self, options: Mapping[str, object]) -> frozenset[DetectionKind]:
        return frozenset()

    def weakened(self, options: Mapping[str, object]) -> tuple[str, ...]:
        return ()  # Every level covers the whole frame.

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        level = ctx.options["level"]
        if isinstance(previous, BlackoutMode) and previous.level == level:
            return previous
        return BlackoutMode(level)


BLACKOUT = BlackoutFactory()
```

Install the package into the environment petrify runs in, as you would any Python package. (For the transcript below the package was not installed: its `src` directory, with a hand-written `petrify_blackout-0.1.0.dist-info` directory holding `METADATA` and `entry_points.txt`, was put on `PYTHONPATH`. Python finds the entry point the same way.)

The mode is then listed, and `--mode` with `--list-modes` shows its options:

```
$ petrify run --list-modes --mode blackout
Modes (select one with --mode NAME):
  erase (built in): Show the clean background while a subject is present, the live frame otherwise.
  blackout (plugin petrify-blackout 0.1.0)

Mode blackout (plugin petrify-blackout 0.1.0): Cover the whole frame with a flat gray level while a subject is present.
Options (set with --mode-option NAME=VALUE):
  level
    kind: int
    default: 0
    range: 0 to 255
    changeable while running: yes
    help: Gray level of the cover, 0 (black) to 255 (white).
```

Without `--mode`, the listing shows the plugin's name and origin only, since reading its summary would mean importing it.

A run on the 655-frame corridor clip (log excerpt, progress bar and FFmpeg lines left out):

```
$ petrify run --source camera2.mp4 --mode blackout --mode-option level=128 --output out.mp4
[WARNING] petrify.cli: The mode 'blackout' comes from the installed plugin petrify-blackout 0.1.0. It runs in the petrify process with petrify's permissions and receives the unprocessed frames, people included.
[INFO] petrify.sources.file: Input: camera2.mp4  |  640x480 @ 6.00 fps  |  ~655 frames
[INFO] petrify.cli: Source: camera2.mp4 (file), mode: blackout
[INFO] petrify.pipeline: Detector mode: motion
[INFO] petrify.pipeline: ROI: not used, the mode 'blackout' does not allow the overlay
[INFO] petrify.pipeline: Run done: frames=655 frozen=405 pass-through=250 failed-closed=0 (over budget 0) source-dropped=0 sink-failures=0
```

In `out.mp4` the 405 frames counted as frozen are flat gray; the other 250 are the live frames.

This mode needs the motion detector, which is the default (`--detector motion`). With `--detector person` no detector provides motion, and the run stops at startup with an error that the mode requires `motion` but no detector or refiner provides it.

## Safety checklist

A mode is the last step before the outputs. These rules are what keeps a person out of the output. petrify checks some of them, as noted; the rest are yours to get right.

- [ ] **Cover the subject whenever `subject_present` is True.** This includes cold-start and gap-filled frames, where there may be no detection at all.
- [ ] **Cover the whole frame when you have no regions to draw on.** A region mode (one that blurs or masks boxes, faces, or a mask) receives no regions on frames frozen by the lookahead or the cold start, and a field that is `None` means "not computed", not "nothing found". When `subject_present` is True and the regions you need are missing or empty, cover everything.
- [ ] **Never return the input image, or a view of it, on a frame where `subject_present` is True.** Return a new array. petrify checks this with `numpy.may_share_memory` and writes the frame black otherwise; even a view of a disjoint slice is rejected. When `subject_present` is False the mode may return `frame.image` itself.
- [ ] **Return an image of the right type.** It must be a `numpy.ndarray` (not a subclass) with the input frame's shape and dtype (BGR `uint8`, height by width by 3). petrify checks this and writes the frame black otherwise.
- [ ] **Declare `detect_policy` on the mode.** `"first_hit"` stops running stateless detectors on a frame once one has found a subject, which suits a mode that only needs a yes or no. `"all"` runs every active detector on every frame, which a mode that draws on regions needs. A plugin mode without `detect_policy` is refused at startup, because the default `"first_hit"` would hand a region mode incomplete regions. A mode that requires `faces`, `landmarks`, or `mask` together with any other kind must declare `"all"`; with `"first_hit"` it is refused at startup (see [The `first_hit` rule](#the-first_hit-rule)).
- [ ] **Set `roi_overlay = False` on the factory of a region mode.** The ROI overlay stage pastes the live pixels of the timestamp region over the mode's output. That is right for `erase`, which shows a frozen background with a live clock, and wrong for a mode that obscures regions: a face inside the ROI would be pasted back unobscured. The setting is per mode, and a mode with `roi_overlay = False` gets no ROI stage at all.
- [ ] **Declare `mandatory` kinds** for detections the mode cannot work without. Every kind in `mandatory` must be provided by an active detector or refiner, or petrify refuses to start (and refuses a settings change that would remove it). A non-empty `mandatory` requires `detect_policy = "all"`. The factory's `mandatory(options)` must return the same set as the mode's `mandatory` attribute (missing means empty).
- [ ] **Give runtime number options a `minimum` and a `maximum`.** An `int` or `float` option with `runtime=True` must declare both, so a value sent from the web page is always bounded; `OptionSpec` raises `ValueError` otherwise.
- [ ] **Report weakened protection in `weakened(options)`.** Return one plain sentence for each option value that lets more of a person through (for example a low blur strength, or a style that leaves the body visible). The web page lists these sentences under "Weakened protections" next to petrify's own. Return an empty tuple when nothing is weakened.

## The contract in detail

### `ModeFactory`

The object the entry point names. petrify checks it when the mode is loaded and refuses it with a message when something is missing or has the wrong type.

| Member | Meaning |
|---|---|
| `name: str` | The mode name. Must match `[a-z][a-z0-9_]{0,31}`, the entry point name, and the mode's `name` |
| `summary: str` | One line for listings |
| `options: tuple[OptionSpec, ...]` | Every option the mode accepts, with distinct names |
| `needs_background: bool` | True when the mode needs the clean background. The pipeline then builds the locked image or the online model and passes it in `ModeContext.background` |
| `roi_overlay: bool` | Whether the live timestamp ROI stage may run after this mode (see the checklist) |
| `requires(options) -> frozenset[DetectionKind]` | The `requires` of the mode built with these options |
| `mandatory(options) -> frozenset[DetectionKind]` | The `mandatory` of the mode built with these options |
| `create(ctx, previous=None) -> Mode` | Build the mode. `previous` is the mode currently in use, or None. When nothing the mode depends on changed, return `previous` itself |
| `weakened(options) -> tuple[str, ...]` | One sentence per option value that reduces protection |

The methods that take `options` receive the resolved options (every declared option, validated, defaults filled in) and must not keep or mutate them.

### `Mode`

| Member | Meaning |
|---|---|
| `name: str` | Same as the factory's |
| `requires: frozenset[DetectionKind]` | The detections the mode uses; must equal `factory.requires(options)` |
| `render(frame, detections, subject_present) -> numpy.ndarray` | The output image (see the checklist) |
| `detect_policy` | `"first_hit"` or `"all"`; required for a plugin mode |
| `mandatory` | Optional, a `frozenset[DetectionKind]`; must equal `factory.mandatory(options)` |

`frame` is a `petrify.core.frame.Frame` with `image`, `index`, `timestamp`, and `source`. `detections` is described in [What the engine does to detections](#what-the-engine-does-to-detections).

### `ModeContext`

What `create` receives:

| Field | Meaning |
|---|---|
| `source` | `SourceInfo`: frame rate, width, height, frame count (None when unknown), whether the source is seekable and live |
| `config` | The whole `PetrifyConfig` of the run |
| `options` | A read-only mapping of every declared option, each in its declared type |
| `background` | The locked background image or the online `BackgroundModel` when the factory sets `needs_background`, None otherwise |

### `OptionSpec`

Each option has a `name` (same pattern as mode names), a `kind`, a `default`, a `help` text, and optionally `choices`, `minimum`, `maximum` (inclusive), and `runtime`.

| `kind` | Python type of every value | Written on the command line as |
|---|---|---|
| `bool` | `bool` | `true`, `false`, `yes`, `no`, `on`, `off`, `1`, `0` (any case) |
| `int` | `int` | a whole number, such as `3` |
| `float` | `float` | a number, such as `0.5` (no `inf` or `nan`) |
| `choice` | `str`, one of `choices` | a choice name |
| `choices` | `tuple[str, ...]`, distinct entries of `choices`, possibly empty | comma-separated choice names, such as `a,b` (empty text is the empty list) |

Every value reaches the mode in that single canonical type, so `1`, `True`, and `1.0` never stand in for each other. `minimum` and `maximum` apply only to `int` and `float`. Choice names follow the same pattern as option names. The spec checks itself when it is constructed and raises `ValueError` for a bad declaration.

## Options on the command line and the web page

On the command line, set options with `--mode-option KEY=VALUE`, once per option. An unknown option, a repeated option, a value of the wrong form, or a value out of range is a usage error (exit code 2) that names the mode's options:

```
$ petrify run --source camera2.mp4 --mode blackout --mode-option level=300 --output out.mp4
petrify run: error: --mode-option: option 'level' must be at most 255, got 300. Options of the mode 'blackout': level (int)
```

`petrify run --list-modes --mode NAME` prints each option's kind, default, choices or range, whether it can be changed while running, and its help.

With `--web`, the page's Mode section shows the active mode, its origin, its summary, and its options with their current values. Options declared with `runtime=True` get a control and can be changed while the run is in progress; the others are shown as fixed for the run. A change is validated against the spec, applied at the next frame boundary, and, like every settings change, briefly freezes the output around the switch. The mode itself cannot be switched during a run.

Text the plugin supplies (summary, option help, weakened sentences, error messages) is shown as plain text: control and formatting characters become spaces and the length is capped. Text from package metadata (mode names, distribution names, versions) is reduced to printable ASCII, with other characters shown as `?`, so a name cannot pass for another mode's.

## How errors are treated

| Where | What happens |
|---|---|
| The entry point cannot be imported, or the object is not a valid factory | The run stops at startup with `Fatal error:` and a message naming the mode and its distribution (exit code 1) |
| `create`, `requires`, or `mandatory` raises `ValueError` at startup | The settings are not acceptable for this mode: the run stops with `Fatal error:` and the plugin's message (exit code 1) |
| `create`, `requires`, or `mandatory` raises any other exception at startup, or returns something that breaks the contract | The plugin is treated as faulty: the run stops with `Fatal error: mode 'NAME': create failed: TypeName: message (mode 'NAME', distribution version)` (exit code 1) |
| Any of these fails during a run, after a change from the web page | The change is rejected, the previous settings stay in effect, and the message is shown as the page's last error and logged |
| `render` raises or returns a bad image | That frame is written black and counted as failed closed; the run continues |
| `weakened` raises or returns something other than a tuple of strings | The page lists a sentence saying the mode could not report its weakened protections |

Raise `ValueError` from `create` for options or a context your mode cannot work with. Reserve other exceptions for real bugs.

A mode name that is registered by more than one distribution, that is not a valid name, whose entry point value is not `module:attr`, or that is `erase` (the built-in cannot be shadowed) is listed by `--list-modes` as not selectable, with the reason, and `--mode` refuses it. A distribution whose metadata cannot be read is skipped with a warning; it does not stop petrify.

## What the engine does to detections

Before a mode sees them, detector results go through these steps:

- **Only what the mode requires runs.** A detector is active when it provides a kind in the mode's `requires`. petrify refuses to start when the mode requires detections and no active detector or refiner provides any of them, and when it requires `faces`, `landmarks`, or `mask` and the detector that provides that kind is not enabled (see [Which detector provides what](#which-detector-provides-what)).
- **Built-in detectors.** `motion` (MOG2) is built with `--detector motion` or `both`, `persons` (HOG, or the DNN with `--person-detector dnn`) with `--detector person` or `both`, and `background_diff` only for a mode with `needs_background` (unless `--no-bg-diff`). A mode that requires `background_diff` without a background is refused unless the differ is turned off. The face detector and the segmenter are built only when enabled and the mode requires what they provide.
- **Validated and normalized.** Each detector result is checked: a field may be set only when the detector declares the matching kind, flags become Python `bool`, boxes must have integer coordinates and non-negative size, and the mask must be a `bool` array of the frame's height and width. A malformed result fails the frame closed (black). The mode receives new objects built from the checked values, never the detector's own.
- **Boxes are clipped to the frame.** A box that starts before 0 or extends past the edge is clipped; a box wholly outside becomes a zero-area box on the nearest edge. Landmark points are not clipped.
- **The mask is a read-only copy.**
- **Results are merged as a union.** Flags are OR-ed, boxes and faces are concatenated, masks are OR-ed pixel by pixel. A field left `None` by every detector stays `None` ("not computed").
- **Refiners run after the detectors.** With `--track` (or `--track-hold N`), petrify adds a box tracker for `persons` and for `faces` when the mode requires them. It holds the last box of a lost subject for N frames (default 15) without moving it. Refiners need `detect_policy = "all"`, which is why `erase` refuses `--track`. Refiners do not run on a frame whose detection failed; that frame is written black.
- **A subject is present** when motion, a person box, a background difference, a face, or a mask pixel is found.

## Which detector provides what

| `DetectionKind` | Detector | Enabled by |
|---|---|---|
| `MOTION` | MOG2 | `--detector motion` (the default) or `both` |
| `PERSONS` | HOG, or NanoDet-Plus with `--person-detector dnn` | `--detector person` or `both` |
| `BACKGROUND_DIFF` | The static background differ | A factory with `needs_background`, unless `--no-bg-diff` |
| `FACES` | YuNet | `--faces` |
| `LANDMARKS` | YuNet (on the faces it returns) | `--faces` |
| `MASK` | PP-HumanSeg | `--segment` |

YuNet, NanoDet-Plus, and PP-HumanSeg need model files that the user places on disk; petrify never downloads them. The README's [Model-based detectors](../README.md#model-based-detectors) section lists the files, where petrify looks for them, and the measured limits of each detector. Read the limits before you rely on one: the segmenter, for example, misses small, distant people and marks parts of empty scenes.

When a mode requires `faces`, `landmarks`, or `mask` and the user did not enable the detector that provides it, petrify refuses to start, whatever else the mode requires:

```
Fatal error: The mode 'facebox' requires faces but no enabled detector provides it; enable the YuNet face detector with --faces
```

The same check rejects a settings change from the web page that would leave such a kind without its detector. `motion` and `persons` keep the older rule: the run starts as long as at least one required kind is provided, which is what lets `erase` run with `--detector motion`.

Every model file a run needs is loaded and verified at startup, before any frame is read. A missing or corrupt file stops the run with `Fatal error:` (exit code 1).

### The `first_hit` rule

Under `"first_hit"` the engine skips the later detectors on a frame where an earlier one fired, so a mode that needs faces together with motion, for example, would get `faces = None` on most frames with a subject. petrify therefore refuses, at startup, a `"first_hit"` mode whose `requires` contains `faces` or `landmarks` together with a kind other than those two, or `mask` together with any other kind. Such a mode must declare `detect_policy = "all"`. A mode that requires only `faces` and `landmarks`, or only `mask`, may keep `"first_hit"`, since a single detector runs.

### What the mode receives

Each field below is `None` when it was not computed (its detector is not active, or `"first_hit"` skipped it on that frame) and empty when the detector ran and found nothing: an empty tuple, or a mask with no True pixel. On a gap-filled or cold-start frame the fields hold what the detectors found on that frame itself, usually nothing.

- **`detections.faces`**: a tuple of `petrify.core.detections.Face`. `face.box` is a `Box` with integer `x`, `y`, `w`, `h` in pixels of the full frame, rounded outward to whole pixels and clipped to the frame, and `score` set to the detector's confidence. YuNet runs on the full-resolution frame. A frame with a side of 32 px or less is refused and written black.
- **`face.landmarks`**: set exactly when `LANDMARKS` is provided (it is, whenever `--faces` is on): a tuple of five `(x, y)` float points in frame pixels, in this order: right eye, left eye, nose tip, right mouth corner, left mouth corner. Landmark points are not clipped and can lie outside the frame.
- **`detections.mask`**: a read-only `numpy` array of dtype `bool` and shape `(H, W)`, the full resolution of the frame. True marks a person pixel. The model runs on a 192 by 192 input; when its output is resized to the frame, every pixel the model flagged is kept. `--segment-threshold` sets the probability from which a pixel is a person.
- **`detections.persons`** from the DNN: boxes padded by 10 percent of their width and 7 percent of their height on each side, overlapping candidates merged into their union rather than dropped. An outstretched arm can still stick out of a box.

With `--track`, the box tracker also holds face boxes for `--track-hold` frames after a miss.

## A mode that uses faces

This plugin adds a `facebox` mode: it paints every detected face box with a flat gray level, and covers the whole frame when a subject is present but no face was found. It is a sketch for trying out the face detector, not a safe anonymizer: it shows every body, and a face YuNet misses is shown. Package it like the `blackout` example, with the entry point `facebox = "petrify_facebox:FACEBOX"`.

```python
"""A petrify mode that paints every detected face box with a flat gray level."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from petrify.core.detections import DetectionKind, Detections
from petrify.core.frame import Frame
from petrify.core.plugin import Mode, ModeContext, OptionSpec

_REQUIRES = frozenset({DetectionKind.FACES})


class FaceboxMode:
    name = "facebox"
    requires = _REQUIRES
    detect_policy = "all"  # A region mode: it draws on the boxes.
    mandatory = _REQUIRES  # Without the face detector there is nothing to draw on.

    def __init__(self, level: int) -> None:
        self.level = level

    def render(self, frame: Frame, detections: Detections, subject_present: bool) -> np.ndarray:
        if not subject_present:
            return frame.image
        faces = detections.faces
        if not faces:
            # Gap-filled or cold-start frame, or faces not computed: cover everything.
            return np.full_like(frame.image, self.level)
        out = frame.image.copy()  # A new array, never a view of frame.image.
        for face in faces:
            b = face.box
            out[b.y : b.y + b.h, b.x : b.x + b.w] = self.level
        return out


class FaceboxFactory:
    name = "facebox"
    summary = "Paint every detected face box with a flat gray level."
    options = (
        OptionSpec(
            name="level",
            kind="int",
            default=0,
            help="Gray level of the boxes, 0 (black) to 255 (white).",
            minimum=0,
            maximum=255,
            runtime=True,
        ),
    )
    needs_background = False
    roi_overlay = False  # A face inside the timestamp region must not be pasted back.

    def requires(self, options: Mapping[str, object]) -> frozenset[DetectionKind]:
        return _REQUIRES

    def mandatory(self, options: Mapping[str, object]) -> frozenset[DetectionKind]:
        return _REQUIRES

    def weakened(self, options: Mapping[str, object]) -> tuple[str, ...]:
        return ()

    def create(self, ctx: ModeContext, previous: Mode | None = None) -> Mode:
        level = ctx.options["level"]
        if isinstance(previous, FaceboxMode) and previous.level == level:
            return previous
        return FaceboxMode(level)


FACEBOX = FaceboxFactory()
```

Without `--faces` the run stops at startup with the error shown in [Which detector provides what](#which-detector-provides-what). With it, on the 1169-frame 1080p driveway clip (log excerpt, progress bar and FFmpeg lines left out; the model file was in `PETRIFY_MODEL_DIR`):

```
$ petrify run --source camera.mp4 --mode facebox --faces --output out.mp4
[WARNING] petrify.cli: The mode 'facebox' comes from the installed plugin petrify-facebox 0.1.0. It runs in the petrify process with petrify's permissions and receives the unprocessed frames, people included.
[INFO] petrify.sources.file: Input: camera.mp4  |  1920x1080 @ 25.00 fps  |  ~1169 frames
[INFO] petrify.cli: Source: camera.mp4 (file), mode: facebox
[INFO] petrify.pipeline: Detector FaceYuNetDetector ready: model .../face_detection_yunet_2023mar.onnx loaded and verified
[INFO] petrify.pipeline: Detector mode: motion (not used: the mode 'facebox' does not require motion). Detectors built: FaceYuNetDetector
[INFO] petrify.pipeline: ROI: not used, the mode 'facebox' does not allow the overlay
[INFO] petrify.pipeline: Run done: frames=1169 frozen=1034 pass-through=135 failed-closed=0 (over budget 0) source-dropped=0 sink-failures=0
```

YuNet found a face on 805 frames, and those frames have the face box painted. The other 229 frames counted as frozen are gap-filled or cold-start frames with no face of their own, and are covered whole. The `Detector mode` line reports the `--detector` setting and says that MOG2 does not run, since the mode does not require motion; YuNet is the only detector built.

## What is not available yet

- **No built-in mode uses faces or a mask.** `erase` uses motion, person boxes, and the background differ only, so `--faces` and `--segment` matter only to a plugin mode today.
- **No `petrify-shadow` plugin.** The face blur, eye bar, and silhouette modes are planned for Phase 5 of the [refactor plan](REFACTOR.md).
