# Writing a petrify mode plugin

A mode decides what petrify writes for each frame. The built-in mode, `erase`, shows the clean background while a subject is present and the live frame otherwise. A plugin is an installed Python package that adds another mode, selected with `petrify run --mode NAME`.

This guide describes the contract as it is implemented today. Read [What is not available yet](#what-is-not-available-yet) before you plan a mode: the face, DNN person, and segmentation detectors do not exist yet, so a mode can only rely on motion, HOG person boxes, and the background differ.

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
- [ ] **Declare `detect_policy` on the mode.** `"first_hit"` stops running stateless detectors on a frame once one has found a subject, which suits a mode that only needs a yes or no. `"all"` runs every active detector on every frame, which a mode that draws on regions needs. A plugin mode without `detect_policy` is refused at startup, because the default `"first_hit"` would hand a region mode incomplete regions.
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

- **Only what the mode requires runs.** A detector is active when it provides a kind in the mode's `requires`. petrify refuses to start when the mode requires detections and no active detector or refiner provides any of them.
- **Built-in detectors.** `motion` (MOG2) is built with `--detector motion` or `both`, `persons` (HOG) with `--detector person` or `both`, and `background_diff` only for a mode with `needs_background` (unless `--no-bg-diff`). A mode that requires `background_diff` without a background is refused unless the differ is turned off.
- **Validated and normalized.** Each detector result is checked: a field may be set only when the detector declares the matching kind, flags become Python `bool`, boxes must have integer coordinates and non-negative size, and the mask must be a `bool` array of the frame's height and width. A malformed result fails the frame closed (black). The mode receives new objects built from the checked values, never the detector's own.
- **Boxes are clipped to the frame.** A box that starts before 0 or extends past the edge is clipped; a box wholly outside becomes a zero-area box on the nearest edge. Landmark points are not clipped.
- **The mask is a read-only copy.**
- **Results are merged as a union.** Flags are OR-ed, boxes and faces are concatenated, masks are OR-ed pixel by pixel. A field left `None` by every detector stays `None` ("not computed").
- **Refiners run after the detectors.** With `--track` (or `--track-hold N`), petrify adds a box tracker for `persons` and for `faces` when the mode requires them. It holds the last box of a lost subject for N frames (default 15) without moving it. Refiners need `detect_policy = "all"`, which is why `erase` refuses `--track`. Refiners do not run on a frame whose detection failed; that frame is written black.
- **A subject is present** when motion, a person box, a background difference, a face, or a mask pixel is found.

## What is not available yet

- **No face, DNN person, or segmentation detector.** `DetectionKind.FACES`, `LANDMARKS`, and `MASK` exist in the contract, but no detector provides them yet: YuNet faces, DNN person boxes, and person segmentation need model files that are not available yet. A mode that requires only those kinds is refused at startup. A mode that requires them alongside motion or persons will receive `None` in those fields; declare them `mandatory` if the mode cannot work without them.
- **No model files.** petrify has a local model lookup with pinned SHA-256 checks, but its list of known models is empty, and `--model-dir` has no effect yet. petrify never downloads anything.
- **No `petrify-shadow` plugin.** The face blur, eye bar, and silhouette modes are planned for Phase 5 of the [refactor plan](REFACTOR.md).
