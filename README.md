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

The run form is selected when the first argument that is not an option is `run` and `--source` (or `--list-modes`) is given. Otherwise the arguments are `INPUT OUTPUT`, so a file named `run` still works: `petrify run out.mp4`. A command that starts with `run` and uses an option only the run form has, but no `--source` (for example `petrify run --output out.mp4`), is a usage error that says `petrify run` requires `--source` (exit code 2).

### Modes and plugins

| Flag | Meaning |
|---|---|
| `--mode NAME` | The processing mode. Default: `erase` (built in), which freezes the subject out of the scene. Installed plugins can add modes |
| `--mode-option KEY=VALUE` | Set an option of the selected mode; repeat it for several options. The erase mode has none |
| `--list-modes` | Print the available modes and exit; needs no `--source`. With `--mode NAME`, also print that mode's options (kind, default, range or choices, whether it can be changed while running) |

```bash
petrify run --list-modes
petrify run --list-modes --mode NAME
```

A plugin is a Python package that registers a mode under the `petrify.modes` entry point group. `--list-modes` reads package metadata only and imports no plugin except the one named by `--mode`. A mode name registered twice, or a plugin that tries to use the name `erase`, is listed as not selectable with the reason. A package with unreadable metadata is skipped with a warning.

> **A mode plugin is ordinary Python code.** It runs in the petrify process with petrify's permissions and receives the unprocessed frames, people included. Install only plugins you trust. petrify's checks on a plugin catch mistakes, not a hostile plugin. When the selected mode is a plugin, the run logs a warning naming its package and version.

[docs/PLUGINS.md](docs/PLUGINS.md) describes how to write one.

#### Shadow mode (plugin)

The repository holds one mode plugin, `petrify-shadow`, in `plugins/petrify-shadow`. It registers the mode `shadow`, which keeps the live frame and covers the people in it: faces with a blur, pixelation, noise, a solid box, or an eye bar, and bodies with a filled silhouette or a filled person box. It is not installed with petrify; install it from the repository root with `python -m pip install --no-deps -e plugins/petrify-shadow`.

```bash
petrify run --source input.mp4 --output out.mp4 --mode shadow \
    --mode-option styles=bodybox,facebox \
    --detector both --person-detector dnn --faces --track
```

Its styles depend on the model-based detectors below, each of which misses people, and the blur and pixelation styles can be partly reversed. Read the [plugin README](plugins/petrify-shadow/README.md), in particular its limits, before you rely on it.

### Tracking and model files

| Flag | Meaning |
|---|---|
| `--track` | Enable the box tracker: a person or face box is held for a few frames after the detector stops finding it. Only for modes that use boxes; the erase mode does not accept it, and the run stops with an error |
| `--track-hold N` | Frames a tracked box is held after its last detection (implies `--track`). Default: 15 |
| `--model-dir DIR` | Directory searched first for detector model files, before `PETRIFY_MODEL_DIR` and the per-user model directory. Only the [model-based detectors](#model-based-detectors) and their background checks use it |

### Model-based detectors

Three detectors run an ONNX model from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo) through OpenCV. They are off by default, and without their flags a run behaves as before.

| Detector | Flag | What it is for |
|---|---|---|
| NanoDet-Plus person boxes | `--person-detector dnn` | Person boxes for `--detector person` and `both`, in place of HOG |
| YuNet faces | `--faces` | Face boxes with five landmarks, for a mode that uses faces or face landmarks |
| PP-HumanSeg person mask | `--segment` | A pixel mask of people, for a mode that uses a person mask |

| Flag | Meaning |
|---|---|
| `--person-detector {hog,dnn}` | Person box detector used by `--detector person` and `both`. Default: `hog`. With `--detector motion` it has no effect, and a warning says so |
| `--dnn-score-threshold FLOAT` | Minimum score in (0, 1) to accept a DNN person box. Default: 0.4. Needs `--person-detector dnn` |
| `--faces` | Enable the YuNet face detector |
| `--face-score-threshold FLOAT` | Minimum score in (0, 1) to accept a face. Default: 0.6. Needs `--faces` |
| `--segment` | Enable the PP-HumanSeg person segmenter |
| `--segment-threshold FLOAT` | Person probability in (0, 1) from which a pixel belongs to the mask. Default: 0.5. Needs `--segment` |
| `--background-check NAMES` | Detectors that check the frames a `running` or `calibrate` background is learned from, comma-separated: `hog`, `yunet`, `dnn`. Default: `hog`. See [Background strategies](#background-strategies) |

No built-in mode uses faces or a mask: `--faces` and `--segment` only matter to a [plugin mode](docs/PLUGINS.md) that requires faces, landmarks, or a mask, such as [`shadow`](#shadow-mode-plugin). With `erase` they have no effect, and a warning says so.

#### Limits, before you choose one

- **DNN person boxes replace HOG, they do not add to it.** With `--person-detector dnn` HOG no longer runs on the frames. The locked background pre-scan still uses HOG.
- **The DNN misses small people.** The frame is scaled down to 416 px for the model, and no person smaller than about 110 px tall on a 1080p frame was found. Wide frames shrink people further. The background differ covers that gap only while it is on: with `--no-bg-diff`, or a mode that does not use it, a warning says that small or distant people can be shown.
- **The DNN is slower than HOG.** Measured on an idle machine, the DNN person detector takes a median of about 33 ms per frame on the 640x480 demo clip, against a few ms with HOG, and about 45 ms at 1080p, so a 25 fps 1080p live source drops frames. On the 640x480 clip erase with `--detector person --person-detector dnn` freezes 611 of 655 frames against 494 with HOG. See [Performance](#performance) for the other detectors.
- **DNN person boxes are estimates.** Each box is padded by 10 percent of its width and 7 percent of its height on each side, because the model's boxes were measured to fall short of the person. An outstretched arm can still stick out of the box.
- **The segmenter is tuned for close, large people.** It misses small, distant people and marks parts of empty scenes. That is why it is not offered as a background check, and it should not be the only person signal on CCTV footage.
- **The face detector refuses tiny frames.** A frame with a side of 32 px or less is not processed and fails closed (black). One face detector instance is not thread-safe.

#### Model files

petrify never downloads a model. You place the files yourself, and petrify loads a file only when its size and SHA-256 match the values pinned in its code (`petrify/models.py`):

| File | Detector | Size (bytes) | SHA-256 | License |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet faces | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |
| `object_detection_nanodet_2022nov.onnx` | NanoDet-Plus person boxes | 3800954 | `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186` | Apache-2.0 |
| `human_segmentation_pphumanseg_2023mar.onnx` | PP-HumanSeg person mask | 6163938 | `552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24` | Apache-2.0 |

To get them, download each file by hand from the OpenCV model zoo at github.com/opencv/opencv_zoo: `models/face_detection_yunet/`, `models/object_detection_nanodet/`, and `models/human_segmentation_pphumanseg/`. Keep the file names as they are. You only need the files of the detectors you enable.

Each file is looked up in this order, and the first location where a file of that name exists is the only one used (a wrong file there stops the run instead of falling through to the next location):

1. `--model-dir DIR`
2. the directory in the `PETRIFY_MODEL_DIR` environment variable
3. the per-user data directory: `~/Library/Application Support/petrify/models` on macOS; `$XDG_DATA_HOME/petrify/models` (when the variable is an absolute path) or else `~/.local/share/petrify/models` on Linux; `%LOCALAPPDATA%\petrify\models` on Windows, falling back to `%USERPROFILE%\AppData\Local\petrify\models`

For library use, a `model_path` in the detector's config section names one file and replaces the search.

Every model file a run needs, for its detectors and its background checks, is loaded and verified at startup, before any frame is read. The log names the file used (`Detector PersonDnnDetector ready: model PATH loaded and verified`). A missing or corrupt file stops the run with one `Fatal error:` line (exit code 1). For a missing file it lists the locations searched, the expected size and SHA-256, the source, and the license; for a corrupt one, the expected and actual size or SHA-256. During a run, a change from the web page that would need a missing or corrupt model is rejected and the run continues.

```bash
# Erase with DNN person boxes in place of HOG
petrify run --source input.mp4 --output out.mp4 \
    --detector both --person-detector dnn --model-dir ~/petrify-models
```

#### Startup errors and warnings

petrify refuses to start, before reading any frame, when:

- a threshold flag is given without its detector: `--dnn-score-threshold` without `--person-detector dnn`, `--face-score-threshold` without `--faces`, `--segment-threshold` without `--segment` (usage error, exit code 2)
- `--background-check` is combined with `--no-background-check`, used with a `locked` background, or names a detector other than `hog`, `yunet`, or `dnn` (usage error, exit code 2)
- the mode requires faces, landmarks, or a mask and its detector is not enabled, for example `The mode 'NAME' requires faces but no enabled detector provides it; enable the YuNet face detector with --faces` (exit code 1)
- a mode with the `first_hit` detection policy requires faces, landmarks, or a mask together with another kind (exit code 1; see [docs/PLUGINS.md](docs/PLUGINS.md))
- a model file it needs is missing or fails its size or SHA-256 check (exit code 1)

It logs a warning when `--person-detector dnn` is given with `--detector motion`, when `--faces` or `--segment` is given for a mode that does not use it, when `--background-check` is given for a mode that uses no background, when the DNN person detector runs without the background differ, and when the background sample check has rejected 50 samples in a row while the background is still not ready.

### Sources

| `--source` | Reads |
|---|---|
| A file path, such as `input.mp4` | A video file. A bare number such as `0` is a file name, not a camera |
| A directory, or a glob such as `'frames/*.png'` | Still images in natural name order (`frame2.png` before `frame10.png`), at `--fps` frames per second (default 25) |
| `camera:N` | Webcam number N |
| `rtsp://`, `rtsps://`, `http://`, `https://` URL | A network stream (RTSP, HTTP MJPEG) |

On a live source (`camera:N` or a stream URL) a capture thread keeps only the newest frame, so a slow run drops frames instead of falling behind. By default a live run waits for the next frame without limit; `--source-timeout SECONDS` ends it when the source stalls (see [Run options](#run-options)). Stream URLs are redacted in logs and error messages: at most the scheme, host, and port are shown. Credentials are replaced with `***`, and the path and query are dropped.

For library use, `petrify.sources.PushSession` feeds in-memory frames to an engine one by one and returns the processed frames.

### Outputs

At least one is required, and several can be combined.

| Flag | Output |
|---|---|
| `--output PATH` | A video file, encoded by FFmpeg in one pass through a pipe (no intermediate file). FFmpeg writes to `<name>.partial<suffix>`, which is moved onto `PATH` only when the encode succeeds, so a failed run never leaves a file at `PATH` and never replaces an existing one. For a video file source its audio is copied |
| `--virtualcam` | A virtual camera that video call apps can select. `--virtualcam-device NAME` picks the device (default: the first one). Needs `petrify[virtualcam]`, and OBS 30 or newer on macOS |
| `--preview` | A local window. `q` or Escape ends the run. Needs an OpenCV build with GUI support |
| `--web` | A local web page to watch the processed feed and change settings while the run is in progress. Needs `petrify[web]`. See [The web page](#the-web-page) |

On a live source the run ends with Ctrl-C or SIGTERM (or `q` in the preview window), and the output file is still finalized. Both signals are handled the same way: a first signal that lands while the outputs are opened, a frame is processed, or the outputs are finalized takes effect when that step is done, and a second signal aborts. A live run stopped before its first frame writes no file. On a video file or image folder source a signal aborts the run and no output file is written (exit code 130). If a live run fails, the recording up to that point is kept and the error is logged. [Running several cameras](#running-several-cameras) describes what a stop from a service manager does to the recording.

### Run options

| Flag | Meaning |
|---|---|
| `--profile cctv` | A preset of defaults for a fixed security camera; see below. `cctv` is the only profile. Default: no profile |
| `--source-timeout SECONDS` | End a live run (`camera:N` or a stream URL) when no frame arrives for SECONDS, the first frame included. The recording made so far is kept, the error is logged, and petrify exits with code 1, so a service manager can restart it. Refused for a video file or image folder (usage error, exit code 2). Default: no limit |
| `--threads N` | Limit OpenCV's worker threads in this process to N (1 to 64), for several petrify processes on one machine. The run logs `OpenCV worker threads: N` with the count OpenCV reports, and a warning when it differs from the request: some OpenCV builds (the macOS ones) honor only 1. FFmpeg's encoder threads are not limited. Default: OpenCV's own thread count |

`--profile cctv` sets these values for every flag not given on the command line. A flag given on the command line always wins, also in an abbreviated or `--flag=value` form.

| Set | When |
|---|---|
| `--detector both` | Always |
| `--person-detector dnn` | Always. The DNN model file must be in place, or the run stops at startup |
| `--lookahead 10` | Always |
| `--faces` | When the mode uses faces or face landmarks (for example `shadow` with a face style) |
| `--background-check hog,dnn` | When the mode uses a background and the background is `running` or `calibrate` (the default on a live source), and neither `--background-check` nor `--no-background-check` was given |
| `--source-timeout 10` | On a live source |

It never turns on `--track`, `--segment`, or a frame budget. The run logs what the profile set, for example `Profile cctv: set --detector both, --person-detector dnn, --lookahead 10, --source-timeout 10, --background-check hog,dnn; explicit flags win`. There is no webcam profile: no setting was tested on a webcam.

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

Frames are written black until the background is ready. With `running` or `calibrate`, every frame the background is learned from is first checked, by default with the HOG person detector, and a frame in which it finds a person is not used. `--background-check NAMES` chooses the checkers from `hog`, `yunet` (faces), and `dnn` (person boxes); a frame is not used when any of them finds a person. The list replaces the default, so include `hog` to keep it, for example `--background-check hog,dnn`. `yunet` and `dnn` need their [model files](#model-files). A model checker uses the score threshold of its detector flag, and those flags need the detector enabled for the frames too (`--dnn-score-threshold` needs `--person-detector dnn`, `--face-score-threshold` needs `--faces`), so without them the checker runs at its default threshold. The segmenter is not offered as a checker, because it marks parts of empty scenes and would reject almost every sample. When the checkers reject 50 samples in a row and the background is still not ready, a warning names them. `--no-background-check` turns the check off and is unsafe. At startup petrify logs a warning that a person in view while the background is learned can be absorbed into it. The check reduces this but does not eliminate it: start with the scene empty either way. See [Before you rely on it](#before-you-rely-on-it).

### Fail closed

A frame that cannot be processed safely is written black and counted: a detector error or malformed detector result, a mode or stage error or an output image of the wrong shape or type, the background not ready yet, or a frame over `--frame-budget-ms` (the per-frame time budget for detection plus rendering; no budget by default). The end-of-run summary reports how many frames failed closed.

### The web page

`--web` serves a page on `127.0.0.1` at `--web-port` (default 8765; 0 picks a free port). At start petrify prints the link to the terminal:

```
Web page (this link gives control of this run to whoever has it): http://127.0.0.1:PORT/#token=...
```

The page shows the processed feed (what the outputs receive), the run's status (source, background state, frame counters, frames failed closed and dropped, output frame rate, detect, render, and write timings, the mean and last call time of each detector and refiner, the ROI in use, the last error), a settings form, and a button that stops the run and finalizes the outputs. While the run is in progress the page can change:

- the detector (`motion`, `person`, `both`)
- the motion detector's area and variance thresholds and shadow detection
- the HOG person detector's minimum confidence, minimum box area, and pyramid scale. The page shows which person detector the run uses; with `--person-detector dnn` these HOG settings do not apply, and a change to them is rejected
- the background differ (on or off) and its pixel and area thresholds
- the lookahead and the frame budget
- the ROI: disabled, or a manual box no larger than the run's `--roi-max-area` of the frame
- the options the active mode marks as changeable while running

The values are bounded more tightly than on the command line, so every detector can still fire. The background strategy, the source, the outputs, the mode, the encoding, the MOG2 history, and the downsample factors are fixed for the run. Changes are applied at the next frame boundary, and each one briefly freezes the output: the frames buffered at that moment and at least one lookahead window after them show the background. A change of only the frame budget freezes nothing.

A Mode section shows the active mode, its origin (built in, or the plugin's package and version), its description, its options with their current values, and the installed modes. Only options the mode marks as changeable while running get a control; the others are shown as fixed for the run. The mode itself cannot be switched during a run: stop the run and start a new one with `--mode`. When the mode comes from a plugin, the section shows a notice that it runs in the petrify process with petrify's permissions and receives the unprocessed frames.

A "Weakened protections" list on the page names every setting in effect that can let a person be shown: the background differ off, the person detector alone with the differ off, a detector threshold more than twice its startup value, a lookahead below its startup value or 0, a manual ROI over 5 percent of the frame, and the raw preview. It also lists the sentences the active mode reports for its own options; a mode that cannot report them is listed as possibly weakening the protection. Each newly weakened protection is also logged as a warning.

`--realtime` processes a video file or image folder no faster than its frame rate, so it can be watched on the page. It changes no output frame and is ignored for live sources.

Security model:

- The server listens on `127.0.0.1` only, and every request must carry a `Host` of exactly `127.0.0.1:PORT`. Use the printed `http://127.0.0.1:PORT/` link: `localhost` is refused on purpose, because the server does not listen on `[::1]`, where another local user could.
- The link holds a per-run token. Whoever has the link controls the run, so do not share it. The token is never logged, and the server's access log is off.
- The unprocessed input feed is never served unless the run was started with `--web-raw-preview`, which shows it next to the processed feed and lists it as a weakened protection.
- The page does not protect against the active mode: a plugin mode runs in the petrify process and receives the unprocessed frames whatever the page shows.
- Text that comes from a plugin (its description, option help, weakened sentences, error messages) is shown as plain text, with control and formatting characters removed and the length capped. Text from package metadata (mode names, package names, versions) is reduced to printable ASCII, other characters shown as `?`, so a name cannot pass for another mode's. Only runtime mode options can be changed, and a numeric one only within the bounds the mode declares.

---

## Running several cameras

Run one `petrify run` process per camera. One process cannot read several cameras. Give each process its own `--output` and, with `--web`, its own `--web-port`. `--threads N` limits the OpenCV worker threads of each process (see [Run options](#run-options)): some OpenCV builds honor only 1, and FFmpeg's encoder threads are not limited.

### A systemd unit

A template unit, one environment file per camera, and an output file named per start, so a restart never replaces an earlier recording (petrify moves a finished encode onto `--output`, replacing a file of that name):

```ini
# /etc/systemd/system/petrify@.service
[Unit]
Description=petrify camera %i
Wants=network-online.target
After=network-online.target

[Service]
User=petrify
EnvironmentFile=/etc/petrify/%i.env
ExecStart=/bin/sh -c 'exec petrify run --source "$${SOURCE}" --output "$${OUTPUT_DIR}/%i-$$(date +%%Y%%m%%d-%%H%%M%%S).mp4" --profile cctv --threads 2 --web --web-port "$${WEB_PORT}"'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/petrify/front.env (readable by root only: the URL can hold credentials)
SOURCE=rtsp://user:password@192.0.2.10/stream
OUTPUT_DIR=/var/lib/petrify/front
WEB_PORT=8766
PETRIFY_MODEL_DIR=/var/lib/petrify/models
```

Start it with `systemctl enable --now petrify@front`. `$$` and `%%` are systemd's escapes for `$` and `%`, so the shell expands the variables and `date` on every start. Use the full path of `petrify` if it is not on the service's `PATH`, and leave out `--web` if the page is not needed: the page link with its token is printed to the service's log, so whoever can read that log can control the run (from reading the code: the link goes to stderr, which systemd logs; not tested under systemd).

### What a stop does to the recording

- **`systemctl stop`.** systemd sends SIGTERM to every process of the unit, FFmpeg included (the default `KillMode=control-group`). petrify handles it like Ctrl-C and finalizes the recording. FFmpeg, which received the signal too, finalizes its own file and exits, so the frames still in the lookahead buffer are not recorded. petrify reads that file back and keeps it only when it reads back completely, with the warning `FFmpeg received the stop signal too and finalized the recording itself, so it ends early: PATH holds N frames, petrify produced M. The file was read back and is complete up to that point.` If it does not read back, nothing is saved and the run fails. Any other FFmpeg failure fails the run as before. In a test of 10 such group stops, all 10 kept the recording.
- **A signal to petrify alone** (`kill PID`, or Ctrl-C in a terminal, since FFmpeg runs in its own session): the lookahead tail is written and the recording is finalized.
- **A second signal** while the recording is being finalized aborts: the partial file is removed.
- Keep `TimeoutStopSec` at its default (90 s) or higher: on a stop petrify waits up to 60 s for FFmpeg to finish before it kills it and discards the file (from the code, not measured).
- **The older form `petrify INPUT OUTPUT`** writes its final encode directly to `OUTPUT`, so a stop during that encode can leave a truncated file there. Use `petrify run` under a service manager.

### A stalled camera

Without a limit a live run waits for the next frame forever. With `--source-timeout SECONDS` (set to 10 by `--profile cctv` on a live source) a source that delivers no frame for that long, the first frame included, ends the run with an error and exit code 1. The recording made so far is kept, and `Restart=on-failure` starts a new run with a new file. The timeout watches the source only: a detector call that never returns is not covered (see [Before you rely on it](#before-you-rely-on-it)).

### Capacity

Measured with the DNN person detector on the 1080p demo clip, on a 12 core machine. One machine and one clip: treat these as an order of magnitude, not a guarantee.

| Processes at once | Frames per second, each |
|---|---|
| 1 | about 23 to 25 |
| 2 | about 8.5 to 10 |
| 4 | about 6.5 to 7.5 |

Two processes together deliver less than one alone, because the video decoder and the FFmpeg encoder also use threads, and `--threads` does not limit the encoder. A live source faster than its process drops frames (see [Sources](#sources)). Per-detector costs are under [Performance](#performance).

---

## Before you rely on it

Known limitations, in particular for live use:

1. **Live paths are unverified on real hardware.** Webcams (`camera:N`), stream URLs, `--virtualcam`, and `--preview` were developed and tested against test doubles. They have not been verified with real cameras, streams, virtual camera drivers, or windows.
2. **An online background can absorb a person.** With `--background running` or `calibrate`, a person in view while the background is learned can be absorbed into it and then shown on frozen frames after they leave. The person check on background samples reduces this, but by default it uses the HOG pedestrian detector, which often misses a seated person framed from the chest up (the usual webcam view). `--background-check` can add the face and DNN person checks. Start with the scene empty. Do not rely on erase mode on a webcam for privacy yet.
3. **The model-based detectors have measured limits.** The DNN person detector misses people smaller than about 110 px tall on a 1080p frame and replaces HOG rather than adding to it. The segmenter misses small, distant people and marks parts of empty scenes. No built-in mode uses faces or a mask; the `shadow` plugin does, and its [README](plugins/petrify-shadow/README.md) lists its own measured limits. See [Limits, before you choose one](#limits-before-you-choose-one).
4. **A plugin mode is trusted code.** It runs in the petrify process and receives the unprocessed frames. petrify does not sandbox or vet it.
5. **The timestamp ROI is shown live.** Anything inside the ROI region is not anonymized, by design.
6. **A slow entry at the frame edge can show through.** A subject entering slowly at the edge of the frame can be partly visible for a few frames before it is detected, more so with a small `--lookahead` (the `camera:N` default is 3).
7. **A lasting lighting change freezes an online background.** With `running` or `calibrate` and the background differ on (the default), a lasting change in lighting makes every frame differ from the learned background, so the output freezes and keeps showing the old background.
8. **Golden parity tests need demo clips that are not in the repository.** Without `data/camera.mp4` and `data/camera2.mp4` they are skipped.
9. **The shadow mode can show distant people.** It does not use the background differ, and the DNN person detector found no person smaller than about 110 px tall on a 1080p frame. Running the DNN on tiles of the frame was tried and not kept: it cost 3.3 times as much and still found nobody under 110 px. A distant person that no detector localizes is shown whenever the frame has another region to draw (see the [plugin README](plugins/petrify-shadow/README.md#coverage-rules)).
10. **Every detector runs on every processed frame.** There is no option to detect on every Nth frame only. When the detectors are slower than a live source, frames are dropped (see [Performance](#performance)), so the output frame rate falls.
11. **A hung detector stalls the output.** `--frame-budget-ms` is checked after a detector call returns, and `--source-timeout` watches the source only, so a detector call that never returns stops the output until the process is stopped.
12. **A stop can truncate the older form's output.** `petrify INPUT OUTPUT` writes its final encode directly to `OUTPUT`, so a stop during that encode can leave a truncated file there. `petrify run --output` writes to a partial file and moves it into place only when the encode succeeds.

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
         ├── ROI overlay: live timestamp pasted from the real frame on every frame
         │            (erase allows it; a mode that does not allow the overlay
         │            skips this stage), so the timestamp keeps ticking
         │            regardless of freeze state
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

In `petrify run`, `--person-detector dnn` puts the NanoDet-Plus person detector in the HOG slot. It is slower and misses small, distant people. See [Model-based detectors](#model-based-detectors).

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

A 25 fps source leaves 40 ms per frame for every detector together, plus rendering and writing. A `shadow` run with the DNN person detector and faces at 1080p ran at about 12 to 13 frames per second. For several processes on one machine, see [Capacity](#capacity).

The DNN person detector resizes a large frame for the model one axis at a time, which gives the same bytes as a single resize and saved about 9.5 ms per 1080p frame. It is used only for the frame sizes where the two were shown to be byte-identical; every other size keeps the single resize.

To see where the time goes on your own setup, the web page lists the time of each detector and refiner, and every `petrify run` logs one line after the summary, in this form (each `N` a time to one decimal):

```
[INFO] petrify.pipeline: Detector and refiner times (moving average): MotionDetector N ms, PersonDnnDetector N ms, FaceYuNetDetector N ms, TrackerRefiner (persons) N ms, TrackerRefiner (faces) N ms
```

Each label is the class name, with the detection kind for the two tracker refiners, in run order; a component never sampled shows `n/a`. The times are moving averages that weight recent frames most, so on a long run they describe its end. The line is left out when the run had no detector or refiner.

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
