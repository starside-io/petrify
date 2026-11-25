# `petrify run`: cameras, streams, and image folders

Back to [README](../README.md)

`petrify INPUT OUTPUT` is unchanged. A second command form, `petrify run --source SPEC`, reads from more kinds of source and writes to one or more outputs in one streaming pass. Run `petrify run --help` for every option.

> The live paths (webcams, stream URLs, the virtual camera, and the preview window) have not been verified on real hardware yet. Read [Before you rely on it](limitations.md) before you depend on them.

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

## Modes and plugins

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

[docs/PLUGINS.md](PLUGINS.md) describes how to write one.

### Shadow mode (plugin)

The repository holds one mode plugin, `petrify-shadow`, in `plugins/petrify-shadow`. It registers the mode `shadow`, which keeps the live frame and covers the people in it: faces with a blur, pixelation, noise, a solid box, or an eye bar, and bodies with a filled silhouette or a filled person box. It is not installed with petrify; install it from the repository root with `python -m pip install --no-deps -e plugins/petrify-shadow`.

```bash
petrify run --source input.mp4 --output out.mp4 --mode shadow \
    --mode-option styles=bodybox,facebox \
    --detector both --person-detector dnn --faces --track
```

Its styles depend on the model-based detectors below, each of which misses people, and the blur and pixelation styles can be partly reversed. Read the [plugin README](../plugins/petrify-shadow/README.md), in particular its limits, before you rely on it.

## Tracking and model files

| Flag | Meaning |
|---|---|
| `--track` | Enable the box tracker: a person or face box is held for a few frames after the detector stops finding it. Only for modes that use boxes; the erase mode does not accept it, and the run stops with an error |
| `--track-hold N` | Frames a tracked box is held after its last detection (implies `--track`). Default: 15 |
| `--model-dir DIR` | Directory searched first for detector model files, before `PETRIFY_MODEL_DIR` and the per-user model directory. Only the [model-based detectors](#model-based-detectors) and their background checks use it |

## Model-based detectors

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

No built-in mode uses faces or a mask: `--faces` and `--segment` only matter to a [plugin mode](PLUGINS.md) that requires faces, landmarks, or a mask, such as [`shadow`](#shadow-mode-plugin). With `erase` they have no effect, and a warning says so.

### Limits, before you choose one

- **DNN person boxes replace HOG, they do not add to it.** With `--person-detector dnn` HOG no longer runs on the frames. The locked background pre-scan still uses HOG.
- **The DNN misses small people.** The frame is scaled down to 416 px for the model, and no person smaller than about 110 px tall on a 1080p frame was found. Wide frames shrink people further. The background differ covers that gap only while it is on: with `--no-bg-diff`, or a mode that does not use it, a warning says that small or distant people can be shown.
- **The DNN is slower than HOG.** Measured on an idle machine, the DNN person detector takes a median of about 33 ms per frame on the 640x480 demo clip, against a few ms with HOG, and about 45 ms at 1080p, so a 25 fps 1080p live source drops frames. On the 640x480 clip erase with `--detector person --person-detector dnn` freezes 611 of 655 frames against 494 with HOG. See [Performance](usage.md#performance) for the other detectors.
- **DNN person boxes are estimates.** Each box is padded by 10 percent of its width and 7 percent of its height on each side, because the model's boxes were measured to fall short of the person. An outstretched arm can still stick out of the box.
- **The segmenter is tuned for close, large people.** It misses small, distant people and marks parts of empty scenes. That is why it is not offered as a background check, and it should not be the only person signal on CCTV footage.
- **The face detector refuses tiny frames.** A frame with a side of 32 px or less is not processed and fails closed (black). One face detector instance is not thread-safe.

### Model files

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

### Startup errors and warnings

petrify refuses to start, before reading any frame, when:

- a threshold flag is given without its detector: `--dnn-score-threshold` without `--person-detector dnn`, `--face-score-threshold` without `--faces`, `--segment-threshold` without `--segment` (usage error, exit code 2)
- `--background-check` is combined with `--no-background-check`, used with a `locked` background, or names a detector other than `hog`, `yunet`, or `dnn` (usage error, exit code 2)
- the mode requires faces, landmarks, or a mask and its detector is not enabled, for example `The mode 'NAME' requires faces but no enabled detector provides it; enable the YuNet face detector with --faces` (exit code 1)
- a mode with the `first_hit` detection policy requires faces, landmarks, or a mask together with another kind (exit code 1; see [docs/PLUGINS.md](PLUGINS.md))
- a model file it needs is missing or fails its size or SHA-256 check (exit code 1)

It logs a warning when `--person-detector dnn` is given with `--detector motion`, when `--faces` or `--segment` is given for a mode that does not use it, when `--background-check` is given for a mode that uses no background, when the DNN person detector runs without the background differ, and when the background sample check has rejected 50 samples in a row while the background is still not ready.

## Sources

| `--source` | Reads |
|---|---|
| A file path, such as `input.mp4` | A video file. A bare number such as `0` is a file name, not a camera |
| A directory, or a glob such as `'frames/*.png'` | Still images in natural name order (`frame2.png` before `frame10.png`), at `--fps` frames per second (default 25) |
| `camera:N` | Webcam number N |
| `rtsp://`, `rtsps://`, `http://`, `https://` URL | A network stream (RTSP, HTTP MJPEG) |

On a live source (`camera:N` or a stream URL) a capture thread keeps only the newest frame, so a slow run drops frames instead of falling behind. By default a live run waits for the next frame without limit; `--source-timeout SECONDS` ends it when the source stalls (see [Run options](#run-options)). Stream URLs are redacted in logs and error messages: at most the scheme, host, and port are shown. Credentials are replaced with `***`, and the path and query are dropped.

For library use, `petrify.sources.PushSession` feeds in-memory frames to an engine one by one and returns the processed frames.

## Outputs

At least one is required, and several can be combined.

| Flag | Output |
|---|---|
| `--output PATH` | A video file, encoded by FFmpeg in one pass through a pipe (no intermediate file). FFmpeg writes to `<name>.partial<suffix>`, which is moved onto `PATH` only when the encode succeeds, so a failed run never leaves a file at `PATH` and never replaces an existing one. For a video file source its audio is copied |
| `--virtualcam` | A virtual camera that video call apps can select. `--virtualcam-device NAME` picks the device (default: the first one). Needs `petrify[virtualcam]`, and OBS 30 or newer on macOS |
| `--preview` | A local window. `q` or Escape ends the run. Needs an OpenCV build with GUI support |
| `--web` | A local web page to watch the processed feed and change settings while the run is in progress. Needs `petrify[web]`. See [The web page](web.md) |

On a live source the run ends with Ctrl-C or SIGTERM (or `q` in the preview window), and the output file is still finalized. Both signals are handled the same way: a first signal that lands while the outputs are opened, a frame is processed, or the outputs are finalized takes effect when that step is done, and a second signal aborts. A live run stopped before its first frame writes no file. On a video file or image folder source a signal aborts the run and no output file is written (exit code 130). If a live run fails, the recording up to that point is kept and the error is logged. [Running several cameras](deployment.md) describes what a stop from a service manager does to the recording.

## Run options

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

## Defaults by source kind

Some defaults depend on the source kind. Every value can be overridden with the flag named in the column.

| Source | `--background` | `--lookahead` | Timestamp ROI |
|---|---|---|---|
| Video file | `locked` | 10 | Auto-detected on the whole clip before processing |
| Image folder or glob | `locked` | 10 | Auto-detected on the first `--roi-warmup` seconds (default 10), then locked |
| Stream URL | `running` | 10 | Auto-detected on the first `--roi-warmup` seconds (default 10), then locked |
| `camera:N` | `running` | 3 | Disabled. `--roi-auto` turns warmup detection on |

`--lookahead` is the run form's name for `--gap-fill`: every frame is delayed by this many frames, and every detection is expanded by as many frames on both sides. A smaller value lowers the latency but lets more of a slowly entering subject through before it is detected. It is capped at 600, because the buffered frames are kept uncompressed. `--gap-fill` itself is not accepted by `petrify run`.

During warmup ROI detection nothing is pasted until the region locks. A detected box larger than `--roi-max-area` of the frame (default 0.15) is discarded as too large to be a timestamp, so nothing is pasted. A webcam has no burned-in timestamp and detection there can lock onto a person, which is why the ROI is off for `camera:N` unless `--roi` or `--roi-auto` is given. A manual `--roi` larger than that cap is accepted, with a warning that the region is shown live.

## Background strategies

| `--background` | How the clean background is found |
|---|---|
| `locked` | Pre-scan the source for one clean frame (video files and image folders only). `--background-frame N` picks the frame |
| `running` | Rolling median of recent frames with no subject |
| `calibrate` | Median of the first `--calibrate-seconds` (default 3), while the scene is empty |

Frames are written black until the background is ready. With `running` or `calibrate`, every frame the background is learned from is first checked, by default with the HOG person detector, and a frame in which it finds a person is not used. `--background-check NAMES` chooses the checkers from `hog`, `yunet` (faces), and `dnn` (person boxes); a frame is not used when any of them finds a person. The list replaces the default, so include `hog` to keep it, for example `--background-check hog,dnn`. `yunet` and `dnn` need their [model files](#model-files). A model checker uses the score threshold of its detector flag, and those flags need the detector enabled for the frames too (`--dnn-score-threshold` needs `--person-detector dnn`, `--face-score-threshold` needs `--faces`), so without them the checker runs at its default threshold. The segmenter is not offered as a checker, because it marks parts of empty scenes and would reject almost every sample. When the checkers reject 50 samples in a row and the background is still not ready, a warning names them. `--no-background-check` turns the check off and is unsafe. At startup petrify logs a warning that a person in view while the background is learned can be absorbed into it. The check reduces this but does not eliminate it: start with the scene empty either way. See [Before you rely on it](limitations.md).

## Fail closed

A frame that cannot be processed safely is written black and counted: a detector error or malformed detector result, a mode or stage error or an output image of the wrong shape or type, the background not ready yet, or a frame over `--frame-budget-ms` (the per-frame time budget for detection plus rendering; no budget by default). The end-of-run summary reports how many frames failed closed.
