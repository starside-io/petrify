# Installation

Back to [README](../README.md)

## 1. System FFmpeg (required)

petrify calls `ffmpeg` via subprocess. Install it first:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows: download from https://ffmpeg.org/download.html and add to PATH
```

## 2. petrify

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

## 3. Optional extras (for `petrify run`)

Some outputs of [`petrify run`](run.md) need extra packages. The `petrify INPUT OUTPUT` form needs none of them.

| Output | Install | Notes |
|---|---|---|
| `--web` (local web page) | `pip install 'petrify[web]'` | Installs FastAPI and uvicorn |
| `--virtualcam` (virtual camera) | `pip install 'petrify[virtualcam]'` | Installs pyvirtualcam. On macOS it also needs OBS 30 or newer, whose virtual camera is the device pyvirtualcam drives |
| `--preview` (local window) | `pip uninstall opencv-python-headless && pip install opencv-python` | petrify depends on the headless OpenCV build, which has no GUI support |

With pipx, add the extras to the install: `pipx install 'petrify[web,virtualcam]'`.

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
