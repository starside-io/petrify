# Copyright 2026 petrify contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""Local lookup and SHA-256 integrity checking for ONNX model files.

petrify never downloads anything at run time. Model files are placed on disk
by the user, found here by a fixed search order, and loaded only when their
size and SHA-256 match the values pinned in a :class:`ModelSpec`. The bytes
that were hashed are the bytes returned, so there is no window between the
check and the use.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

MODEL_DIR_ENV = "PETRIFY_MODEL_DIR"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_READ_CHUNK = 1 << 20


class ModelError(RuntimeError):
    """A model file could not be loaded."""


class ModelNotFoundError(ModelError, FileNotFoundError):
    """No model file was found in any searched location."""


class ModelIntegrityError(ModelError):
    """A model file was found but its size or SHA-256 does not match the pinned value."""


@dataclass(frozen=True)
class ModelSpec:
    """Pinned identity of one model file.

    Attributes
    ----------
    key:
        Short identifier used in code and messages, for example
        ``"face_yunet_2023mar"``.
    filename:
        Bare file name looked up in each model directory. It must not contain
        a path separator, a drive colon, or be ``.`` or ``..``.
    sha256:
        Expected SHA-256 of the file, 64 lowercase hex characters.
    size:
        Expected file size in bytes, a positive integer.
    source_url:
        Where the user can obtain the file. Shown in error messages only;
        petrify never fetches it.
    license:
        License of the model file, shown in error messages.

    Raises
    ------
    ValueError
        On construction, if any field is malformed.
    """

    key: str
    filename: str
    sha256: str
    size: int
    source_url: str
    license: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("ModelSpec.key must be a non-empty string")
        _validate_filename(self.filename)
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(
                f"ModelSpec {self.key!r}: sha256 must be 64 lowercase hex characters, "
                f"got {self.sha256!r}"
            )
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError(
                f"ModelSpec {self.key!r}: size must be a positive integer, got {self.size!r}"
            )
        if not isinstance(self.source_url, str):
            raise ValueError(f"ModelSpec {self.key!r}: source_url must be a string")
        if not isinstance(self.license, str):
            raise ValueError(f"ModelSpec {self.key!r}: license must be a string")


def _validate_filename(filename: object) -> None:
    if not isinstance(filename, str) or not filename:
        raise ValueError("ModelSpec.filename must be a non-empty string")
    if filename in (".", ".."):
        raise ValueError(f"ModelSpec.filename must be a bare file name, got {filename!r}")
    for bad in ("/", "\\", ":", "\x00"):
        if bad in filename:
            raise ValueError(
                f"ModelSpec.filename must be a bare file name without {bad!r}, got {filename!r}"
            )


# The pinned model files. They are defined here, not in the detector modules, so
# that listing them needs no OpenCV import; each detector module's MODEL is one of
# these objects. Entries are added once each file has been verified by hand.
FACE_YUNET_MODEL = ModelSpec(
    key="face_yunet_2023mar",
    filename="face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232589,
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    license="MIT",
)
PERSON_DNN_MODEL = ModelSpec(
    key="object_detection_nanodet_2022nov",
    filename="object_detection_nanodet_2022nov.onnx",
    sha256="4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186",
    size=3800954,
    source_url=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/"
        "object_detection_nanodet_2022nov.onnx"
    ),
    license="Apache-2.0",
)
SEGMENT_MODEL = ModelSpec(
    key="human_segmentation_pphumanseg_2023mar",
    filename="human_segmentation_pphumanseg_2023mar.onnx",
    sha256="552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
    size=6163938,
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg",
    license="Apache-2.0",
)

# Every pinned model, by key. Read-only.
KNOWN_MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {spec.key: spec for spec in (FACE_YUNET_MODEL, PERSON_DNN_MODEL, SEGMENT_MODEL)}
)


# Where each model was last loaded from, by key, until the caller takes it with
# pop_loaded_path. Detectors receive only the bytes, so this is how the caller that
# built a detector can report which file it used without reading it again.
_LOADED_PATHS: dict[str, Path] = {}


def pop_loaded_path(spec: ModelSpec) -> Path | None:
    """Return and forget the file *spec* was last loaded from, or None.

    Only a successful :func:`load_model_bytes` records a path.
    """
    return _LOADED_PATHS.pop(spec.key, None)


def _home(env: Mapping[str, str]) -> Path:
    home = env.get("HOME", "")
    if home and Path(home).is_absolute():
        return Path(home)
    return Path.home()


def user_model_dir(env: Mapping[str, str] = os.environ, platform: str = sys.platform) -> Path:
    """Return the per-user data directory that holds model files.

    This is a data directory, not a cache directory: caches can be purged by
    the operating system or cleaning tools, and a purged model would leave
    petrify unable to start. The directory is not created.

    - macOS: ``~/Library/Application Support/petrify/models``
    - Windows: ``%LOCALAPPDATA%\\petrify\\models``, falling back to
      ``%USERPROFILE%\\AppData\\Local\\petrify\\models`` and then to the home
      directory when those are unset
    - Linux and other POSIX: ``$XDG_DATA_HOME/petrify/models`` when the
      variable is an absolute path, else ``~/.local/share/petrify/models``

    Parameters
    ----------
    env:
        Environment to read. Defaults to :data:`os.environ`.
    platform:
        Platform identifier in :data:`sys.platform` form.
    """
    if platform == "darwin":
        return _home(env) / "Library" / "Application Support" / "petrify" / "models"
    if platform == "win32":
        local = env.get("LOCALAPPDATA", "")
        if local and PureWindowsPath(local).is_absolute():
            return Path(local) / "petrify" / "models"
        profile = env.get("USERPROFILE", "")
        if profile and PureWindowsPath(profile).is_absolute():
            return Path(profile) / "AppData" / "Local" / "petrify" / "models"
        return Path.home() / "AppData" / "Local" / "petrify" / "models"
    xdg = env.get("XDG_DATA_HOME", "")
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / "petrify" / "models"
    return _home(env) / ".local" / "share" / "petrify" / "models"


def model_dirs(
    explicit_dir: str | Path | None, env: Mapping[str, str] = os.environ
) -> tuple[Path, ...]:
    """Return the model search directories, in order.

    The order is *explicit_dir* (when given and non-empty), then
    ``$PETRIFY_MODEL_DIR`` (when set and non-empty), then
    :func:`user_model_dir`. A leading ``~`` is expanded. Duplicates are
    dropped, keeping the first occurrence. Nothing on disk is touched.
    """
    candidates: list[Path] = []
    if explicit_dir is not None and str(explicit_dir) != "":
        candidates.append(Path(explicit_dir).expanduser())
    env_dir = env.get(MODEL_DIR_ENV, "")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(user_model_dir(env))

    out: list[Path] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return tuple(out)


def _not_found(spec: ModelSpec, searched: Sequence[Path]) -> ModelNotFoundError:
    if searched:
        locations = "\n".join(f"  {i}. {p}" for i, p in enumerate(searched, start=1))
    else:
        locations = "  (no locations were given)"
    return ModelNotFoundError(
        f"Model {spec.key!r} (file {spec.filename!r}) was not found.\n"
        f"Locations searched, in order:\n{locations}\n"
        "petrify never downloads models. Obtain the file yourself and place it in "
        f"one of the locations above, or point {MODEL_DIR_ENV} at its directory.\n"
        f"Source: {spec.source_url}\n"
        f"License: {spec.license}\n"
        f"Expected size: {spec.size} bytes\n"
        f"Expected SHA-256: {spec.sha256}"
    )


def _integrity_hint(spec: ModelSpec) -> str:
    return (
        f"Replace it with the file published at {spec.source_url}. "
        "petrify only loads model files whose SHA-256 is pinned in its code."
    )


def _read_checked(spec: ModelSpec, path: Path) -> bytes | None:
    """Read and verify *path*. Return ``None`` when nothing exists at *path*."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise ModelError(
            f"Model {spec.key!r}: cannot open {path}: {exc.strerror or exc}"
        ) from exc

    try:
        try:
            st = os.fstat(fd)
        except OSError as exc:
            raise ModelError(
                f"Model {spec.key!r}: cannot read {path}: {exc.strerror or exc}"
            ) from exc
        if not stat.S_ISREG(st.st_mode):
            raise ModelError(
                f"Model {spec.key!r}: {path} exists but is not a regular file. "
                "Remove it and place the model file there instead."
            )
        if st.st_size != spec.size:
            raise ModelIntegrityError(
                f"Model {spec.key!r}: size mismatch for {path}.\n"
                f"Expected size: {spec.size} bytes\n"
                f"Actual size: {st.st_size} bytes\n"
                f"Expected SHA-256: {spec.sha256}\n"
                "The file was refused before reading, so its SHA-256 was not computed.\n"
                + _integrity_hint(spec)
            )

        chunks: list[bytes] = []
        remaining = spec.size + 1  # one extra byte detects a file that grew
        try:
            while remaining > 0:
                chunk = os.read(fd, min(remaining, _READ_CHUNK))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError as exc:
            raise ModelError(
                f"Model {spec.key!r}: cannot read {path}: {exc.strerror or exc}"
            ) from exc
    finally:
        os.close(fd)

    data = b"".join(chunks)
    if len(data) != spec.size:
        raise ModelIntegrityError(
            f"Model {spec.key!r}: size mismatch for {path} while reading.\n"
            f"Expected size: {spec.size} bytes\n"
            f"Bytes read: {len(data)}\n"
            f"Expected SHA-256: {spec.sha256}\n"
            "The file changed while it was being read, so its SHA-256 was not computed.\n"
            + _integrity_hint(spec)
        )
    actual = hashlib.sha256(data).hexdigest()
    if actual != spec.sha256:
        raise ModelIntegrityError(
            f"Model {spec.key!r}: SHA-256 mismatch for {path}.\n"
            f"Expected SHA-256: {spec.sha256}\n"
            f"Actual SHA-256: {actual}\n" + _integrity_hint(spec)
        )
    return data


def load_model_bytes(
    spec: ModelSpec, *, path: str | Path | None = None, dirs: Sequence[Path] = ()
) -> bytes:
    """Find, verify, and return the bytes of the model file described by *spec*.

    When *path* is given only that file is tried and *dirs* is ignored.
    Otherwise each directory in *dirs* is tried in order for
    ``spec.filename``, and the first location where something exists is the
    only one used: a bad file there stops the search instead of falling
    through to a later directory.

    The file is read once. Its on-disk size is checked before reading, so a
    file of the wrong size is never read into memory. The SHA-256 is computed
    on the bytes read, and those same bytes are returned.

    Raises
    ------
    ModelNotFoundError
        If no file exists at any searched location.
    ModelIntegrityError
        If the first file found has the wrong size or SHA-256.
    ModelError
        If the first entry found cannot be read or is not a regular file.
    """
    if path is not None:
        candidates = [Path(path).expanduser()]
    else:
        candidates = [Path(d) / spec.filename for d in dirs]

    for candidate in candidates:
        data = _read_checked(spec, candidate)
        if data is not None:
            _LOADED_PATHS[spec.key] = candidate
            return data
    raise _not_found(spec, candidates)
