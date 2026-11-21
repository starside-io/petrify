"""Tests for local model lookup and integrity checking.

Every model file here is a tiny fake payload written to a temp directory, with
a spec built from its real SHA-256. Nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from petrify import models
from petrify.models import (
    KNOWN_MODELS,
    MODEL_DIR_ENV,
    ModelError,
    ModelIntegrityError,
    ModelNotFoundError,
    ModelSpec,
    load_model_bytes,
    model_dirs,
    user_model_dir,
)

PAYLOAD = b"fake onnx model bytes\x00\x01\x02"
FILENAME = "fake_model.onnx"


def _spec(payload: bytes = PAYLOAD, **overrides: object) -> ModelSpec:
    fields: dict[str, object] = {
        "key": "fake_model",
        "filename": FILENAME,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_url": "https://example.invalid/fake_model.onnx",
        "license": "Apache-2.0",
    }
    fields.update(overrides)
    return ModelSpec(**fields)  # type: ignore[arg-type]


def _place(directory: Path, payload: bytes = PAYLOAD, name: str = FILENAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_bytes(payload)
    return target


def _same_size_other(payload: bytes = PAYLOAD) -> bytes:
    return bytes(b ^ 0xFF for b in payload)


# ----------------------------------------------------------------------
# KNOWN_MODELS
# ----------------------------------------------------------------------


def test_known_models_is_empty_and_read_only() -> None:
    assert len(KNOWN_MODELS) == 0
    with pytest.raises(TypeError):
        KNOWN_MODELS["x"] = _spec()  # type: ignore[index]


# ----------------------------------------------------------------------
# ModelSpec validation
# ----------------------------------------------------------------------


def test_valid_spec_constructs() -> None:
    spec = _spec()
    assert spec.filename == FILENAME
    assert spec.size == len(PAYLOAD)


@pytest.mark.parametrize(
    "sha256",
    ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, " " + "a" * 63, "a" * 64 + "\n"],
)
def test_spec_rejects_bad_sha256(sha256: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        _spec(sha256=sha256)


@pytest.mark.parametrize(
    "filename",
    ["", ".", "..", "../x.onnx", "a/b.onnx", "/abs.onnx", "a\\b.onnx", "C:x.onnx", "a\x00b"],
)
def test_spec_rejects_non_bare_filename(filename: str) -> None:
    with pytest.raises(ValueError, match="filename"):
        _spec(filename=filename)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "10"])
def test_spec_rejects_bad_size(size: object) -> None:
    with pytest.raises(ValueError, match="size"):
        _spec(size=size)


def test_spec_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key"):
        _spec(key="")


# ----------------------------------------------------------------------
# user_model_dir
# ----------------------------------------------------------------------


def test_user_model_dir_macos() -> None:
    got = user_model_dir({"HOME": "/Users/alice"}, platform="darwin")
    assert got == Path("/Users/alice/Library/Application Support/petrify/models")


def test_user_model_dir_linux_xdg_absolute() -> None:
    env = {"HOME": "/home/alice", "XDG_DATA_HOME": "/data/alice"}
    assert user_model_dir(env, platform="linux") == Path("/data/alice/petrify/models")


def test_user_model_dir_linux_default() -> None:
    env = {"HOME": "/home/alice"}
    assert user_model_dir(env, platform="linux") == Path("/home/alice/.local/share/petrify/models")


@pytest.mark.parametrize("xdg", ["", "relative/data"])
def test_user_model_dir_linux_ignores_empty_or_relative_xdg(xdg: str) -> None:
    env = {"HOME": "/home/alice", "XDG_DATA_HOME": xdg}
    assert user_model_dir(env, platform="linux") == Path("/home/alice/.local/share/petrify/models")


def test_user_model_dir_other_posix_uses_xdg_rules() -> None:
    env = {"HOME": "/home/bob"}
    assert user_model_dir(env, platform="freebsd14") == Path(
        "/home/bob/.local/share/petrify/models"
    )


def test_user_model_dir_is_not_a_cache_dir() -> None:
    for platform in ("darwin", "linux"):
        got = user_model_dir({"HOME": "/h", "XDG_CACHE_HOME": "/cache"}, platform=platform)
        assert "cache" not in str(got).lower()


def test_user_model_dir_windows_localappdata() -> None:
    local = "C:\\Users\\alice\\AppData\\Local"
    got = user_model_dir({"LOCALAPPDATA": local}, platform="win32")
    assert got == Path(local) / "petrify" / "models"


def test_user_model_dir_windows_falls_back_to_userprofile() -> None:
    profile = "C:\\Users\\alice"
    got = user_model_dir({"USERPROFILE": profile}, platform="win32")
    assert got == Path(profile) / "AppData" / "Local" / "petrify" / "models"


def test_user_model_dir_windows_falls_back_to_home_when_unset() -> None:
    got = user_model_dir({}, platform="win32")
    assert got == Path.home() / "AppData" / "Local" / "petrify" / "models"


def test_user_model_dir_does_not_create_directory(tmp_path: Path) -> None:
    for platform in ("darwin", "linux"):
        got = user_model_dir({"HOME": str(tmp_path)}, platform=platform)
        assert not got.exists()
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# model_dirs
# ----------------------------------------------------------------------


def test_model_dirs_order_explicit_env_user(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home"), MODEL_DIR_ENV: str(tmp_path / "env")}
    got = model_dirs(tmp_path / "explicit", env)
    assert got == (tmp_path / "explicit", tmp_path / "env", user_model_dir(env))


def test_model_dirs_without_explicit_or_env(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home")}
    assert model_dirs(None, env) == (user_model_dir(env),)
    assert model_dirs(None, {**env, MODEL_DIR_ENV: ""}) == (user_model_dir(env),)


def test_model_dirs_accepts_str(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home")}
    assert model_dirs(str(tmp_path / "x"), env)[0] == tmp_path / "x"


def test_model_dirs_drops_duplicates_keeping_order(tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home")}
    user = user_model_dir(env)
    env[MODEL_DIR_ENV] = str(tmp_path / "shared")
    assert model_dirs(tmp_path / "shared", env) == (tmp_path / "shared", user)
    env[MODEL_DIR_ENV] = str(user)
    assert model_dirs(tmp_path / "a", env) == (tmp_path / "a", user)


# ----------------------------------------------------------------------
# load_model_bytes: lookup
# ----------------------------------------------------------------------


def test_returns_exact_file_bytes(tmp_path: Path) -> None:
    _place(tmp_path / "d")
    assert load_model_bytes(_spec(), dirs=[tmp_path / "d"]) == PAYLOAD


def test_lookup_order_across_the_three_sources(tmp_path: Path) -> None:
    home = tmp_path / "home"
    env = {"HOME": str(home), MODEL_DIR_ENV: str(tmp_path / "env")}
    explicit = tmp_path / "explicit"
    user_dir = user_model_dir(env)
    payloads = {explicit: b"explicit payload", tmp_path / "env": b"env payload!!", user_dir: b"u"}
    for d, p in payloads.items():
        _place(d, p)

    dirs = model_dirs(explicit, env)
    assert load_model_bytes(_spec(b"explicit payload"), dirs=dirs) == b"explicit payload"

    (explicit / FILENAME).unlink()
    assert load_model_bytes(_spec(b"env payload!!"), dirs=dirs) == b"env payload!!"

    (tmp_path / "env" / FILENAME).unlink()
    assert load_model_bytes(_spec(b"u"), dirs=dirs) == b"u"


def test_explicit_path_wins_and_ignores_dirs(tmp_path: Path) -> None:
    _place(tmp_path / "d")
    other = _place(tmp_path / "elsewhere", PAYLOAD, name="renamed.bin")
    assert load_model_bytes(_spec(), path=other, dirs=[tmp_path / "d"]) == PAYLOAD


def test_explicit_path_missing_does_not_fall_back_to_dirs(tmp_path: Path) -> None:
    _place(tmp_path / "d")
    missing = tmp_path / "nope" / FILENAME
    with pytest.raises(ModelNotFoundError) as info:
        load_model_bytes(_spec(), path=missing, dirs=[tmp_path / "d"])
    msg = str(info.value)
    assert str(missing) in msg
    assert str(tmp_path / "d") not in msg


def test_not_found_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_bytes(_spec(), dirs=[tmp_path / "empty"])
    assert issubclass(ModelNotFoundError, ModelError)


def test_first_found_bad_file_stops_the_search(tmp_path: Path) -> None:
    _place(tmp_path / "first", _same_size_other())
    _place(tmp_path / "second", PAYLOAD)
    with pytest.raises(ModelIntegrityError) as info:
        load_model_bytes(_spec(), dirs=[tmp_path / "first", tmp_path / "second"])
    assert str(tmp_path / "first" / FILENAME) in str(info.value)


def test_first_found_wrong_size_stops_the_search(tmp_path: Path) -> None:
    _place(tmp_path / "first", PAYLOAD + b"extra")
    _place(tmp_path / "second", PAYLOAD)
    with pytest.raises(ModelIntegrityError):
        load_model_bytes(_spec(), dirs=[tmp_path / "first", tmp_path / "second"])


def test_missing_directories_are_skipped(tmp_path: Path) -> None:
    _place(tmp_path / "second")
    dirs = [tmp_path / "missing", tmp_path / "second"]
    assert load_model_bytes(_spec(), dirs=dirs) == PAYLOAD


# ----------------------------------------------------------------------
# load_model_bytes: integrity
# ----------------------------------------------------------------------


def test_hash_mismatch_raises_with_both_hashes(tmp_path: Path) -> None:
    bad = _same_size_other()
    target = _place(tmp_path / "d", bad)
    spec = _spec()
    with pytest.raises(ModelIntegrityError) as info:
        load_model_bytes(spec, dirs=[tmp_path / "d"])
    msg = str(info.value)
    assert str(target) in msg
    assert spec.sha256 in msg
    assert hashlib.sha256(bad).hexdigest() in msg


def test_size_mismatch_refused_before_hashing_a_huge_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "d" / FILENAME
    target.parent.mkdir()
    huge = 8 << 30  # 8 GiB, sparse, so the test itself stays cheap
    with open(target, "wb") as fh:
        fh.truncate(huge)

    def _no_hash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the file was hashed although its size was wrong")

    def _no_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the file was read although its size was wrong")

    monkeypatch.setattr(models.hashlib, "sha256", _no_hash)
    monkeypatch.setattr(models.os, "read", _no_read)
    spec = ModelSpec(
        key="fake_model",
        filename=FILENAME,
        sha256="0" * 64,
        size=len(PAYLOAD),
        source_url="https://example.invalid/m.onnx",
        license="MIT",
    )
    with pytest.raises(ModelIntegrityError) as info:
        load_model_bytes(spec, dirs=[tmp_path / "d"])
    msg = str(info.value)
    assert str(target) in msg
    assert str(len(PAYLOAD)) in msg
    assert str(huge) in msg
    assert spec.sha256 in msg


def test_integrity_error_is_a_model_error() -> None:
    assert issubclass(ModelIntegrityError, ModelError)
    assert issubclass(ModelError, RuntimeError)


# ----------------------------------------------------------------------
# load_model_bytes: other OS errors
# ----------------------------------------------------------------------


def test_directory_in_place_of_file_is_model_error(tmp_path: Path) -> None:
    (tmp_path / "first" / FILENAME).mkdir(parents=True)
    _place(tmp_path / "second")
    with pytest.raises(ModelError) as info:
        load_model_bytes(_spec(), dirs=[tmp_path / "first", tmp_path / "second"])
    assert not isinstance(info.value, ModelNotFoundError)
    assert str(tmp_path / "first" / FILENAME) in str(info.value)


def test_explicit_path_directory_is_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError) as info:
        load_model_bytes(_spec(), path=tmp_path)
    assert not isinstance(info.value, ModelNotFoundError)
    assert str(tmp_path) in str(info.value)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores file permissions"
)
def test_unreadable_file_is_model_error(tmp_path: Path) -> None:
    target = _place(tmp_path / "d")
    target.chmod(0)
    try:
        with pytest.raises(ModelError) as info:
            load_model_bytes(_spec(), dirs=[tmp_path / "d"])
    finally:
        target.chmod(0o600)
    assert type(info.value) is ModelError
    assert str(target) in str(info.value)


# ----------------------------------------------------------------------
# Error message content
# ----------------------------------------------------------------------


def test_not_found_message_is_complete(tmp_path: Path) -> None:
    dirs = [tmp_path / "one", tmp_path / "two", tmp_path / "three"]
    spec = _spec()
    with pytest.raises(ModelNotFoundError) as info:
        load_model_bytes(spec, dirs=dirs)
    msg = str(info.value)
    assert spec.key in msg
    assert spec.filename in msg
    assert spec.source_url in msg
    assert spec.license in msg
    assert spec.sha256 in msg
    assert "never downloads" in msg
    assert "place" in msg
    positions = [msg.index(str(d / FILENAME)) for d in dirs]
    assert positions == sorted(positions)


def test_not_found_message_with_no_locations() -> None:
    with pytest.raises(ModelNotFoundError) as info:
        load_model_bytes(_spec(), dirs=())
    assert "no locations" in str(info.value)

