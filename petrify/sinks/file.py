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

"""Video file sinks.

``FileSink`` writes an intermediate mp4v file, then re-encodes it with FFmpeg and
the original audio. ``FfmpegPipeSink`` pipes raw frames to FFmpeg and encodes them
in one step.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np

from petrify.core.config import EncodeConfig
from petrify.core.frame import FrameMeta, SourceInfo
from petrify.encoder import ffmpeg_encode, find_ffmpeg

logger = logging.getLogger(__name__)


class FileSink:
    """Write frames to a temporary mp4v file and re-encode it on ``close``.

    ``abort`` discards the temporary file without encoding. ``close`` and
    ``abort`` are idempotent and do nothing when ``open`` never succeeded.
    """

    critical = True

    def __init__(
        self,
        output_path: str | Path,
        audio_source: str | Path,
        encode: EncodeConfig,
    ) -> None:
        self._output_path = Path(output_path)
        self._audio_source = Path(audio_source)
        self._encode = encode
        self._writer: cv2.VideoWriter | None = None
        self._temp_path: str | None = None

    def open(self, info: SourceInfo) -> None:
        """Create the temporary file and open the intermediate writer."""
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        logger.info("Compositing frames -> %s", temp_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        try:
            writer = cv2.VideoWriter(temp_path, fourcc, info.fps, (info.width, info.height))
        except BaseException:
            os.unlink(temp_path)
            raise
        if not writer.isOpened():
            os.unlink(temp_path)
            raise OSError("cv2.VideoWriter failed to open the intermediate file.")

        self._writer = writer
        self._temp_path = temp_path

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Append one output image."""
        if self._writer is None:
            raise RuntimeError("FileSink is not open")
        self._writer.write(image)

    def close(self) -> None:
        """Release the writer, re-encode with FFmpeg, then remove the temporary file."""
        if self._temp_path is None:
            return
        temp_path = self._temp_path
        self._temp_path = None
        try:
            self._release_writer()
            logger.info("Re-encoding with FFmpeg -> %s", self._output_path)
            ffmpeg_encode(
                video_path=temp_path,
                original_audio_source=self._audio_source,
                output_path=self._output_path,
                codec=self._encode.codec,
                crf=self._encode.crf,
                preset=self._encode.preset,
                hwaccel=self._encode.hwaccel,
            )
        finally:
            _remove_temp(temp_path)

        logger.info("Done. Output written to: %s", self._output_path)

    def abort(self) -> None:
        """Release the writer and remove the temporary file without encoding."""
        if self._temp_path is None:
            return
        temp_path = self._temp_path
        self._temp_path = None
        try:
            self._release_writer()
        finally:
            _remove_temp(temp_path)

    def _release_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.release()


def _remove_temp(temp_path: str) -> None:
    try:
        os.unlink(temp_path)
    except OSError as exc:
        logger.warning("Could not remove temp file %s: %s", temp_path, exc)


# Seconds ``close`` waits for FFmpeg to finish encoding the tail and exit.
_CLOSE_TIMEOUT_S = 60.0
# Seconds ``abort`` (and a failed write) waits after terminate before killing.
_STOP_TIMEOUT_S = 5.0
# How much of FFmpeg's stderr goes into error messages.
_STDERR_TAIL_BYTES = 4096
_STDERR_TAIL_LINES = 20
# Seconds to wait, once FFmpeg has exited after a stop signal, for the same stop to
# reach this run. A service manager signals every process of the service at once,
# so both arrive within milliseconds; FFmpeg stopped on its own never qualifies.
_STOP_SIGNAL_GRACE_S = 2.0
# Seconds the check of a file FFmpeg finalized after a stop signal may take.
_VERIFY_TIMEOUT_S = 120.0
# What FFmpeg prints when it ends early but cleanly because it received a signal:
# it then writes the trailer, so the file is finalized, and exits with code 255.
_SIGNAL_EXIT_RE = re.compile(r"Exiting normally, received signal \d+\.")
_SIGNAL_EXIT_CODE = 255


class FfmpegPipeSink:
    """Encode frames in one step by piping raw BGR frames to an FFmpeg subprocess.

    There is no intermediate file and no second encode. FFmpeg writes to a sibling
    ``<name>.partial<suffix>`` that ``close`` moves onto ``output_path`` only after
    FFmpeg exits with code 0, so a failed or aborted run never leaves a file at
    ``output_path`` and never replaces an existing one.

    ``audio_source`` (optional) is muxed as the output's audio, copied without
    re-encoding; a source without audio is skipped silently.

    ``wallclock_timestamps=True`` stamps each frame with the time FFmpeg receives
    it. Use it for live sources: a stream that drops frames then keeps real time
    instead of being compressed to a constant frame rate. Leave it off for files,
    where every frame is delivered and the source frame rate is the right clock
    (wallclock stamps would follow processing speed instead). The live variant
    passes the stamps through unchanged (``-vsync passthrough``, with a microsecond
    encoder time base), so frames that arrive in a burst are all kept instead of
    being dropped to fit the nominal rate.

    FFmpeg runs in its own session (its own process group on Windows), so a Ctrl-C
    in the terminal reaches petrify only; FFmpeg ends when ``close`` closes its
    stdin, or when ``abort`` terminates it.

    FFmpeg's stderr goes to a temporary file, never to an undrained pipe, so it
    cannot block the pipeline; its tail is quoted in error messages. The command
    line, including the output path, is logged. Paths are passed to FFmpeg as
    absolute paths or prefixed with ``./``, so a name such as ``-y.mp4`` is never
    read as an option.

    ``close`` raises ``OSError`` and installs nothing when no frame was written. If
    ``close`` fails in any way (``KeyboardInterrupt`` included), FFmpeg is stopped
    and the partial file removed before it returns, and a later ``abort`` does
    nothing. ``close`` and ``abort`` are idempotent; ``abort`` is safe before
    ``open`` and after a failed ``open``.

    A service manager (``systemctl stop``, ``docker stop``) may signal FFmpeg too,
    not only petrify. FFmpeg then stops reading, finalizes the partial file itself
    and exits with code 255. ``stop_requested`` (a callable, true once the run
    received a stop signal) lets the sink tell that case from FFmpeg failing: when
    FFmpeg reports that it ended because of a signal and the run was stopped by a
    signal as well, the frames written after that are dropped instead of failing
    the run, and ``close`` installs FFmpeg's file, but only after FFmpeg has read
    it back without an error, and with a warning that gives how many frames it
    holds. Without ``stop_requested``, or when FFmpeg ended for any other reason,
    nothing changes: the write or the close fails and the partial file is removed.

    Starting FFmpeg and stopping it are not interrupted by SIGINT or SIGTERM on
    the main thread: a signal received then is delivered once the section is over,
    so FFmpeg is never left running untracked and no stray partial file remains.
    """

    critical = True

    def __init__(
        self,
        output_path: str | Path,
        encode: EncodeConfig,
        *,
        audio_source: str | Path | None = None,
        wallclock_timestamps: bool = False,
        popen: Callable[..., Any] = subprocess.Popen,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._output_path = _validate_output_path(output_path)
        self._partial_path = self._output_path.with_name(
            f"{self._output_path.stem}.partial{self._output_path.suffix}"
        )
        self._encode = encode
        self._audio_source = None if audio_source is None else Path(audio_source)
        self._wallclock = wallclock_timestamps
        self._popen = popen
        self._stop_requested = stop_requested
        self._ffmpeg_bin = ""
        self._proc: Any = None
        self._stderr: IO[bytes] | None = None
        self._shape: tuple[int, int, int] | None = None
        self._frames = 0
        # Set once FFmpeg was stopped by the signal that also stopped the run.
        self._stopped_by_signal = False
        self._dropped = 0

    @property
    def partial_path(self) -> Path:
        """Where FFmpeg writes until ``close`` succeeds."""
        return self._partial_path

    def build_command(self, ffmpeg_bin: str, info: SourceInfo) -> list[str]:
        """Return the FFmpeg command line for a stream described by ``info``."""
        cmd: list[str] = [ffmpeg_bin, "-y", "-hide_banner", "-nostats"]
        if self._encode.hwaccel:
            cmd += ["-hwaccel", self._encode.hwaccel]
        # An input -r replaces the wallclock stamps with a constant rate, so the live
        # variant gives the nominal rate with -framerate instead.
        rate_flag = "-r"
        if self._wallclock:
            cmd += ["-use_wallclock_as_timestamps", "1"]
            rate_flag = "-framerate"
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{info.width}x{info.height}",
            rate_flag, _format_fps(info.fps),
            "-i", "pipe:0",
        ]
        if self._audio_source is not None:
            cmd += ["-i", _ffmpeg_path(self._audio_source)]
        cmd += ["-map", "0:v:0"]
        if self._audio_source is not None:
            cmd += ["-map", "1:a?"]
        cmd += [
            "-c:v", self._encode.codec,
            "-crf", str(self._encode.crf),
            "-preset", self._encode.preset,
            "-pix_fmt", "yuv420p",
        ]
        if self._wallclock:
            # Keep every frame with its wallclock stamp. The default rate handling
            # rounds the stamps to the nominal rate and drops frames that land on the
            # same tick, which loses frames delivered in a burst. "-vsync" (deprecated
            # in favour of "-fps_mode" since FFmpeg 5.1, still accepted) is the
            # spelling older releases understand too; the microsecond encoder time
            # base keeps stamps a few milliseconds apart distinct.
            cmd += ["-vsync", "passthrough", "-enc_time_base", "1:1000000"]
        if self._audio_source is not None:
            cmd += ["-c:a", "copy"]
        cmd.append(_ffmpeg_path(self._partial_path))
        return cmd

    def open(self, info: SourceInfo) -> None:
        """Start FFmpeg reading raw frames from its stdin."""
        if self._proc is not None:
            raise RuntimeError("FfmpegPipeSink is already open")
        if info.width <= 0 or info.height <= 0:
            raise ValueError(f"Invalid frame size {info.width}x{info.height}")
        ffmpeg_bin = find_ffmpeg()
        cmd = self.build_command(ffmpeg_bin, info)
        logger.info("FFmpeg command: %s", " ".join(cmd))

        stderr = tempfile.TemporaryFile()
        # A KeyboardInterrupt between the start of FFmpeg and storing its handle
        # would leave FFmpeg running with nothing to stop it.
        with _deferred_stop_signals():
            try:
                proc = self._popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    # Unbuffered: a buffered pipe holds back the tail of each frame
                    # until the next write, so FFmpeg would stamp it late in wallclock
                    # mode.
                    bufsize=0,
                    **_detached_kwargs(),
                )
            except BaseException:
                stderr.close()
                raise
            self._proc = proc
            self._stderr = stderr
        self._ffmpeg_bin = ffmpeg_bin
        self._shape = (info.height, info.width, 3)
        self._frames = 0
        self._stopped_by_signal = False
        self._dropped = 0

    def write(self, image: np.ndarray, meta: FrameMeta) -> None:
        """Send one BGR uint8 image to FFmpeg."""
        proc = self._proc
        if proc is None or self._shape is None:
            raise RuntimeError("FfmpegPipeSink is not open")
        if image.shape != self._shape or image.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 image of shape {self._shape}, "
                f"got {image.dtype} {image.shape}"
            )
        if self._stopped_by_signal:
            self._dropped += 1
            return
        if proc.poll() is not None:
            if self._stopped_with_the_run():
                self._dropped += 1
                return
            raise OSError(self._failure("FFmpeg exited while frames were still being written"))
        try:
            # An unbuffered pipe may accept part of the frame; write the rest.
            view = memoryview(np.ascontiguousarray(image)).cast("B")
            while view:
                view = view[proc.stdin.write(view) :]
        except (OSError, ValueError) as exc:  # BrokenPipeError is an OSError.
            self._wait_quietly(_STOP_TIMEOUT_S)
            if self._stopped_with_the_run():
                self._dropped += 1  # This frame did not reach FFmpeg completely.
                return
            raise OSError(self._failure("Writing a frame to FFmpeg failed")) from exc
        self._frames += 1

    def close(self) -> None:
        """Finish the encode; on success move the partial file onto ``output_path``.

        On any failure, FFmpeg is stopped and the partial file removed before the
        error propagates; the existing ``output_path``, if any, is left untouched.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            if self._frames == 0:
                raise OSError(
                    f"No frames were written; nothing was saved to {self._output_path}"
                )
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
            try:
                proc.wait(timeout=_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                self._wait_quietly(_STOP_TIMEOUT_S)
                raise OSError(
                    self._failure(f"FFmpeg did not exit within {_CLOSE_TIMEOUT_S:g} s and was killed")
                ) from None
            kept = None
            if proc.returncode != 0:
                if not self._stopped_with_the_run():
                    raise OSError(self._failure("FFmpeg failed"))
                kept = self._check_signal_finalized_file()
            os.replace(self._partial_path, self._output_path)
        except BaseException:
            # Clean up before the handle is released, so nothing is left for abort.
            self._discard(proc)
            raise
        finally:
            self._release()
        if kept is None:
            logger.info("Done. Output written to: %s", self._output_path)
            return
        logger.warning(
            "FFmpeg received the stop signal too and finalized the recording itself, "
            "so it ends early: %s holds %d frames, petrify produced %d. The file was "
            "read back and is complete up to that point.",
            self._output_path,
            kept,
            self._frames + self._dropped,
        )

    def abort(self) -> None:
        """Stop FFmpeg and remove the partial file. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        try:
            self._discard(proc)
        finally:
            self._release()

    def _stopped_with_the_run(self) -> bool:
        """Return True, and drop writes from now on, when FFmpeg ended with the run's stop.

        That is: FFmpeg has exited, it reports that it ended because of a signal
        (so it finalized its file), and this run received a stop signal too,
        possibly a moment later (see ``_STOP_SIGNAL_GRACE_S``). Anything else,
        FFmpeg failing on its own included, returns False.
        """
        if self._stopped_by_signal:
            return True
        proc = self._proc
        if self._stop_requested is None or proc is None:
            return False
        code = proc.poll()
        if code != _SIGNAL_EXIT_CODE or not _SIGNAL_EXIT_RE.search(self._stderr_tail()):
            return False
        deadline = time.monotonic() + _STOP_SIGNAL_GRACE_S
        while not self._stop_requested():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        self._stopped_by_signal = True
        logger.warning(
            "FFmpeg received the stop signal and finalized its file; frames processed "
            "from now on are not recorded"
        )
        return True

    def _check_signal_finalized_file(self) -> int:
        """Return the frame count of the file FFmpeg finalized after a stop signal.

        FFmpeg reads the file back (stream copy to the null muxer); any error, a
        timeout, or no frame at all raises ``OSError``, so a truncated or unplayable
        file is never installed.
        """
        frames = _count_readable_frames(self._ffmpeg_bin, self._partial_path)
        if frames is None or frames < 1:
            raise OSError(
                self._failure(
                    "FFmpeg was stopped by a signal and its file could not be read back "
                    "completely; nothing was saved"
                )
            )
        return frames

    def _discard(self, proc: Any) -> None:
        """Stop FFmpeg (terminate, then kill if needed) and remove the partial file.

        A second Ctrl-C while FFmpeg is being stopped would otherwise skip the kill
        or the removal of the partial file; it is delivered once both are done.
        """
        with _deferred_stop_signals():
            self._stop_and_remove(proc)

    def _stop_and_remove(self, proc: Any) -> None:
        try:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=_STOP_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=_STOP_TIMEOUT_S)
            except Exception as exc:
                logger.warning("Could not stop FFmpeg cleanly: %r", exc)
            except BaseException:
                # Interrupted while stopping it: do not leave FFmpeg running.
                with contextlib.suppress(Exception):
                    proc.kill()
                raise
        finally:
            _remove_quietly(self._partial_path)

    def _failure(self, what: str) -> str:
        """Build an error message with FFmpeg's exit code and stderr tail."""
        code = None if self._proc is None else self._proc.poll()
        tail = self._stderr_tail()
        message = f"{what} (exit code {code})"
        return f"{message}:\n{tail}" if tail else message

    def _stderr_tail(self) -> str:
        stderr = self._stderr
        if stderr is None:
            return ""
        try:
            stderr.flush()
            size = stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, size - _STDERR_TAIL_BYTES))
            data = stderr.read()
        except (OSError, ValueError):
            return ""
        lines = data.decode("utf-8", errors="replace").strip().splitlines()
        return "\n".join(lines[-_STDERR_TAIL_LINES:])

    def _wait_quietly(self, timeout: float) -> None:
        """Give an exiting FFmpeg a moment so its exit code and stderr are complete."""
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=timeout)

    def _release(self) -> None:
        self._proc = None
        self._shape = None
        stderr, self._stderr = self._stderr, None
        if stderr is not None:
            with contextlib.suppress(OSError):
                stderr.close()


def _format_fps(fps: float) -> str:
    """Format a frame rate for FFmpeg, keeping NTSC rates exact (30000/1001)."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Frame rate must be a positive number, got {fps!r}")
    return str(Fraction(fps).limit_denominator(1001))


def _validate_output_path(output_path: str | Path) -> Path:
    """Return ``output_path`` as a Path; reject paths that name no file."""
    raw = os.fspath(output_path)
    path = Path(raw)
    separators = tuple(s for s in (os.sep, os.altsep, "/") if s)
    if path.name in ("", ".", "..") or raw.endswith(separators):
        raise ValueError(f"Output path {raw!r} has no file name")
    if path.is_dir():
        raise ValueError(f"Output path {raw!r} is an existing directory")
    return path


def _ffmpeg_path(path: Path) -> str:
    """Spell ``path`` so FFmpeg cannot read it as an option or a protocol."""
    return str(path) if path.is_absolute() else os.path.join(os.curdir, path)


def _count_readable_frames(ffmpeg_bin: str, path: Path) -> int | None:
    """Read ``path`` back with FFmpeg; return its video frame count, or None on any error."""
    cmd = [
        ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "error",
        "-i", _ffmpeg_path(path),
        "-map", "0:v:0", "-c", "copy", "-f", "null", "-",
        "-progress", "pipe:1", "-nostats",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_VERIFY_TIMEOUT_S,
            check=False,
            **_detached_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not read back %s: %r", path, exc)
        return None
    if result.returncode != 0 or result.stderr.strip():
        logger.warning("Reading back %s failed: %s", path, result.stderr.strip()[-500:])
        return None
    counts = [
        line[len("frame=") :] for line in result.stdout.splitlines() if line.startswith("frame=")
    ]
    try:
        return int(counts[-1]) if counts else None
    except ValueError:
        return None


def _stop_signals() -> list[int]:
    """SIGINT and SIGTERM (when the platform defines it)."""
    signals = [signal.SIGINT]
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signals.append(sigterm)
    return signals


@contextlib.contextmanager
def _deferred_stop_signals() -> Iterator[None]:
    """Hold SIGINT and SIGTERM during a short section, then deliver them.

    Python runs signal handlers between bytecodes of the main thread, so a
    ``KeyboardInterrupt`` can land between starting FFmpeg and storing its handle,
    or between stopping FFmpeg and removing its file. ``pthread_sigmask`` cannot be
    used for this: FFmpeg would inherit the blocked mask and could no longer be
    stopped by a signal. Instead a handler that only records the signal is
    installed for the section, and every recorded signal is raised again once the
    previous handlers are back, so it is handled as if it arrived just after.
    Off the main thread, where handlers cannot be set, the section runs as is; a
    signal whose handler was not set from Python is left alone.

    A restored handler may raise as soon as it is back (a signal arriving right
    then), before the other one is restored. The recording handler therefore
    forwards to the previous handler once the section is over, so a recording
    handler left in place behaves exactly like the one it replaced.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held: list[int] = []
    previous: dict[int, Any] = {}
    section_open = True

    def hold(signum: int, frame: Any) -> None:
        if section_open:
            held.append(signum)
            return
        _forward(signum, previous[signum], frame)

    try:
        for signum in _stop_signals():
            current = signal.getsignal(signum)
            if current is not None:
                previous[signum] = current
                signal.signal(signum, hold)
        yield
    finally:
        section_open = False
        error: BaseException | None = None
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except BaseException as exc:  # noqa: BLE001 (raised below, after the others)
                if error is None:
                    error = exc
        if error is not None:
            raise error
        for signum in held:
            signal.raise_signal(signum)


def _forward(signum: int, handler: Any, frame: Any) -> None:
    """Handle ``signum`` as ``handler`` (a callable, ``SIG_DFL`` or ``SIG_IGN``) would."""
    if callable(handler):
        handler(signum, frame)
    elif handler == signal.SIG_DFL:
        signal.signal(signum, handler)
        signal.raise_signal(signum)


def _detached_kwargs() -> dict[str, Any]:
    """Popen arguments that keep a terminal's Ctrl-C (SIGINT to the group) off FFmpeg."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove partial file %s: %s", path, exc)
