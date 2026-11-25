# Running several cameras

Back to [README](../README.md)

Run one `petrify run` process per camera. One process cannot read several cameras. Give each process its own `--output` and, with `--web`, its own `--web-port`. `--threads N` limits the OpenCV worker threads of each process (see [Run options](run.md#run-options)): some OpenCV builds honor only 1, and FFmpeg's encoder threads are not limited.

## A systemd unit

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

Start it with `systemctl enable --now petrify@front`. `$$` and `%%` are systemd's escapes for `$` and `%`, so the shell expands the variables and `date` on every start. Use the full path of `petrify` if it is not on the service's `PATH`, and leave out `--web` if the page is not needed: the page link with its token is printed to the service's log, so whoever can read that log can control the run (from reading the code: the link goes to stderr, which systemd logs; not tested under systemd). With `PETRIFY_WEB_PASSWORD` set (an `Environment=` line, or a line in the `EnvironmentFile=`, kept readable only by the service), no link with a token is logged.

## What a stop does to the recording

- **`systemctl stop`.** systemd sends SIGTERM to every process of the unit, FFmpeg included (the default `KillMode=control-group`). petrify handles it like Ctrl-C and finalizes the recording. FFmpeg, which received the signal too, finalizes its own file and exits, so the frames still in the lookahead buffer are not recorded. petrify reads that file back and keeps it only when it reads back completely, with the warning `FFmpeg received the stop signal too and finalized the recording itself, so it ends early: PATH holds N frames, petrify produced M. The file was read back and is complete up to that point.` If it does not read back, nothing is saved and the run fails. Any other FFmpeg failure fails the run as before. In a test of 10 such group stops, all 10 kept the recording.
- **A signal to petrify alone** (`kill PID`, or Ctrl-C in a terminal, since FFmpeg runs in its own session): the lookahead tail is written and the recording is finalized.
- **A second signal** while the recording is being finalized aborts: the partial file is removed.
- Keep `TimeoutStopSec` at its default (90 s) or higher: on a stop petrify waits up to 60 s for FFmpeg to finish before it kills it and discards the file (from the code, not measured).
- **The older form `petrify INPUT OUTPUT`** writes its final encode directly to `OUTPUT`, so a stop during that encode can leave a truncated file there. Use `petrify run` under a service manager.

## A stalled camera

Without a limit a live run waits for the next frame forever. With `--source-timeout SECONDS` (set to 10 by `--profile cctv` on a live source) a source that delivers no frame for that long, the first frame included, ends the run with an error and exit code 1. The recording made so far is kept, and `Restart=on-failure` starts a new run with a new file. The timeout watches the source only: a detector call that never returns is not covered (see [Before you rely on it](limitations.md)).

## Capacity

Measured with the DNN person detector on the 1080p demo clip, on a 12 core machine. One machine and one clip: treat these as an order of magnitude, not a guarantee.

| Processes at once | Frames per second, each |
|---|---|
| 1 | about 23 to 25 |
| 2 | about 8.5 to 10 |
| 4 | about 6.5 to 7.5 |

Two processes together deliver less than one alone, because the video decoder and the FFmpeg encoder also use threads, and `--threads` does not limit the encoder. A live source faster than its process drops frames (see [Sources](run.md#sources)). Per-detector costs are under [Performance](usage.md#performance).
