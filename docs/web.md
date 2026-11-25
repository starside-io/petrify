# The web page

Back to [README](../README.md)

`--web` serves a page on `127.0.0.1` at `--web-port` (default 8765; 0 picks a free port). At start petrify prints the link to the terminal:

```
Web page (this link gives control of this run to whoever has it): http://127.0.0.1:PORT/#token=...
```

To log in with a password instead, set `PETRIFY_WEB_PASSWORD` in the environment before the run (there is no flag, so the password does not show in the process list, and there is no default password):

```
read -rs PETRIFY_WEB_PASSWORD && export PETRIFY_WEB_PASSWORD
petrify run --source camera:0 --web
```

`read -rs` asks for the password without echoing it, so it stays out of the shell history.

petrify then makes no token and prints `Web page (protected by PETRIFY_WEB_PASSWORD): http://127.0.0.1:PORT/`. The browser asks for the password on its own login prompt (any user name is accepted) and remembers it until the browser is quit, so reloading the page works. The page is still served on `127.0.0.1` only. When the variable is unset or empty, the token link is used as before; a value of only spaces is refused.

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
- With `PETRIFY_WEB_PASSWORD` set, every request, the page itself included, needs the password (HTTP Basic), and the token is not accepted. Because the browser sends a remembered password by itself, a request that the browser marks as coming from another site is refused. The password is never printed or logged.
- The unprocessed input feed is never served unless the run was started with `--web-raw-preview`, which shows it next to the processed feed and lists it as a weakened protection.
- The page does not protect against the active mode: a plugin mode runs in the petrify process and receives the unprocessed frames whatever the page shows.
- Text that comes from a plugin (its description, option help, weakened sentences, error messages) is shown as plain text, with control and formatting characters removed and the length capped. Text from package metadata (mode names, package names, versions) is reduced to printable ASCII, other characters shown as `?`, so a name cannot pass for another mode's. Only runtime mode options can be changed, and a numeric one only within the bounds the mode declares.
