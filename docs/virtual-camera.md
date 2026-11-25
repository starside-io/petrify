# Use petrify as your camera in a video call

Back to [README](../README.md)

`--virtualcam` publishes the processed feed as a webcam that Teams, Zoom, Google Meet, and other call apps can select. This path has not yet been verified with a real camera and a real call app (see [Before you rely on it](limitations.md)): the steps below follow from the code and from how pyvirtualcam works.

## Why OBS on macOS

On macOS an app can appear in the camera list of Teams and similar apps only through a signed camera extension. petrify does not ship one. pyvirtualcam sends the frames to the camera extension that OBS Studio installs, so OBS is needed for the one-time setup. It does not need to be open during calls.

## One-time setup on macOS

1. Install OBS Studio 30 or newer from obsproject.com.
2. Open OBS once and click **Start Virtual Camera**. macOS asks to allow the OBS camera extension.
3. Approve it in System Settings, under General > Login Items & Extensions > Camera Extensions (the wording can differ between macOS versions).
4. Click **Stop Virtual Camera** and quit OBS.
5. Install the extra: `pip install 'petrify[virtualcam]'`, or with pipx `pipx install 'petrify[virtualcam]'` (add `web` to the list for `--web`).

## Run it

```bash
# Webcam to the virtual camera, with a local window to watch the result
petrify run --source camera:0 --virtualcam --preview
```

Add `--web` instead of, or next to, `--preview` to watch the feed and change settings on the [web page](web.md) during the call. At start petrify logs `Virtual camera started: DEVICE (WxH at N fps)`. The virtual camera uses the webcam's frame size and frame rate as the driver reports them, or 30 fps when the driver reports none. `--virtualcam-device NAME` picks a device when there are several (default: the first one).

With the default `erase` mode the call shows the empty room, with you frozen out. On `camera:N` the background is learned while the run is in progress, and frames are black until it is ready, so start with the scene empty. If you stay in view, the person check can reject every background sample and the output can stay black (see [Background strategies](run.md#background-strategies)). To stay in the picture with your face covered instead, the [shadow plugin](run.md#shadow-mode-plugin) is an option, for example `--mode shadow --faces`, which needs the plugin and the YuNet [model file](run.md#model-files). Read its limits first: the face detector misses faces.

## Select it in the call app

- **Teams:** Settings > Devices > Camera, then pick **OBS Virtual Camera**.
- **Zoom, Google Meet, and others:** pick **OBS Virtual Camera** in the app's video or camera settings.

Start petrify before you join the call. If the call app was started first, select the camera again so the app picks up the feed.

## Things to know

- petrify holds the real webcam while it runs. Select the virtual camera in the call app, not the webcam.
- macOS asks for camera permission for the app that runs petrify (Terminal, iTerm, or your editor). Allow it in System Settings > Privacy & Security > Camera.
- Do not run the virtual camera from OBS at the same time: both would drive the same device.
- When petrify stops (Ctrl-C, `q` in the preview window, or the stop button on the web page), it sends one black frame and closes the virtual camera, so the call does not keep showing the last processed frame. What the call app shows after that depends on the app. A frame that [fails closed](run.md#fail-closed) is sent black as well.
- The virtual camera is a critical output: if it cannot be opened or a frame cannot be sent, the run ends with an error. Without pyvirtualcam installed the error starts with `The virtual camera needs the optional pyvirtualcam package`.

## Windows and Linux

- **Windows:** install OBS Studio. pyvirtualcam drives its virtual camera in the same way, and the call app lists it as **OBS Virtual Camera**. There is no extension to approve.
- **Linux:** pyvirtualcam drives a v4l2loopback device. Install and load the v4l2loopback kernel module (for example `sudo modprobe v4l2loopback`), then run petrify as above and pick the loopback device in the call app. With several video devices, name it with `--virtualcam-device`, for example `--virtualcam-device /dev/video10`.
