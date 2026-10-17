# camdist: the basic commands

Measure an object's size with one or two cameras in the checkerboard room. Everything is `camdist.py`. The long guide (install details, options, training) is `README_full.md`.

## Once

```
cd ~/Downloads/4UNI/5UNI/grad_project/camdist
source ../.venv/bin/activate
python camdist.py selfcheck          # 17 tests, about a minute. Every other command needs it to have passed
```

**Thonny:** open `thonny_run.py` and press Run (F5), then answer in the Shell panel. It needs these packages (Tools, Manage packages): `numpy scipy opencv-python pillow requests pyserial matplotlib`. Choice 1 measures the sample photo with no models and no hardware.

## Run it

| Camera | Command |
|---|---|
| a photo | `python camdist.py room --image empty_room.jpg --out annotated.jpg` |
| the phone (Android app) | `python camdist.py room --url phone --name` |
| the phone + the ESP32-CAM plugged into it (the rig) | the same command, see below |
| the ESP32-CAM on the laptop's USB (`/dev/ttyUSB0`) | `python camdist.py room --url /dev/ttyUSB0 --name` |
| a USB webcam | `python camdist.py room --url 0 --name` |

Windows: write the ESP port as `serial:COM3`. Linux: your user must be in the `dialout` (Arch: `uucp`) group.

Keys in the window: **space** log the measurement, **m** report, **k** CAD model, **s** change shape, **r** reset views, **c** calibrate again, **q** quit.

Name the object to store its true size with every log: `--object NAME`. A new name needs its size once, in metres: `--object cream_container --truth 0.0778 0.0778 0.0562`.

## The phone

1. Phone and laptop on the same Wi-Fi, VPN off on the phone. Allow TCP 8080 and UDP 8081 on the laptop.
2. Install the app once: `cd ../camdist_app && ./build.sh install` (phone connected with adb).
3. Start the laptop command above, open the app, allow the camera. The top left shows `laptop 192.168.x.x` and the frame rate. Tap an object to select it.
4. Buttons on the right: Stream ON/OFF, ESP connect, ESP turn, ESP mirror, ESP light, Log, Mesh, Report, CAD, Clear, Reset, Shape (the column scrolls).

## The ESP32-CAM on the phone (both cameras)

1. Plug the ESP's USB adapter into the phone's USB-C port, allow the permission, tap **ESP connect**.
2. Keep **ESP mirror** ON (the board's picture is mirrored). **ESP light** switches its flash LED. **ESP turn** rotates the picture.
3. Run `python camdist.py room --url phone --name --object NAME`. When the laptop prints `second camera: the ESP32-CAM relayed by the phone app` the window splits in two. Point the ESP at the room. Log with space: both frames are stored (the log is labelled `rig`).

## The ESP32-CAM on its own

Flash once (board in boot mode, GPIO0 to GND while resetting, then remove the jumper and reset):

```
cd ../esp32cam
arduino-cli compile --fqbn esp32:esp32:esp32cam SerialCam
arduino-cli upload  --fqbn esp32:esp32:esp32cam -p /dev/ttyUSB0 SerialCam
```

Then `python camdist.py room --url /dev/ttyUSB0`. Add `--denoise` to clean its picture (optional, it does not improve the sizes).

## Train and check

Three separate datasets: the phone, the ESP32-CAM alone, and the rig (both cameras).

```
python camdist.py log-tags                                               # what each log is
python camdist.py train-corrector --room --min-views 1                   # phone
python camdist.py train-corrector --room --camera esp32 --min-views 1    # ESP32-CAM alone
python camdist.py train-corrector --room --camera rig --min-views 1      # both (needs 3 objects)
python camdist.py rig-eval --corrected                                   # phone vs ESP vs both, error in mm
python camdist.py denoise-train ; python camdist.py denoise-eval         # the denoiser
python camdist.py room --url phone --corrector                           # measure with the trained corrector
```

A corrector only changes the sizes it improves by 10 % on objects it never saw.
