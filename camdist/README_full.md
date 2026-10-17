# camdist

Measure the size of an object from one camera. The object stands in a "checkerboard room" (a floor and two walls covered in 10 mm squares). One frame of the panels calibrates the camera, the corner where the three panels meet gives a fixed coordinate frame, and the object is found as the squares it hides. The program fits an upright shape to the outline and prints length, width and height.

It works with the ESP32-CAM (over its USB cable), a phone (through the Android app in this repository, or a phone browser), a webcam, or a single photo. It also contains a small neural denoiser for the ESP32-CAM picture and a residual corrector for the sizes.

Everything is one Python file, `camdist.py`.

Developed and tested on Linux (Arch). Windows and macOS are untested.

## What is in the folder

| Path | What it is |
|---|---|
| `camdist.py` | the whole program, every command |
| `requirements.txt` | the Python packages, with the versions it was tested with |
| `grid.pdf` | the checkerboard sheets to print (A4, 10 mm squares) |
| `thonny_run.py`, `sample_jar.jpg` | a menu launcher for Thonny and a sample photo |
| `data/` | `camdist.sqlite` (logs, calibrations, frames), `checkpoints/` (denoiser), `denoise/` (recorded bursts) |
| `reports/` | mesh reports (JPEG, JSON, OBJ) and CAD files written by `room` |
| `../camdist_app/` | the Android app (Kotlin) |
| `../esp32cam/SerialCam/` | the ESP32-CAM sketch that streams over USB serial |
| `../esp32cam/CameraWebServer/` | the stock WiFi sketch (older, not needed) |

## Install

### 1. Python

Python 3.12 is what it was tested on.

```
cd camdist
python3 -m venv ../.venv
source ../.venv/bin/activate          # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `torchvision` should be installed from https://pytorch.org for your hardware first if you want a GPU. A GPU is not required, but the segmentation network takes a few seconds per frame on a CPU and the live view is then slow. The denoiser trains on a GPU in reasonable time only.

### 2. The models (once, needs internet)

The program reads its models from the local Hugging Face cache and never goes online by itself (`HF_HUB_OFFLINE=1`). Download them once:

```
HF_HUB_OFFLINE=0 python -c "
from transformers import AutoModelForImageSegmentation, CLIPModel, CLIPProcessor
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')"
```

That is a few GB. Depth Anything V2 (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`) is only used by the older `scene` command and downloads the same way. On Windows, set the variable with `set HF_HUB_OFFLINE=0` first.

### 3. Optional tools

| Tool | Needed for |
|---|---|
| FreeCAD (`freecadcmd` on the PATH) | the CAD export (`--cad`, key k) |
| `openssl` | the phone browser page (https). The Android app does not need it |
| `arduino-cli` | flashing the ESP32-CAM |
| Android Studio and `adb` | building and installing the phone app |

### 4. Check the install

```
python camdist.py selfcheck
```

It runs 17 synthetic tests (rendered rooms with a known truth) and takes about a minute. Every other command refuses to run until `selfcheck` has passed on the exact file, and runs it by itself if needed. Edit `camdist.py` and the next command repeats it.

### Using Thonny

The quickest way is `thonny_run.py`: open it in Thonny, press Run (F5) and pick from the menu in the Shell panel (sample photo, your own photo, phone, ESP32-CAM, webcam). It needs only the packages listed in its first lines for the sample photo. For the full set, point Thonny at the virtual environment: Tools, Options, Interpreter, "Alternative Python 3 interpreter", and choose `../.venv/bin/python`. The program takes command-line arguments, so turn on View, Program arguments and type for example `room --image empty_room.jpg --out out.jpg`. The live windows (OpenCV) work from Thonny, but a terminal is more reliable.

## Try it without any hardware

Measure one photo and write the annotated result:

```
python camdist.py room --image empty_room.jpg --out annotated.jpg
```

The room calibrates itself from the photo (it needs the floor and both walls in view). Add `--report out.jpg` for a one-page mesh report with three views, volume and fit quality, and `--cad` to also write a STEP and a FreeCAD file next to it. `--name` names the object with CLIP.

## The room

Print `grid.pdf` on A4 sheets (10 mm squares) and glue them to stiff cardboard: one floor and two walls meeting in a corner. They do not have to be square to each other or line up at the joints. All three panels must be in view of the camera, and the squares must be sharp and well lit.

## Live use

```
python camdist.py room --url phone                     # the phone app (or a phone browser)
python camdist.py room --url /dev/ttyUSB0              # the ESP32-CAM on its USB cable
python camdist.py room --url 0                         # a webcam, or /dev/video10 for a scrcpy stream
```

On Windows, write the ESP32-CAM port as `serial:COM3`. Add `@baud` to change the speed (default 1000000).

The first frame with all three panels in view calibrates the camera (hold it still for a second). After that every object in the room gets an outline, a shape and a size with an uncertainty, which tells you when to add another view.

Useful options:

| Option | What it does |
|---|---|
| `--object NAME` | tell it what is in the room and store that name and its true size with every log. A new name needs `--truth L W H` in metres once |
| `--denoise` | clean every ESP32-CAM frame with the trained denoiser before measuring |
| `--corrector` | apply the trained size corrector to the sizes it improves |
| `--shape auto\|box\|round\|cylinder\|sphere\|revolve\|dome` | force a shape family instead of letting CLIP choose |
| `--seg salient\|room\|detect\|none` | how the outline is found (`salient` is the default) |
| `--table` | boards scattered on a table instead of the room |
| `--stack N` | average frames while the camera is still |
| `--cad` | write a CAD model with every report |

Keys in the live window (the buttons of the phone app send the same keys):

| Key | Action |
|---|---|
| space | log the current measurement (and its frame) to the database |
| m | write a mesh report to `reports/` |
| k | write a CAD model to `reports/` (needs FreeCAD) |
| s | cycle the shape family |
| c | calibrate again from this frame |
| f | pick the floor by hand |
| o | show the covered cells |
| r | reset the tracked views |
| Tab or n | next object |
| e, E, g, G, a | exposure and gain of the ESP32-CAM (a: automatic) |
| q or Esc | quit |

Known objects have their true size stored (built in: see `python camdist.py room -h`, plus the ones added to the database such as `crane`, `box_2`, `esp_box`, `nivea_cream` and `small_2020_profile`). A new name needs `--truth L W H` once.

### Logging for training

Log many views of one object with `--object NAME`, then label the logs:

```
python camdist.py log-tags            # camera, room or outside, object, usable for training
```

Logs are labelled in the table `log_tags` (camera: `esp32`, `phone-usb`, `phone-wifi`; setup: `room` or `outside room`; `use`: 0 keeps a log out of training). Logs without a label count as normal logs in the room.

## The phone app

The app streams the phone's camera to the laptop, draws the outlines the laptop sends back, and has the same buttons as the keys above. It can also show the ESP32-CAM plugged into the phone as a second view (split screen).

### Install the ready-made APK

The built APK is `../camdist_app/app/build/outputs/apk/debug/app-debug.apk`.

1. On the phone: Settings, About phone, tap "Build number" 7 times, then Settings, System, Developer options, and turn on USB debugging (or Wireless debugging).
2. Connect the phone by USB and run `adb install -r app-debug.apk`.
3. Wireless instead of USB: in Developer options open Wireless debugging, "Pair device with pairing code", then `adb pair IP:PORT` and `adb connect IP:PORT`, and install as above. With several devices attached, add `-s IP:PORT` to every adb command.

### Build it yourself

You need Android Studio (its bundled JDK 17) and the Android SDK. Put the SDK path in `../camdist_app/local.properties` (`sdk.dir=...`) and check the two paths at the top of `../camdist_app/build.sh` (`JAVA_HOME`, `ANDROID_HOME`). Then:

```
cd ../camdist_app
./build.sh            # builds app-debug.apk
./build.sh install    # builds, installs with adb and starts the app
```

### Connect it

1. Put the phone and the laptop on the same Wi-Fi network. Switch the VPN off on the phone: a VPN blocks the connection.
2. Allow incoming TCP 8080 and UDP 8081 on the laptop's firewall (the app finds the laptop by a UDP broadcast and then sends frames to port 8080).
3. Start the laptop side: `python camdist.py room --url phone`.
4. Open the app and allow the camera. The top left shows "laptop 192.168.x.x" and the frame rate. If it only says "looking for the laptop", tap that line and type the laptop's IP address.
5. Tap an object to select it and track it. Tap empty space to clear the selection.

Buttons: Stream ON/OFF, ESP connect, ESP turn, Log, Report, CAD, Clear, Reset, Shape.

The stream has its own field of view, so the room calibrates it anew (that is the "calibration" the first frame does). A phone browser also works without the app: open `https://LAPTOP_IP:8443`, accept the certificate warning and start the camera (needs `openssl` on the laptop).

### The ESP32-CAM on the phone

Plug the ESP32-CAM's USB serial adapter into the phone's USB-C port (it needs OTG support), allow the USB permission, and tap "ESP connect". The app shows the ESP status line (bytes, frames, incomplete frames) and a split view. "ESP turn" rotates the ESP picture in 90 degree steps.

Status: working since 17 Oct. The cause of the long "0 bytes" problem was the read buffer in the app: the CH340 driver returns nothing on a 16 KB buffer and about 100 KB/s on 4 KB. The app also used to start two readers on the same port after the activity was recreated, which makes them steal each other's bytes. Both are fixed. `adb logcat -s camdist-esp` shows what the reader is doing (`first frame 800x600` when it works). See `handoff.md` 10.17.

## The ESP32-CAM

The sketch `../esp32cam/SerialCam/SerialCam.ino` sends JPEG frames over the USB serial adapter at 1 Mbaud (about 4 fps at 800 x 600) and takes settings as text lines (`framesize`, `quality`, `aec`, `agc` ...). Flash it with the board in boot mode (GPIO0 to GND while resetting):

```
arduino-cli core install esp32:esp32
arduino-cli compile --fqbn esp32:esp32:esp32cam ../esp32cam/SerialCam
arduino-cli upload  --fqbn esp32:esp32:esp32cam -p /dev/ttyUSB0 ../esp32cam/SerialCam
```

Then remove the GPIO0 jumper, press reset, and run `python camdist.py room --url /dev/ttyUSB0`. On Linux your user must be in the `dialout` group (or `uucp` on Arch) to open the port.

The room is dim for this camera: it runs with a long exposure and high gain. A lamp over the room helps more than anything else.

## Training

### Size corrector

```
python camdist.py train-corrector --room --min-views 1
```

It learns the geometry's residual from logged objects of known size and evaluates on whole held-out objects. A size is only corrected live if the corrected error beats the geometry's by 10 percent. By default only logs of objects in the room are used (`--setup room`), and `--camera esp32|phone-usb|phone-wifi` selects one camera. The command prints how many logs each camera contributes. The result is stored in the database and used by `room --corrector`.

### ESP32-CAM denoiser

A small grey U-Net, no diffusion. Record bursts of the still scene, train, then judge it on the measurement:

```
python camdist.py esp-capture --url /dev/ttyUSB0      # space records a scene (a burst of steady frames), q quits
python camdist.py denoise-train                       # checkpoints every 250 steps, run the same command again to resume
python camdist.py denoise-eval                        # corner jitter and plane fit: raw, blur, denoiser
python camdist.py room --url /dev/ttyUSB0 --denoise   # use it live
```

`denoise-train` writes `data/checkpoints/denoise_gray.pt` and a comparison picture `data/denoise/held_out_comparison.jpg`. `--fresh` starts over. It trains on synthetic pairs made from the logged room frames, so it needs the database. Training takes a while: start it when you can leave it.

## Other commands

`python camdist.py -h` lists every command. The older ones (`calib-board`, `calib-ruler`, `calib-phone`, `measure`, `ground`, `scene`, `fisher`, `collect`, `train-classifier`, `export`, `report`) belong to the first phase of the project and are described in `handoff.md`.

## Troubleshooting

| Problem | What to do |
|---|---|
| "no passing selfcheck for this version" | normal after editing the file, it runs by itself |
| `calibrating: need the floor and both walls in view` | all three panels must be in the frame, sharp and lit. Move back a little |
| the program waits on the network at start | the models are not in the cache. Run the download step once (`HF_HUB_OFFLINE=0`) |
| the app says "looking for the laptop" | same Wi-Fi, VPN off, ports 8080 and 8081 open. Or tap the line and type the IP |
| `adb: more than one device` | add `-s IP:PORT` to the adb command |
| no object is found | the object must stand on the floor inside the room, away from the panel joints |
| a size has a large uncertainty (plus or minus) | view from lower or from the side, then add another view with the camera moved at least 3 cm |
| CAD file missing | `freecadcmd` must be on the PATH |
| out of GPU memory | run only one copy of the program at a time |

## Notes

- The database is large (about 900 MB: it holds every logged frame). `data/camdist.before_*.sqlite` are backups made before changes to the data. Delete them when you no longer need them.
- Measurements and calibrations are stored per session, so every number can be recomputed from the database.
- The object sizes in `room_objects` (database, table `config`) are the true sizes used for training. Keep them honest: a wrong truth teaches the corrector a wrong residual.
