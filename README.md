# sheda

A small project that measures the size of an object with one or two cameras, a phone and an ESP32-CAM.

The object stands in a "checkerboard room": a floor and two walls covered in 10 mm squares. The squares give the camera's calibration and a fixed coordinate frame, so there is no calibration step. The program finds the object's outline, fits an upright shape to it and prints length, width and height with an uncertainty. With both cameras it fits the shape to the outlines of both together.

| Folder | What it is |
|---|---|
| `camdist/` | the Python program (`camdist.py`) and the checkerboard sheets. Start with `camdist/README.md` |
| `camdist_app/` | the Android app (Kotlin) that streams the phone camera and the ESP32-CAM. `./build.sh install` builds and installs it |
| `esp32cam/` | the ESP32-CAM sketches (`SerialCam` streams over USB) |

Quickest start: open `camdist/thonny_run.py` in Thonny and press Run.

The database (`camdist/data/`: logs, frames, calibrations, trained models) is not in the repository. Without it the program starts a fresh one.
