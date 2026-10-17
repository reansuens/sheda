# Demo cheat sheet

## Before anyone arrives (5 minutes)

1. Terminal: `cd ~/Downloads/4UNI/5UNI/grad_project/camdist && source ../.venv/bin/activate`
2. `python camdist.py selfcheck` should end with `selfcheck PASSED` (about a minute). Do it now, not in front of people.
3. Phone and laptop on the same Wi-Fi, VPN off on the phone. Open the CamDist app once to check that the top left says `laptop 192.168.x.x`.
4. Checkerboard room built, well lit, object inside, all three panels (floor and both walls) in view of the phone.
5. For the two-camera demo: the ESP32-CAM's USB adapter plugged into the phone, tap **ESP connect** until the right half of the app shows its picture. Keep **ESP mirror** ON.

## The demo

Phone only (start with this one):

```
python camdist.py room --url phone --name
```

Wait for `self-calibrated: f ...` (hold the phone still for a second). Each object gets its outline, a name, and `L x W x H mm`. Tap an object in the app to select it. Move the phone around the object: the numbers settle and the error bars shrink. Space logs a measurement.

Both cameras (phone + ESP32-CAM): the same command. The laptop prints `second camera: the ESP32-CAM relayed by the phone app` and the window splits in two.

The ESP32-CAM alone on the laptop's USB:

```
python camdist.py room --url /dev/ttyUSB0 --name
```

Keys: **space** log, **m** report picture, **k** CAD model, **w** wireframe, **q** quit.

## If something goes wrong

| What you see | Do |
|---|---|
| app: `laptop not reachable` | the command is not running, or VPN is on, or the firewall blocks TCP 8080 / UDP 8081 |
| app: `looking for the laptop` | tap that line and type the laptop's IP address |
| no ESP picture in the app | tap **ESP connect** once and wait 25 seconds. Do not keep tapping it |
| `calibrating: need the floor and both walls in view` | move the camera back until all three panels are in view |
| the live demo fails completely | no hardware needed: `python camdist.py room --image sample_jar.jpg --out annotated.jpg` (a cream jar, true size 78 x 78 x 56 mm) or open `thonny_run.py` in Thonny and press Run |

## Numbers to quote (from the stored logs)

- Phone alone, geometry only: about 8 mm mean error per size (525 logs of 27 objects, 3 or more views each), 5 mm median.
- Phone + ESP32-CAM on one jar, live: 1.5 / 1.7 / 2.5 mm error in length / width / height (28 logs, one easy cylinder, true size taken with a caliper).
- ESP32-CAM alone: about 10 mm per size.
- Run `python camdist.py rig-eval --corrected` beforehand if you want the phone vs ESP vs both table (it takes a few minutes).
