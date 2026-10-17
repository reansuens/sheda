"""Thonny: open this file and press Run (F5), then answer in the Shell panel below.

It starts camdist.py with the right arguments, so nothing has to be typed on a command line.
The first run checks the program against rendered rooms (selfcheck, about a minute).

Packages (Thonny: Tools, Manage packages): numpy scipy opencv-python pillow requests pyserial matplotlib
That is enough for the sample photo. The phone/ESP32-CAM live views need the same. The neural outline and the
object names also need torch, transformers, timm, einops, kornia and the models (README_full.md, "The models").
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def models_ready():
    """True when the neural outline (BiRefNet) can be used: packages and the local model cache are there."""
    try:
        import torch, transformers  # noqa: F401
    except ImportError:
        return False
    hub = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub')
    return os.path.isdir(os.path.join(hub, 'models--ZhengPeng7--BiRefNet'))


def run(args):
    print('\ncamdist.py ' + ' '.join(args) + '\n')
    sys.argv = ['camdist.py'] + args
    try:
        runpy.run_path(os.path.join(HERE, 'camdist.py'), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print('stopped:', e.code)


MENU = """
camdist: object size in the checkerboard room

  1  measure the sample photo (a cream jar, 78 x 78 x 56 mm)
  2  measure my own photo
  3  live: the phone (Android app, same Wi-Fi)
  4  live: the ESP32-CAM on a USB serial port
  5  live: a webcam
  q  quit
"""

if __name__ == '__main__':
    fast = [] if models_ready() else ['--seg', 'room', '--shape', 'round']
    if fast:
        print('Neural models not installed: using the pattern outline and a round shape (fine for a first try).')
    print(MENU)
    choice = input('choose: ').strip().lower()
    if choice == '1':
        run(['room', '--image', 'sample_jar.jpg', '--out', 'annotated.jpg', '--shape', 'cylinder'] + (['--seg', 'room'] if fast else []))
        print('\nThe annotated picture is annotated.jpg in this folder.')
    elif choice == '2':
        path = input('photo (the room: floor and both walls in view): ').strip().strip('"')
        run(['room', '--image', path, '--out', 'annotated.jpg'] + fast)
        print('\nThe annotated picture is annotated.jpg in this folder.')
    elif choice == '3':
        print('Phone and laptop on the same Wi-Fi, open the CamDist app. Close the window with q.')
        run(['room', '--url', 'phone'] + fast)
    elif choice == '4':
        port = input('port (Linux /dev/ttyUSB0, Windows COM3): ').strip()
        run(['room', '--url', ('serial:' + port) if port.upper().startswith('COM') else port] + fast)
    elif choice == '5':
        run(['room', '--url', '0'] + fast)
    else:
        print('bye')
