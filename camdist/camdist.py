#!/usr/bin/env python3
"""camdist: single-camera distance and size estimation (ESP32-CAM + laptop).

    python camdist.py selfcheck                 synthetic tests, gate everything else
    python camdist.py stream  --url http://IP   live stream, lock camera settings
    python camdist.py calib-board               checkerboard calibration (reference)
    python camdist.py calib-ruler               ruler calibration (supervisor's method)
    python camdist.py calib-nominal             datasheet estimate of f (no target needed)
    python camdist.py fisher                    estimability report for the ruler design
    python camdist.py measure                   live: distance to a ruler of known length
    python camdist.py ground                    live: distance and size of objects on the floor
    python camdist.py scene                     live: approximate distance and size of every object
    python camdist.py room                      live or from a photo: object size in the checkerboard room (no calibration step)
    python camdist.py log-tags                  labels of the recent room logs: camera, setup, object, usable for training
    python camdist.py esp-capture               record still bursts of the ESP32-CAM (denoiser data)
    python camdist.py denoise-train             train the grey ESP32-CAM denoiser (checkpoints, resumes on its own)
    python camdist.py denoise-eval              does the denoiser help the measurement? (corner jitter, plane fit)
    python camdist.py rig-eval                  phone alone, ESP32-CAM alone, both together: error against the true sizes
    python camdist.py collect                   record labelled frames for training
    python camdist.py train-corrector           MLP 1: geometric residual corrector (--room: for the room logs)
    python camdist.py train-classifier          network 2: object identification
    python camdist.py export                    int8 quantization for STM32
    python camdist.py report                    all figures and tables from saved data

Every subcommand except `selfcheck` needs a passing selfcheck of this exact
file (checked by SHA-256); if there is none it is run first.
All data lives in data/camdist.sqlite; figures in figures/ are made from it only.
Section numbers (e.g. "handoff 2.3") refer to handoff.md.
"""
import os as _os
_os.environ.setdefault('HF_HUB_OFFLINE', '1')  # models are read from the cache; set HF_HUB_OFFLINE=0 to download

import argparse
import collections
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# opencv-python bundles Qt with only the xcb plugin (Wayland runs it via XWayland) and no fonts;
# cv2 points QT_QPA_FONTDIR at its empty font folder on import, so redirect it afterwards
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import cv2  # noqa: E402

for _d in ('/usr/share/fonts/TTF', '/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/liberation',
           '/usr/share/fonts/noto', '/usr/share/fonts/truetype'):
    if os.path.isdir(_d):
        os.environ['QT_QPA_FONTDIR'] = _d
        break

ROOT = os.environ.get('CAMDIST_HOME', os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DB_PATH = os.path.join(DATA, 'camdist.sqlite')
FIG = os.path.join(ROOT, 'figures')
EXPORT = os.path.join(DATA, 'export')
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------------------------------------------------------
# database
# ----------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS selfcheck_runs(id INTEGER PRIMARY KEY, ts REAL, code_hash TEXT,
    passed INTEGER, results TEXT);
CREATE TABLE IF NOT EXISTS camera_settings(id INTEGER PRIMARY KEY, ts REAL, url TEXT,
    requested TEXT, readback TEXT);
CREATE TABLE IF NOT EXISTS calibrations(id INTEGER PRIMARY KEY, ts REAL, method TEXT, session TEXT,
    width INTEGER, height INTEGER, fx REAL, fy REAL, cx REAL, cy REAL, dist TEXT, rms REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS board_views(id INTEGER PRIMARY KEY, session TEXT, ts REAL,
    width INTEGER, height INTEGER, corners TEXT, brightness REAL, jpeg BLOB);
CREATE TABLE IF NOT EXISTS ruler_obs(id INTEGER PRIMARY KEY, session TEXT, ts REAL, L REAL,
    z_meas REAL, l_px REAL, l_std REAL, n_frames INTEGER, u1 REAL, v1 REAL, u2 REAL, v2 REAL,
    cx REAL, cy REAL, width INTEGER, height INTEGER, undistorted INTEGER, brightness REAL);
CREATE TABLE IF NOT EXISTS ground_obs(id INTEGER PRIMARY KEY, session TEXT, ts REAL, calib_id INTEGER,
    h REAL, d_true REAL, u REAL, v REAL, brightness REAL);
CREATE TABLE IF NOT EXISTS measurements(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, calib_id INTEGER,
    est TEXT, truth TEXT, brightness REAL, lighting TEXT);
CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, ts REAL, session TEXT, object TEXT,
    lighting TEXT, d_true REAL, w_true REAL, h_true REAL, features TEXT, geo TEXT,
    brightness REAL, crop BLOB, frame BLOB);
CREATE TABLE IF NOT EXISTS sizer_feats(measurement INTEGER PRIMARY KEY, ts REAL, source TEXT, feats TEXT);
CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, ref TEXT,
    data TEXT, blob BLOB);
"""
_con = None


def db():
    global _con
    if _con is None:
        os.makedirs(DATA, exist_ok=True)
        _con = sqlite3.connect(DB_PATH, check_same_thread=False)  # browser view writes from handler threads
        _con.executescript(SCHEMA)
    return _con


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def jdump(o):
    return json.dumps(o, default=_np_default)


def cfg_get(key, default=None):
    row = db().execute('SELECT value FROM config WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def cfg_set(key, value):
    db().execute('INSERT OR REPLACE INTO config VALUES(?,?)', (key, jdump(value)))
    db().commit()


def put_result(kind, data, ref=None, blob=None):
    cur = db().execute('INSERT INTO results(ts,kind,ref,data,blob) VALUES(?,?,?,?,?)',
                       (time.time(), kind, ref, jdump(data), blob))
    db().commit()
    return cur.lastrowid


def latest_result(kind, ref=None):
    q = 'SELECT id, ts, ref, data, blob FROM results WHERE kind=?'
    a = [kind]
    if ref is not None:
        q += ' AND ref=?'
        a.append(str(ref))
    row = db().execute(q + ' ORDER BY id DESC LIMIT 1', a).fetchone()
    if not row:
        return None
    return dict(id=row[0], ts=row[1], ref=row[2], data=json.loads(row[3]), blob=row[4])


def new_session(tag=''):
    return time.strftime('%Y%m%d_%H%M%S') + (f'_{tag}' if tag else '')


def code_hash():
    with open(os.path.abspath(__file__), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def png_bytes(img):
    return cv2.imencode('.png', img)[1].tobytes()


def jpg_bytes(img, q=95):
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def decode_img(blob, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(blob, np.uint8), flags)


# ----------------------------------------------------------------------------
# camera: ESP32-CAM control + newest-frame stream reader (handoff 1)
# ----------------------------------------------------------------------------
class LatestFrame:
    """Background thread that overwrites a single frame slot (handoff 1.3).

    `source_factory()` returns an iterator of frames; on error or exhaustion the
    reader reconnects. `get(after=seq)` waits for a frame newer than `seq` and
    returns (seq, laptop_timestamp, frame), so WiFi delay never piles up.
    """

    def __init__(self, source_factory, retry=1.0):
        self._factory, self._retry = source_factory, retry
        self._cond = threading.Condition()
        self.frame, self.seq, self.t, self.err = None, 0, 0.0, None
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop:
            try:
                for frame in self._factory():
                    if self._stop:
                        return
                    with self._cond:
                        self.frame, self.t = frame, time.time()
                        self.seq += 1
                        self._cond.notify_all()
                    self.err = None
            except Exception as e:  # reconnect after a short pause
                self.err = f'{type(e).__name__}: {e}'
                time.sleep(self._retry)

    def get(self, after=0, timeout=5.0):
        with self._cond:
            ok = self._cond.wait_for(lambda: self.seq > after or self._stop, timeout)
            if not ok or self.frame is None:
                return None
            return self.seq, self.t, self.frame

    def stop(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()


def mjpeg_frames(url, chunk=8192, timeout=6.0):
    """Parse a multipart MJPEG stream by JPEG SOI/EOI markers and decode each frame."""
    import requests
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = bytearray()
        for part in r.iter_content(chunk_size=chunk):
            buf += part
            while True:
                a = buf.find(b'\xff\xd8')
                if a < 0:
                    del buf[:-1]
                    break
                b = buf.find(b'\xff\xd9', a + 2)
                if b < 0:
                    if a:
                        del buf[:a]
                    break
                jpg = bytes(buf[a:b + 2])
                del buf[:b + 2]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img


def capture_frames(base, io_lock, timeout=2.5):
    """Poll the single-frame endpoint. Each request stands alone, so a WiFi hiccup costs one
    request instead of leaving the board's one-client stream handler stuck on a dead socket.
    The board's port-80 server answers one request at a time: `io_lock` lets settings
    changes take turns with the frame requests."""
    import requests
    with requests.Session() as s:
        while True:
            with io_lock:
                r = s.get(base + '/capture', timeout=timeout)
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def cv2_frames(src, loop=False):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {src}')
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                return
            if loop:
                time.sleep(1 / 15)  # video files play at about stream rate
            yield img
    finally:
        cap.release()


PHONE_PAGE = r'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>camdist</title><style>
html,body{margin:0;height:100%;background:#000;color:#eee;font:14px system-ui,sans-serif;overflow:hidden;touch-action:manipulation}
#stage{position:fixed;inset:0}
video,canvas#ov{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#top{position:fixed;top:0;left:0;right:0;padding:6px 10px;background:#000a;font-size:13px;white-space:pre-line;pointer-events:none}
#bar{position:fixed;bottom:0;left:0;right:0;display:flex;gap:6px;padding:8px;background:#000b;flex-wrap:wrap;justify-content:center}
button,select{font-size:15px;padding:10px 13px;border-radius:8px;border:0;background:#2b3a4a;color:#fff}
button.on{background:#0a7d5a}
#warn{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:#000d;font-size:22px;text-align:center}
</style></head><body>
<div id="stage"><video id="v" playsinline muted></video><canvas id="ov"></canvas></div>
<div id="top">tap Start camera</div>
<div id="warn">turn the phone to landscape</div>
<div id="bar">
<button id="go">Start camera</button><button id="send">Stream: off</button><button id="shot">Photo</button>
<button data-cmd="log">Log</button><button data-cmd="report">Report</button><button data-cmd="clear">Clear</button>
<button data-cmd="reset">Reset views</button><button data-cmd="shape">Shape</button>
<select id="res"><option>1600x1200</option><option>1280x960</option><option>1920x1440</option><option>800x600</option></select>
</div>
<script>
const $=id=>document.getElementById(id), v=$('v'), ov=$('ov');
let stream=null, track=null, streaming=false, running=false, msg='', fps=0, last=null;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
$('go').onclick=async()=>{
  try{
    const [w,h]=$('res').value.split('x').map(Number);
    if(stream) stream.getTracks().forEach(t=>t.stop());
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},
      width:{ideal:w},height:{ideal:h},frameRate:{ideal:30}}});
    track=stream.getVideoTracks()[0];
    try{await track.applyConstraints({advanced:[{zoom:1}]})}catch(e){}
    v.srcObject=stream; await v.play();
    ov.width=v.videoWidth; ov.height=v.videoHeight;
    $('go').textContent='Restart camera'; msg='camera '+v.videoWidth+'x'+v.videoHeight;
    if(!running){running=true;pump();}
  }catch(e){msg='camera error: '+e;}
  draw();
};
$('send').onclick=()=>{streaming=!streaming;$('send').textContent='Stream: '+(streaming?'on':'off');$('send').className=streaming?'on':'';};
$('shot').onclick=async()=>{
  try{const b=await new ImageCapture(track).takePhoto();await fetch('/photo',{method:'POST',body:b});msg='photo saved on the laptop';}
  catch(e){msg='photo failed: '+e;}
};
document.querySelectorAll('[data-cmd]').forEach(b=>b.onclick=()=>fetch('/cmd',{method:'POST',body:JSON.stringify({cmd:b.dataset.cmd})}));
ov.addEventListener('click',e=>{
  if(!v.videoWidth) return;
  const r=ov.getBoundingClientRect(), s=Math.min(r.width/v.videoWidth,r.height/v.videoHeight);
  const dw=v.videoWidth*s, dh=v.videoHeight*s, x=(e.clientX-r.left-(r.width-dw)/2)/dw, y=(e.clientY-r.top-(r.height-dh)/2)/dh;
  if(x<0||x>1||y<0||y>1) return;
  fetch('/tap',{method:'POST',body:JSON.stringify({x:x,y:y})});
});
function draw(){
  const g=ov.getContext('2d'); g.clearRect(0,0,ov.width,ov.height);
  const s=last&&last.res; let t=(v.videoWidth?v.videoWidth+'x'+v.videoHeight+'  '+fps.toFixed(1)+' fps  ':'')+msg;
  if(s){ t+='\n'+s.lines.join('\n');
    const k=ov.width/s.w, fs=Math.round(ov.width/38); g.font='bold '+fs+'px system-ui'; g.lineJoin='round';
    for(const o of s.objects){
      g.beginPath(); o.hull.forEach((p,i)=>i?g.lineTo(p[0]*k,p[1]*k):g.moveTo(p[0]*k,p[1]*k)); g.closePath();
      if(o.sel){g.fillStyle='rgba(255,224,0,.25)';g.fill();}
      g.lineWidth=(o.sel?5:2)*ov.width/1000; g.strokeStyle=o.sel?'#ffe000':'#aab'; g.stroke();
      if(o.sel||!s.objects.some(q=>q.sel)){
        const x=Math.min(...o.hull.map(p=>p[0]))*k, y=Math.min(...o.hull.map(p=>p[1]))*k-fs*0.4;
        g.lineWidth=fs/5; g.strokeStyle='#000'; g.strokeText(o.text,x,Math.max(fs,y)); g.fillStyle='#fff'; g.fillText(o.text,x,Math.max(fs,y));
      }
    }
  }
  $('top').textContent=t;
}
async function pump(){
  const c=document.createElement('canvas'), x=c.getContext('2d'); let tl=performance.now();
  while(running){
    const portrait=v.videoWidth&&v.videoHeight>v.videoWidth; $('warn').style.display=portrait?'flex':'none';
    if(!streaming||!v.videoWidth||portrait){await sleep(100);continue;}
    const t0=performance.now(); c.width=v.videoWidth; c.height=v.videoHeight; x.drawImage(v,0,0);
    const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.9));
    try{const r=await fetch('/frame',{method:'POST',body:blob}); last=await r.json();
      const dt=(performance.now()-tl)/1000; tl=performance.now(); fps=0.8*fps+0.2/dt; msg='';}
    catch(e){msg='laptop not reachable: '+e; await sleep(500);}
    draw(); await sleep(Math.max(0,60-(performance.now()-t0)));
  }
}
</script></body></html>'''


PHONE_PLAIN_PORT, PHONE_BEACON_PORT = 8080, 8081


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))  # no packet is sent; picks the interface that reaches the router
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def phone_cert(ip):
    """Self-signed certificate for the laptop's address (a phone browser only opens the camera on https);
    made once, again when the address changes. Returns (cert, key) paths."""
    import subprocess
    d = os.path.join(DATA, 'phone_cert')
    os.makedirs(d, exist_ok=True)
    cert, key, tag = (os.path.join(d, n) for n in ('cert.pem', 'key.pem', 'ip.txt'))
    if not (os.path.exists(cert) and os.path.exists(key) and os.path.exists(tag) and open(tag).read() == ip):
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825', '-keyout', key,
                        '-out', cert, '-subj', '/CN=camdist', '-addext', f'subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost'],
                       check=True, capture_output=True)
        open(tag, 'w').write(ip)
    return cert, key


class PhoneLink:
    """Receiver for a phone's browser camera over the local network (same router, no cable, no app to
    install). The phone opens https://<laptop ip>:<port>, starts its camera and POSTs JPEG frames; the laptop
    keeps the newest one (like LatestFrame), measures it as before and answers each frame with the latest
    result, which the page draws over the live view. Taps and button presses on the phone come back as
    selections and commands. Photos taken with the Photo button are saved in data/phone_photos/."""
    _inst = {}

    @classmethod
    def get(cls, port=8443):
        if port not in cls._inst:
            cls._inst[port] = cls(port)
        return cls._inst[port]

    def __init__(self, port):
        import ssl
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.cond = threading.Condition()
        self.frame, self.seq, self.t_last, self.size = None, 0, 0.0, None
        self.taps, self.cmds, self.result = [], [], None
        self.frame2, self.seq2, self.t2, self.result2 = None, 0, 0.0, None  # the ESP32-CAM relayed by the app
        self.port, self.ip = port, lan_ip()
        link = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, body, ctype='application/json', code=200):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def body(self):
                return self.rfile.read(int(self.headers.get('Content-Length') or 0))

            def do_GET(self):
                if self.path.split('?')[0] in ('/', '/index.html'):
                    self.reply(PHONE_PAGE.encode(), 'text/html; charset=utf-8')
                else:
                    self.reply(b'', 'text/plain', 204)

            def do_POST(self):
                data = self.body()
                if self.path == '/frame':
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with link.cond:
                            link.frame, link.t_last, link.size = img, time.time(), img.shape[1::-1]
                            link.seq += 1
                            link.cond.notify_all()
                    self.reply(jdump(dict(res=link.result)).encode())
                elif self.path == '/frame2':  # the second camera of the rig: an ESP32-CAM on the phone's USB cable
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if b'\xff\xd9' in data[-16:] else None
                    if img is not None:  # a frame that lost bytes on the serial cable is dropped
                        with link.cond:
                            link.frame2, link.t2 = img, time.time()
                            link.seq2 += 1
                    self.reply(jdump(dict(res=link.result2)).encode())
                elif self.path == '/tap':
                    with link.cond:
                        link.taps.append(json.loads(data))
                    self.reply(b'{}')
                elif self.path == '/cmd':
                    with link.cond:
                        link.cmds.append(json.loads(data).get('cmd'))
                    self.reply(b'{}')
                elif self.path == '/photo':
                    d = os.path.join(DATA, 'phone_photos')
                    os.makedirs(d, exist_ok=True)
                    name = os.path.join(d, time.strftime('%Y%m%d_%H%M%S') + '.jpg')
                    open(name, 'wb').write(data)
                    print(f'phone photo: {name}')
                    self.reply(b'{}')
                else:
                    self.reply(b'', 'text/plain', 404)

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # a browser that refuses the certificate, a dropped link
                pass

        cert, key = phone_cert(self.ip)
        self.httpd = Server(('0.0.0.0', port), H)  # https: the browser page
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True, do_handshake_on_connect=False)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.plain = Server(('0.0.0.0', PHONE_PLAIN_PORT), H)  # plain http: the Android app (local network only)
        threading.Thread(target=self.plain.serve_forever, daemon=True).start()
        threading.Thread(target=self._beacon, daemon=True).start()
        print(f'phone: the app finds this laptop by itself ({self.ip}, port {PHONE_PLAIN_PORT}); '
              f'the browser page is https://{self.ip}:{port}')

    def _beacon(self):
        """Tell the app where the laptop is: a UDP broadcast once a second (the app listens on PHONE_BEACON_PORT)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = f'camdist {PHONE_PLAIN_PORT}'.encode()
        targets = [('255.255.255.255', PHONE_BEACON_PORT), (self.ip.rsplit('.', 1)[0] + '.255', PHONE_BEACON_PORT)]
        while True:
            for t in targets:
                try:
                    s.sendto(msg, t)
                except OSError:
                    pass
            time.sleep(1.0)

    def frames(self):
        last = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq > last)
                last, img = self.seq, self.frame
            yield img

    def publish(self, payload):
        self.result = payload

    def publish2(self, payload):
        self.result2 = payload

    def second(self, max_age=3.0):
        """(sequence number, newest frame) of the second camera, None if it is not streaming."""
        with self.cond:
            if self.frame2 is None or time.time() - self.t2 > max_age:
                return None
            return self.seq2, self.frame2

    def take_taps(self):
        with self.cond:
            out, self.taps = self.taps, []
        return [(t['x'], t['y'], t.get('cam', 1)) for t in out]

    def take_cmds(self):
        with self.cond:
            out, self.cmds = self.cmds, []
        return out


def phone_payload(shape, res, sidx, status, cam=None):
    """What the phone page draws: the outlines (in the measured frame's pixels), a label each, the selection.
    With cam = (K, dist) of the camera that took the frame, every object also carries its fitted wireframe: 'mesh' the
    edges facing the camera and 'hid' the hidden ones, each as [x0, y0, x1, y1] in the frame's pixels."""
    objs = []
    for i, o in enumerate(res['objects']):
        nm = (o.get('name') or {}).get('general') or o['label']
        d = dict(hull=np.round(np.asarray(o['hull'], float).reshape(-1, 2)[::max(1, len(o['hull']) // 40)], 1).tolist(),
                 text=f"{nm}  {o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x {o['H'] * 1e3:.0f} mm", sel=(i == sidx))
        if cam is not None:
            vis, hid = mesh_segments(o, res['floor'], *cam)
            d.update(mesh=np.round(vis[::max(1, len(vis) // 500)]).astype(int).tolist(),
                     hid=np.round(hid[::max(1, len(hid) // 300)]).astype(int).tolist())
        objs.append(d)
    return dict(w=int(shape[1]), h=int(shape[0]), objects=objs, lines=[ln for ln in status if ln][:3])


class Camera:
    """ESP32-CAM over WiFi (stock CameraWebServer sketch) or over its USB cable (SerialCam sketch:
    --url /dev/ttyUSB0), other MJPEG URL (phone app), webcam index, or video file."""

    SERIAL_MAGIC = b'\xff\xaa\x55'
    SERIAL_FRAMESIZES = {'QVGA (320x240)': 5, 'CIF (400x296)': 6, 'HVGA (480x320)': 7, 'VGA (640x480)': 8,
                         'SVGA (800x600)': 9, 'XGA (1024x768)': 10, 'HD (1280x720)': 11, 'SXGA (1280x1024)': 12,
                         'UXGA (1600x1200)': 13}

    def __init__(self, url, backend='requests'):
        self.url, self.backend = url, backend
        self.raw_sink = None  # optional callable(jpeg bytes): every complete frame as the board sent it (esp-capture)
        self.kind, self.transport = 'esp32', 'http'
        self._ser, self._status, self._status_seq, self._reading = None, {}, 0, False
        if url.startswith(('/dev/ttyUSB', '/dev/ttyACM', 'serial:')):  # SerialCam over the USB cable
            self.transport = 'serial'
            self.port, _, baud = url.split('serial:', 1)[-1].partition('@')  # /dev/ttyUSB0[@baud]
            self.baud = int(baud) if baud else 1000000
            self.base = self.stream_url = 'serial:' + self.port
            self.settings = {}
            self.io_lock = threading.Lock()
            return
        if url == 'phone' or url.startswith('phone:'):  # a phone's browser camera over the local network
            self.kind, self.transport = 'webphone', 'net'
            self.link = PhoneLink.get(int(url.partition(':')[2] or 8443))
            self.base = self.stream_url = 'phone'
            return
        if re.fullmatch(r'\d+', url) or url.startswith('/dev/video'):
            self.kind = 'webcam'
        elif os.path.exists(url):
            self.kind = 'file'
        if self.kind == 'esp32':
            if not url.startswith('http'):
                url = 'http://' + url
            m = re.match(r'(https?://[^/:]+)(:\d+)?(/.*)?$', url)
            if not m:
                raise SystemExit(f'cannot parse url {url}')
            self.base = m.group(1) if m.group(2) in (None, ':80', ':81') else m.group(1) + m.group(2)
            self.stream_url = url if (m.group(3) or '').startswith('/stream') else f'{m.group(1)}:81/stream'
            if m.group(3) and not m.group(3).startswith('/stream'):
                # any other MJPEG URL, e.g. a phone app (IP Webcam :8080/video, DroidCam :4747/video):
                # read as is, no ESP32 settings to lock (fix exposure and focus in the app instead)
                self.kind, self.stream_url = 'mjpeg', url
        self.settings = {}
        self.io_lock = threading.Lock()

    # --- SerialCam: JPEG packets over USB serial at 2 Mbaud ---
    def _serial_open(self):
        if self._ser is None or not self._ser.is_open:
            import serial
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = self.port, self.baud, 0.3
            if self.port.startswith('/dev/tty'):  # USB adapter: DTR/RTS are wired to reset and boot
                s.dtr = s.rts = False  # keep them released (a pseudo-terminal in tests has no such lines)
            try:
                s.open()
            except serial.SerialException as e:
                raise OSError(f'cannot open {self.port}: {e}') from e
            self._ser, self._buf = s, bytearray()
            self._serial_wait(1, timeout=8.0)  # opening may reset the board: wait for its first frame
        return self._ser

    def _serial_packets(self):
        """Complete packets (type, payload) from the bytes read so far; garbage is skipped."""
        buf = self._buf
        while True:
            i = buf.find(self.SERIAL_MAGIC)
            if i < 0:
                del buf[:max(0, len(buf) - 2)]
                return
            del buf[:i]
            if len(buf) < 8:
                return
            typ, n = buf[3], int.from_bytes(buf[4:8], 'little')
            if typ not in (1, 2) or n > 600000:  # not a header (boot messages, noise)
                del buf[:3]
                continue
            if len(buf) < 8 + n:
                return
            payload = bytes(buf[8:8 + n])
            if typ == 1 and payload[-2:] != b'\xff\xd9':  # bytes were lost in this frame: drop it and
                self.dropped = getattr(self, 'dropped', 0) + 1  # resynchronise on the next header
                del buf[:3]
                continue
            del buf[:8 + n]
            if typ == 2:
                try:
                    self._status = json.loads(payload)
                    self._status_seq += 1
                except ValueError:
                    pass
            yield typ, payload

    def _serial_read(self):
        try:
            chunk = self._ser.read(self._ser.in_waiting or 4096)
        except Exception as e:  # unplugged, board reset: reopen on the next attempt
            self._serial_close()
            raise OSError(f'serial camera: {e}') from e
        self._buf += chunk
        return bool(chunk)

    def _serial_wait(self, want, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self._serial_read()
            for typ, _ in self._serial_packets():
                if typ == want:
                    return True
        if want == 1:
            self._serial_close()
            if self._status.get('error'):  # the sketch runs but reports a problem, e.g. the camera module
                raise OSError(f"the board says: {self._status['error']} (unplug the USB for 10 s; if it persists, "
                              f"reseat the camera's ribbon cable)")
            raise OSError(f'no frames from {self.port} within {timeout:.0f} s: is SerialCam flashed and running?')
        return False

    def _serial_close(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def _serial_frames(self):
        with self.io_lock:
            self._serial_open()
        self._reading = True
        last = time.time()
        try:
            while True:
                got = self._serial_read()
                for typ, payload in self._serial_packets():
                    if typ == 1:
                        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            last = time.time()
                            if self.raw_sink is not None:
                                self.raw_sink(payload)
                            yield img
                if not got and time.time() - last > 5:
                    self._serial_close()
                    raise OSError(f'no frames from {self.port} for 5 s')
        finally:
            self._reading = False

    def _serial_command(self, text):
        with self.io_lock:
            ser = self._serial_open()
            ser.write((text + '\n').encode())
            ser.flush()

    def frames(self):
        if self.transport == 'serial':
            return self._serial_frames()
        if self.kind == 'webphone':
            return self.link.frames()
        if self.kind == 'webcam':  # index or /dev/videoN (e.g. scrcpy --v4l2-sink from a phone)
            return cv2_frames(int(self.url) if self.url.isdigit() else self.url)
        if self.kind == 'file':
            return cv2_frames(self.url, loop=True)
        if self.backend == 'cv2':
            return cv2_frames(self.stream_url)
        if self.backend == 'capture':
            return capture_frames(self.base, self.io_lock)
        return mjpeg_frames(self.stream_url)

    def _get(self, path, timeout=8.0):
        import requests
        with self.io_lock:
            r = requests.get(self.base + path, timeout=timeout)
        r.raise_for_status()
        return r

    def control(self, var, val):
        if self.transport == 'serial':
            self._serial_command(f'{var}={int(val)}')
            return
        self._get(f'/control?var={var}&val={int(val)}')

    def status(self):
        if self.transport == 'serial':
            seq = self._status_seq
            self._serial_command('status')
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if self._status_seq > seq:
                    return dict(self._status)
                if not self._reading:  # nobody is reading the stream yet: read it here
                    with self.io_lock:
                        self._serial_read()
                        for _ in self._serial_packets():
                            pass
                else:
                    time.sleep(0.02)
            raise OSError('no status reply from the serial camera')
        return self._get('/status').json()

    def framesize_options(self):
        """framesize label -> numeric value, read from the board's own web page (handoff 1.2)."""
        if self.transport == 'serial':  # the sketch reports the driver's numbers (they changed between versions)
            st = self.status()
            opts = {k[3:]: int(v) for k, v in st.items() if k.startswith('fs_')}
            return {'SVGA (800x600)' if k == 'SVGA' else k: v for k, v in opts.items()} or dict(self.SERIAL_FRAMESIZES)
        html = self._get('/', timeout=5.0).text
        m = re.search(r'id="framesize".*?</select>', html, re.S)
        opts = re.findall(r'<option value="(\d+)"[^>]*>([^<]*)</option>', m.group(0) if m else '')
        return {label.strip(): int(v) for v, label in opts}

    def lock(self, aec_value, agc_gain, framesize=None, quality=10):
        """Lock resolution, exposure, gain and white balance; read back and log (handoff 1.2)."""
        if framesize is None:
            opts = self.framesize_options()
            hit = [v for k, v in opts.items() if k.upper().startswith('SVGA') or '800X600' in k.upper()]
            if not hit:
                raise SystemExit(f'SVGA not found on the board page; options {opts}. Pass --framesize N.')
            framesize = hit[0]
        req = [('framesize', framesize), ('quality', quality), ('aec', 0), ('aec_value', aec_value),
               ('agc', 0), ('agc_gain', agc_gain), ('awb', 0), ('lenc', 1)]
        for var, val in req:
            self.control(var, val)
            if var == 'framesize':
                time.sleep(0.6)  # sensor reconfigures
        time.sleep(0.2)
        back = self.status()
        bad = {k: (v, back.get(k)) for k, v in req if k in back and int(back[k]) != int(v)}
        if bad:
            print(f'warning: settings did not stick (requested, readback): {bad}')
        db().execute('INSERT INTO camera_settings(ts,url,requested,readback) VALUES(?,?,?,?)',
                     (time.time(), self.base, jdump(dict(req)), jdump(back)))
        db().commit()
        self.settings = dict(req)
        print('locked: ' + ', '.join(f'{k}={back.get(k, "?")}' for k, _ in req))
        return back


# OV2640 analog gain for each agc_gain index of the esp32-camera driver
AGC_GAINS = [1, 2, 2.5] + [i + 1 for i in range(3, 31)]


def local_ipv4():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def scan_for_esp32(timeout=0.7):
    """Probe the local /24 for the stock sketch's /status endpoint."""
    import requests
    ip = local_ipv4()
    if not ip:
        return []
    prefix = ip.rsplit('.', 1)[0]

    def probe(i):
        host = f'{prefix}.{i}'
        try:
            r = requests.get(f'http://{host}/status', timeout=timeout)
            if r.ok and 'framesize' in r.text:
                return host
        except Exception:
            return None

    print(f'scanning {prefix}.0/24 for an ESP32-CAM ...')
    with ThreadPoolExecutor(64) as ex:
        return [h for h in ex.map(probe, range(1, 255)) if h]


def resolve_url(args):
    url = getattr(args, 'url', None) or cfg_get('url')
    if not url:
        raise SystemExit('no camera url: pass --url http://IP (or a webcam index / video file)')
    if getattr(args, 'url', None):
        cfg_set('url', url)
    return url


# ----------------------------------------------------------------------------
# calibration storage and undistortion
# ----------------------------------------------------------------------------
@dataclass
class Calib:
    id: int
    method: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    rms: float
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


CAM_SOURCE = None  # kind of the camera in use, stored with every calibration (the phone's browser stream differs from scrcpy's)


def save_calib(method, session, size, K, dist, rms, meta):
    meta = dict(meta, source=CAM_SOURCE)
    d = np.zeros(5)
    d[:len(dist)] = np.ravel(dist)[:5]
    cur = db().execute(
        'INSERT INTO calibrations(ts,method,session,width,height,fx,fy,cx,cy,dist,rms,meta) '
        'VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
        (time.time(), method, session, size[0], size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2],
         jdump(d), rms, jdump(meta)))
    db().commit()
    return cur.lastrowid


def load_calib(spec=None, required=True):
    """spec: None (latest board, else ruler, else nominal), a method name, or a calibration id."""
    q = 'SELECT id,method,width,height,fx,fy,cx,cy,dist,rms,meta FROM calibrations'
    if spec is None:
        row = None
        for method in ('board', 'phone', 'ruler', 'nominal'):
            row = row or db().execute(q + ' WHERE method=? ORDER BY id DESC LIMIT 1', (method,)).fetchone()
    elif str(spec).isdigit():
        row = db().execute(q + ' WHERE id=?', (int(spec),)).fetchone()
    else:
        row = db().execute(q + ' WHERE method=? ORDER BY id DESC LIMIT 1', (spec,)).fetchone()
    if not row:
        if required:
            raise SystemExit('no calibration found: run calib-board, calib-ruler or calib-nominal first')
        return None
    return Calib(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                 np.array(json.loads(row[8])), row[9], json.loads(row[10] or '{}'))


def ruler_delta():
    """Optical-centre offset behind the reference edge, from the latest ruler fit (handoff 3.2)."""
    c = load_calib('ruler', required=False)
    return float(c.meta.get('delta', 0.0)) if c else 0.0


# ----------------------------------------------------------------------------
# Fisher retention (handoff 3)
# ----------------------------------------------------------------------------
def retention(b, A=None, rcond=1e-10):
    """I_raw = b'b, I_marg = ||(I - AA+) b||^2 via SVD with a relative cut, rho = I_marg/I_raw."""
    b = np.asarray(b, float).ravel()
    i_raw = float(b @ b)
    if i_raw == 0:  # the data do not depend on this parameter at all: no information
        return 0.0, 0.0, 0.0
    if A is None or np.size(A) == 0:
        return i_raw, i_raw, 1.0
    A = np.asarray(A, float).reshape(len(b), -1)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, s > rcond * s[0]] if s[0] > 0 else U[:, :0]
    r = b - Uk @ (Uk.T @ b)
    i_marg = float(r @ r)
    return i_raw, i_marg, i_marg / i_raw


def rho_uniform_w(ratio):
    """rho for f with delta when w = 1/Z is uniform on [w_max/ratio, w_max] (scale free)."""
    a = 1.0 / ratio
    m = lambda k: (1 - a ** (k + 1)) / ((k + 1) * (1 - a))
    return 1 - m(3) ** 2 / (m(2) * m(4))


def rho_uniform_z(ratio):
    """rho for f with delta when Z itself is uniform on [Z_min, ratio * Z_min]; tends to 1/4."""
    m = lambda k: (1 - ratio ** (1 - k)) / ((k - 1) * (ratio - 1))
    return 1 - m(3) ** 2 / (m(2) * m(4))


# ----------------------------------------------------------------------------
# ruler model (handoff 2.2, 3.2)
# ----------------------------------------------------------------------------
RULER_PARAMS = ('f', 'delta', 'k1')
NUISANCE_SETS = [(), ('delta',), ('k1',), ('delta', 'k1')]


def undistort_norm(p, k1, iters=25):
    """Invert x_d = x (1 + k1 r^2) for normalized points by fixed-point iteration."""
    q = np.array(p, float)
    if k1 == 0:
        return q
    for _ in range(iters):
        q = p / (1 + k1 * (q ** 2).sum(-1, keepdims=True))
    return q


def ruler_resid(prm, ob):
    """Pixel residual f * (|p2 - p1| - L / (Z_meas + delta)), endpoints undistorted with k1.

    With k1 = 0 this is l_obs - f L / (Z_meas + delta), the model of handoff 3.2.
    """
    f, d, k1 = prm['f'], prm['delta'], prm['k1']
    p1 = undistort_norm((ob['P1'] - ob['c']) / f, k1)
    p2 = undistort_norm((ob['P2'] - ob['c']) / f, k1)
    n = np.linalg.norm(p2 - p1, axis=1)
    return f * n - f * ob['L'] / (ob['z'] + d)


def ruler_synth(f, delta, k1, c, L, z, pos, angle=0.0, noise=0.0, rng=None):
    """Synthetic ruler observations: centre at normalized image position pos[i] (x, y)."""
    z = np.asarray(z, float)
    pos = np.broadcast_to(np.asarray(pos, float), (len(z), 2))
    Z = z + delta
    half = (L / (2 * Z))[:, None] * np.array([math.cos(angle), math.sin(angle)])
    pts = []
    for sgn in (-1, 1):
        p = pos + sgn * half
        p = p * (1 + k1 * (p ** 2).sum(1, keepdims=True))
        pts.append(np.asarray(c) + f * p)
    if noise:
        pts = [q + rng.normal(0, noise, q.shape) for q in pts]
    return dict(z=z, P1=pts[0], P2=pts[1], L=np.full(len(z), float(L)), c=np.asarray(c, float))


def ruler_init_f(ob):
    l = np.linalg.norm(ob['P2'] - ob['P1'], axis=1)
    return float(np.median(l * ob['z'] / ob['L']))


def fit_ruler(ob, free=('f', 'delta'), base=None):
    from scipy.optimize import least_squares
    p0 = dict(f=ruler_init_f(ob), delta=0.0, k1=0.0)
    if base:
        p0.update(base)
    scale = dict(f=100.0, delta=0.01, k1=0.05)
    fun = lambda x: ruler_resid({**p0, **dict(zip(free, x))}, ob)
    n, p = len(ob['z']), len(free)
    res = least_squares(fun, [p0[k] for k in free], x_scale=[scale[k] for k in free],
                        method='lm' if n > p else 'trf')
    prm = {**p0, **dict(zip(free, res.x))}
    dof = n - p
    s2 = float(res.fun @ res.fun / dof) if dof > 0 else float('nan')
    cov = np.linalg.pinv(res.jac.T @ res.jac) * s2
    se = dict(zip(free, np.sqrt(np.abs(np.diag(cov)))))
    return dict(params=prm, se=se, sigma=math.sqrt(s2) if dof > 0 else float('nan'),
                resid=res.fun, free=list(free), n=n)


def ruler_jacobian(ob, prm, names=RULER_PARAMS):
    """Central-difference sensitivities of the pixel residual to each parameter (columns)."""
    step = dict(f=1e-5 * prm['f'], delta=1e-6, k1=1e-6)
    cols = []
    for k in names:
        hi, lo = dict(prm), dict(prm)
        hi[k] += step[k]
        lo[k] -= step[k]
        cols.append((ruler_resid(hi, ob) - ruler_resid(lo, ob)) / (2 * step[k]))
    return np.stack(cols, 1)


def fisher_table(J, sigma, names=RULER_PARAMS):
    """Retention of f under each nuisance set, whitened by the pixel noise sigma."""
    Jw = J / sigma
    col = {k: Jw[:, i] for i, k in enumerate(names)}
    rows = []
    for nuis in NUISANCE_SETS:
        A = np.stack([col[k] for k in nuis], 1) if nuis else None
        i_raw, i_marg, rho = retention(col['f'], A)
        rows.append(dict(nuisance=list(nuis), I_raw=i_raw, I_marg=i_marg, rho=rho,
                         se_pred=i_marg ** -0.5 if i_marg > 0 else float('inf')))
    return rows


def ruler_obs_from_db(session=None):
    con = db()
    if session is None:
        row = con.execute('SELECT session FROM ruler_obs ORDER BY id DESC LIMIT 1').fetchone()
        if not row:
            return None, None
        session = row[0]
    rows = con.execute('SELECT z_meas,u1,v1,u2,v2,L,cx,cy,l_std,width,height FROM ruler_obs '
                       'WHERE session=? ORDER BY id', (session,)).fetchall()
    if not rows:
        return None, session
    a = np.array(rows, float)
    ob = dict(z=a[:, 0], P1=a[:, 1:3], P2=a[:, 3:5], L=a[:, 5], c=a[0, 6:8], l_std=a[:, 8],
              size=(int(a[0, 9]), int(a[0, 10])))
    return ob, session


# ----------------------------------------------------------------------------
# ground plane (handoff 2.3, 3.4)
# ----------------------------------------------------------------------------
def ground_point(u, v, K, h, theta):
    """Undistorted pixel -> (D forward, X lateral, Zc optical-axis depth) of its floor point.

    Camera at height h, pitched down by theta. For u = cx this is D = h / tan(theta + alpha).
    Zc (not the Euclidean range sqrt(D^2 + h^2)) is the depth that converts pixels to metres.
    """
    a = (np.asarray(u, float) - K[0, 2]) / K[0, 0]
    b = (np.asarray(v, float) - K[1, 2]) / K[1, 1]
    st, ct = math.sin(theta), math.cos(theta)
    den = st + b * ct
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(den > 1e-9, h / den, np.nan)
    return t * (ct - b * st), t * a, t


def object_height(u_t, v_t, D, X, K, h, theta):
    """Intersect the top-row ray with the vertical line above the contact point (D, X)."""
    a = (u_t - K[0, 2]) / K[0, 0]
    b = (v_t - K[1, 2]) / K[1, 1]
    st, ct = math.sin(theta), math.cos(theta)
    F = ct - b * st
    s = (a * X + F * D) / (a * a + F * F)
    return h + s * (-st - b * ct)


def world_to_pixel(P, K, h, theta):
    """World point (X right, D forward, z up; camera at origin, floor at z = -h) -> pixel."""
    st, ct = math.sin(theta), math.cos(theta)
    P = np.atleast_2d(P)
    xc = P[:, 0]
    yc = -st * P[:, 1] - ct * P[:, 2]
    zc = ct * P[:, 1] - st * P[:, 2]
    return np.stack([K[0, 2] + K[0, 0] * xc / zc, K[1, 2] + K[1, 1] * yc / zc], 1), zc


def ground_row_table(K, height, h, theta, sigma_h, sigma_theta, sigma_v=1.0):
    """Sensitivities of D to h, theta and the pixel row across image rows (handoff 3.4)."""
    fy, cy = K[1, 1], K[1, 2]
    v_hor = cy - fy * math.tan(theta)  # horizon row
    rows = []
    for v in np.linspace(max(0, v_hor + 2), height - 1, 60):
        alpha = math.atan((v - cy) / fy)
        s = math.sin(theta + alpha)
        D = h / math.tan(theta + alpha)
        dD_dh = D / h
        dD_dth = -h / s ** 2
        dD_dv = dD_dth * math.cos(alpha) ** 2 / fy
        sig = math.sqrt((dD_dh * sigma_h) ** 2 + (dD_dth * sigma_theta) ** 2 + (dD_dv * sigma_v) ** 2)
        rows.append(dict(v=v, D=D, dD_dh=dD_dh, dD_dtheta=dD_dth, dD_dv=dD_dv, sigma_D=sig,
                         rel=sig / D))
    return dict(v_horizon=v_hor, rows=rows)


def load_ground(calib):
    r = latest_result('ground_cal', ref=calib.id)
    if r is None:
        raise SystemExit(f'no ground calibration for calibration #{calib.id}: run `ground --setup`')
    return r['data']


# ----------------------------------------------------------------------------
# image processing: tape blobs, objects on the floor (handoff 2.2, 2.4)
# ----------------------------------------------------------------------------
HSV_PRESETS = {
    'red': [[[0, 100, 70], [9, 255, 255]], [[170, 100, 70], [179, 255, 255]]],
    'orange': [[[9, 120, 100], [21, 255, 255]]],
    'yellow': [[[21, 90, 100], [34, 255, 255]]],
    'green': [[[40, 70, 50], [85, 255, 255]]],
    'blue': [[[95, 100, 50], [128, 255, 255]]],
    'pink': [[[140, 60, 80], [170, 255, 255]]],
}


def tape_ranges(args):
    if getattr(args, 'color', None):
        if args.color not in HSV_PRESETS:
            raise SystemExit(f'--color must be one of {list(HSV_PRESETS)} (or click the tape to pick)')
        return HSV_PRESETS[args.color]
    return cfg_get('tape_hsv', HSV_PRESETS['red'])


def pick_hsv(bgr, x, y, r=4):
    hsv = cv2.cvtColor(bgr[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1], cv2.COLOR_BGR2HSV)
    h, s, v = np.median(hsv.reshape(-1, 3), 0).astype(int)
    slo, vlo = max(50, s - 70), max(40, v - 80)
    lo, hi = h - 10, h + 10
    if lo < 0:
        return [[[0, slo, vlo], [hi, 255, 255]], [[180 + lo, slo, vlo], [179, 255, 255]]]
    if hi > 179:
        return [[[lo, slo, vlo], [179, 255, 255]], [[0, slo, vlo], [hi - 180, 255, 255]]]
    return [[[lo, slo, vlo], [hi, 255, 255]]]


def tape_mask(bgr, ranges):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def find_tape_blobs(bgr, ranges, min_area=25, k=2):
    """Largest k colour blobs: list of (u, v, area), sorted by u then v.

    Blobs touching the image border are dropped: a cut-off blob has a biased centroid.
    """
    n, _, stats, cents = cv2.connectedComponentsWithStats(tape_mask(bgr, ranges))
    H, W = bgr.shape[:2]
    x, y, w, h = (stats[:, i] for i in range(4))
    inside = (x > 1) & (y > 1) & (x + w < W - 1) & (y + h < H - 1)
    idx = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area and inside[i]]
    idx = sorted(idx, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:k]
    blobs = [(float(cents[i, 0]), float(cents[i, 1]), int(stats[i, cv2.CC_STAT_AREA])) for i in idx]
    return sorted(blobs, key=lambda b: (b[0], b[1]))


def gray_mean(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


@dataclass
class SegParams:
    tau_chroma: float = 0.035  # (r, g) chromaticity change that counts as object
    dark: float = 0.35         # brightness ratio below this is object, not shadow
    bright: float = 1.30       # brightness ratio above this is object
    shadow: float = 0.92       # darker than this but same colour is shadow
    min_area: int = 400


def segment(frame, bg, sp):
    """Foreground vs stored empty floor in normalized chromaticity (handoff 2.4).

    Returns (object_mask, shadow_mask). Shadows change brightness much more than (r, g).
    """
    f = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32) + 1
    b = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32) + 1
    sf, sb = f.sum(2), b.sum(2)
    rg_f = np.stack([f[..., 2] / sf, f[..., 1] / sf], -1)
    rg_b = np.stack([b[..., 2] / sb, b[..., 1] / sb], -1)
    dchroma = np.linalg.norm(rg_f - rg_b, axis=-1)
    ratio = sf / sb
    lit = sb > 3 * 25  # chromaticity is noise in very dark floor pixels
    obj = ((dchroma > sp.tau_chroma) & lit) | (ratio > sp.bright) | (ratio < sp.dark)
    shadow = ~obj & (ratio < sp.shadow)
    obj = obj.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k, iterations=2)
    return obj, shadow


def detect_objects(frame, bg, sp):
    obj, shadow = segment(frame, bg, sp)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < sp.min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        pts = c.reshape(-1, 2)
        v_b = float(pts[:, 1].max())
        u_b = float(pts[pts[:, 1] >= v_b - 3, 0].mean())  # bottom-centre contact point
        v_t = float(pts[:, 1].min())
        u_t = float(pts[pts[:, 1] <= v_t + 3, 0].mean())
        m = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        ring = cv2.dilate(m, np.ones((25, 25), np.uint8)) & ~m
        out.append(dict(contour=c, bbox=(x, y, w, h), area=area, v_b=v_b, u_b=u_b, v_t=v_t, u_t=u_t,
                        b_obj=float(gray[m > 0].mean()), b_sur=float(gray[ring > 0].mean())
                        if ring.any() else float(gray.mean())))
    return sorted(out, key=lambda o: -o['area']), shadow


def object_geometry(o, K, g):
    D, X, Zc = ground_point(o['u_b'], o['v_b'], K, g['h'], g['theta'])
    D, X, Zc = float(D), float(X), float(Zc)
    W = o['bbox'][2] * Zc / K[0, 0] if np.isfinite(Zc) else float('nan')
    H = float(object_height(o['u_t'], o['v_t'], D, X, K, g['h'], g['theta'])) if np.isfinite(D) else float('nan')
    return dict(D=D, X=X, Zc=Zc, W=W, H=H)


FEATURES = ['u0', 'v0', 'u1', 'v1', 'v_b', 'D_geo', 'W_geo', 'H_geo',
            'bright_obj', 'bright_sur', 'area', 'aspect']


def object_features(o, geo, size):
    Wi, Hi = size
    x, y, w, h = o['bbox']
    return [x / Wi, y / Hi, (x + w) / Wi, (y + h) / Hi, o['v_b'] / Hi, geo['D'], geo['W'], geo['H'],
            o['b_obj'] / 255, o['b_sur'] / 255, o['area'] / (Wi * Hi), h / max(w, 1)]


def object_crop(frame, o, size=64):
    x, y, w, h = o['bbox']
    s = int(max(w, h) * 1.15) + 2
    cx, cy = x + w // 2, y + h // 2
    pad = cv2.copyMakeBorder(frame, s, s, s, s, cv2.BORDER_REPLICATE)
    crop = pad[cy - s // 2 + s:cy - s // 2 + 2 * s, cx - s // 2 + s:cx - s // 2 + 2 * s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# live window helpers
# ----------------------------------------------------------------------------
def draw_text(img, lines, org=(8, 22), scale=0.55, color=(255, 255, 255)):
    """Lines of text on a translucent dark panel (an outline drawn thicker would not line up:
    Hershey glyph advances grow with stroke thickness)."""
    lines = [ln for ln in lines if ln]
    if not lines:
        return
    x, y = org
    step = int(24 * scale / 0.55)
    w = max(cv2.getTextSize(ln, FONT, scale, 1)[0][0] for ln in lines)
    x0, y0 = max(0, x - 4), max(0, y - step + 6)
    x1, y1 = min(img.shape[1], x + w + 4), min(img.shape[0], y + step * (len(lines) - 1) + 7)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.45).astype(np.uint8)
    for ln in lines:
        cv2.putText(img, ln, (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += step


class Live:
    """Camera + newest-frame reader + optional undistortion + an OpenCV window."""

    def __init__(self, args, title, calib=None, window=True):
        self.args, self.title, self.calib, self.window = args, title, calib, window
        self.cam = Camera(resolve_url(args), getattr(args, 'backend', 'requests'))
        global CAM_SOURCE
        CAM_SOURCE = self.cam.kind
        if self.cam.kind == 'esp32' and not getattr(args, 'no_lock', False):
            self.aec = args.aec if args.aec is not None else cfg_get('aec_value', 300)
            self.agc = args.agc if args.agc is not None else cfg_get('agc_gain', 0)
            self.quality = args.quality if getattr(args, 'quality', None) is not None else cfg_get('quality', 10)
            cfg_set('aec_value', self.aec)
            cfg_set('agc_gain', self.agc)
            cfg_set('quality', self.quality)
            try:
                self.cam.lock(self.aec, self.agc, getattr(args, 'framesize', None), self.quality)
            except OSError as e:  # requests' connection errors and timeouts derive from OSError
                raise SystemExit(f'cannot reach the ESP32-CAM at {self.cam.base}: {e}\n'
                                 'check the IP printed on the serial monitor, or pass --no-lock')
        self.auto = False
        self.reader = LatestFrame(self.cam.frames, retry=0.2 if self.cam.backend == 'capture' else 1.0)
        self.last, self.maps, self.fps, self._tprev = 0, None, 0.0, None
        self.brightness, self.age, self.click, self.n = 0.0, 0.0, None, 0
        self.max_frames = getattr(args, 'max_frames', None)
        if window:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(title, self._mouse)
        self.locked = self.cam.kind == 'esp32' and not getattr(args, 'no_lock', False)
        if self.locked and args.aec is None:
            try:  # the serial camera runs at a few fps: give white balance longer to settle (no green cast)
                self.auto_lock(settle=5.0 if self.cam.transport == 'serial' else 2.5)
            except (OSError, SystemExit) as e:  # slow or dropped link: keep the saved manual values
                print(f'warning: could not match exposure to the room ({str(e)[:80]}); '
                      f'using aec_value {self.aec}, agc_gain {self.agc}')
                try:
                    self.cam.lock(self.aec, self.agc, getattr(args, 'framesize', None), self.quality)
                except OSError:
                    pass

    def auto_lock(self, settle=2.5):
        """Lock exposure, gain and white balance at what auto mode chooses for this room.

        Auto exposure and white balance run for `settle` s; the white-balance gains are then
        frozen where they ended, and a manual exposure x gain is searched that reproduces the
        brightness auto mode reached. After that nothing changes between frames (handoff 1.2).
        """
        cam = self.cam
        print('matching the locked exposure to the room (auto exposure settles first) ...')
        for var in ('aec', 'agc', 'awb', 'awb_gain'):
            cam.control(var, 1)
        t0, bs = time.time(), []
        while time.time() - t0 < settle or len(bs) < 3:
            self.frame(timeout=8)
            bs.append(self.brightness)
        target = float(np.mean(bs[-3:]))
        cam.control('awb', 0)  # AWB off keeps its last gains (awb_gain stays on)
        cam.control('aec', 0)
        cam.control('agc', 0)
        aec, gi, b = int(np.clip(self.aec, 1, 1200)), 0, float('nan')
        for _ in range(6):
            cam.control('aec_value', aec)
            cam.control('agc_gain', gi)
            for _ in range(2):  # the sensor applies new values from the next frame on
                self.frame(timeout=8)
            b = float(np.mean([self.frame(timeout=8) is not None and self.brightness for _ in range(2)]))
            if abs(b - target) <= 0.1 * target + 2:
                break
            need = aec * AGC_GAINS[gi] * target / max(b, 1.0)  # exposure x gain wanted
            if need <= 1200:
                aec, gi = max(1, int(round(need))), 0
            else:
                aec, gi = 1200, int(np.argmin([abs(g - need / 1200) for g in AGC_GAINS]))
        self.aec, self.agc = aec, gi
        cfg_set('aec_value', aec)
        cfg_set('agc_gain', gi)
        cam.settings.update(aec=0, aec_value=aec, agc=0, agc_gain=gi, awb=0)
        try:
            back = cam.status()
        except OSError:
            back = {}
        db().execute('INSERT INTO camera_settings(ts,url,requested,readback) VALUES(?,?,?,?)',
                     (time.time(), cam.base, jdump(dict(cam.settings, auto_lock_target=target)), jdump(back)))
        db().commit()
        print(f'locked at aec_value {aec}, agc_gain {gi} (x{AGC_GAINS[gi]:g}): brightness {b:.0f}, '
              f'auto mode reached {target:.0f}')

    def _relock_if_reset(self):
        """After a long stall, re-apply the lock if the board restarted (its settings reset)."""
        try:
            st = self.cam.status()
        except (OSError, ValueError):
            return
        if int(st.get('framesize', -1)) != int(self.cam.settings.get('framesize', -2)) or \
                (not self.auto and int(st.get('aec', 0)) != 0):
            print('board restarted: re-applying the locked settings')
            self.cam.lock(self.aec, self.agc, self.cam.settings.get('framesize'), self.quality)

    def _mouse(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def frame(self, timeout=None):
        """Next new frame. With a window and no timeout, a stalled camera (WiFi hiccup, board
        reset) is waited out with a 'reconnecting' notice; q or Esc stops."""
        if timeout is not None or not self.window:
            got = self.reader.get(after=self.last, timeout=timeout or 8.0)
            if got is None:
                raise SystemExit(f'no frame from {self.cam.url} within {timeout or 8.0:.0f} s ({self.reader.err})')
        else:
            t0 = time.time()
            while (got := self.reader.get(after=self.last, timeout=0.5)) is None:
                waited = time.time() - t0
                if waited > 2:
                    last = self.reader.frame
                    disp = np.zeros((600, 800, 3), np.uint8) if last is None else (last * 0.35).astype(np.uint8)
                    draw_text(disp, [f'no frame for {waited:.0f} s - camera reconnecting...  (q quits)',
                                     f'{self.reader.err or ""}'[:90]], org=(10, 30), color=(80, 200, 255))
                    cv2.imshow(self.title, disp)
                if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27) or (self.max_frames and time.time() - t0 > 10):
                    raise SystemExit(f'stopped while waiting for {self.cam.url} ({self.reader.err})')
            if time.time() - t0 > 3 and getattr(self, 'locked', False):
                self._relock_if_reset()
        seq, t, img = got
        self.last = seq
        self.n += 1
        if self.calib is not None:
            h, w = img.shape[:2]
            if (w, h) != (self.calib.width, self.calib.height) and getattr(self.args, 'calib', None) is None:
                other = db().execute('SELECT id FROM calibrations WHERE width=? AND height=? ORDER BY id DESC LIMIT 1',
                                     (w, h)).fetchone()
                if other:  # e.g. phone stream vs ESP32 calibration: use the newest one for this size
                    self.calib = load_calib(other[0])
                    print(f'frame is {w}x{h}: using calibration #{self.calib.id} ({self.calib.method})')
            if (w, h) != (self.calib.width, self.calib.height):
                raise SystemExit(f'frame is {w}x{h} but calibration #{self.calib.id} is for '
                                 f'{self.calib.width}x{self.calib.height}: recalibrate (handoff 1.2)')
            if self.maps is None:
                K = self.calib.K
                self.maps = cv2.initUndistortRectifyMap(K, self.calib.dist, None, K, (w, h), cv2.CV_16SC2)
            img = cv2.remap(img, *self.maps, cv2.INTER_LINEAR)
        now = time.time()
        if self._tprev:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._tprev, 1e-3)
        self._tprev = now
        self.age = now - t
        self.brightness = gray_mean(img)
        return img

    def status_line(self, img):
        h, w = img.shape[:2]
        return f'{w}x{h}  {self.fps:4.1f} fps  age {self.age * 1000:3.0f} ms  brightness {self.brightness:5.1f}'

    def show(self, img, lines=(), wait=1):
        draw_text(img, lines)
        cv2.imshow(self.title, img)
        k = cv2.waitKey(wait)
        if self.max_frames and self.n >= self.max_frames:  # testing aid
            return ord('q')
        return k & 0xFF if k >= 0 else -1

    def ask(self, img, label):
        """Type a number in the window. Enter: value (or None if empty), Esc: cancel ('esc')."""
        buf = ''
        if self.max_frames:
            return None
        while True:
            disp = img.copy()
            cv2.rectangle(disp, (0, disp.shape[0] - 46), (disp.shape[1], disp.shape[0]), (40, 40, 40), -1)
            draw_text(disp, [f'{label}: {buf}_   (Enter ok, Esc cancel)'], org=(10, disp.shape[0] - 16),
                      scale=0.65, color=(120, 255, 255))
            cv2.imshow(self.title, disp)
            k = cv2.waitKey(30)
            if k < 0:
                continue
            k &= 0xFF
            if k in (13, 10, 141):
                if not buf:
                    return None
                try:
                    return float(buf)
                except ValueError:
                    buf = ''
            elif k == 27:
                return 'esc'
            elif k in (8, 127):
                buf = buf[:-1]
            elif chr(k) in '0123456789.-':
                buf += chr(k)

    def close(self):
        self.reader.stop()
        if self.window:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)

    def adjust(self, key):
        """e/E exposure down/up by 25 %, g/G gain -/+1, a auto exposure + white balance on/off.

        Auto is for looking around only: measuring commands always start locked (handoff 1.2).
        Manual values are remembered for later runs.
        """
        if key == 'a':
            self.auto = not getattr(self, 'auto', False)
            for var in ('aec', 'agc', 'awb', 'awb_gain'):
                self.cam.control(var, int(self.auto))
            if not self.auto:  # back to the locked manual values
                self.cam.control('aec_value', self.aec)
                self.cam.control('agc_gain', self.agc)
            return
        if getattr(self, 'auto', False):
            return
        if key in 'eE':
            self.aec = int(np.clip(max(self.aec + 10, self.aec * 1.25) if key == 'E' else self.aec / 1.25, 0, 1200))
            self.cam.control('aec_value', self.aec)
        elif key in 'gG':
            self.agc = int(np.clip(self.agc + (1 if key == 'G' else -1), 0, 30))
            self.cam.control('agc_gain', self.agc)
        cfg_set('aec_value', self.aec)
        cfg_set('agc_gain', self.agc)


def capture_tape(live, ranges, min_area, n=10):
    """Average two tape blob positions over n fresh frames. Returns dict or None."""
    pts, bright, frame = [], [], None
    for _ in range(n * 3):
        frame = live.frame()
        blobs = find_tape_blobs(frame, ranges, min_area)
        if len(blobs) == 2:
            pts.append([blobs[0][:2], blobs[1][:2]])
            bright.append(live.brightness)
        if len(pts) >= n:
            break
    if len(pts) < max(3, n // 2):
        return None
    P = np.array(pts)
    l = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    med = np.median(P, 0)
    return dict(P1=med[0], P2=med[1], l=float(np.linalg.norm(med[1] - med[0])),
                l_std=float(l.std(ddof=1)) if len(l) > 1 else 0.0, n=len(l),
                brightness=float(np.mean(bright)), frame=frame)


def draw_blobs(img, blobs, color=(0, 255, 255)):
    for u, v, a in blobs:
        cv2.circle(img, (int(round(u)), int(round(v))), max(4, int(math.sqrt(a / math.pi))), color, 2)
    if len(blobs) == 2:
        cv2.line(img, tuple(int(round(x)) for x in blobs[0][:2]),
                 tuple(int(round(x)) for x in blobs[1][:2]), color, 1, cv2.LINE_AA)


def handle_click(live, img):
    if live.click is None:
        return None
    x, y = live.click
    live.click = None
    rng = pick_hsv(img, x, y)
    cfg_set('tape_hsv', rng)
    print(f'tape colour picked at ({x},{y}): HSV ranges {rng}')
    return rng


# ----------------------------------------------------------------------------
# self-checks (handoff 7)
# ----------------------------------------------------------------------------
def board_points(cols, rows, square):
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return g


def gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def check_s1():
    rng = np.random.default_rng(1)
    W, H = 800, 600
    K = np.array([[612.0, 0, 405.0], [0, 608.0, 297.0], [0, 0, 1]])
    dist = np.array([-0.21, 0.06, 0, 0, 0])
    objp = board_points(9, 6, 0.025)
    objp -= objp.mean(0)
    objs, imgs = [], []
    while len(objs) < 20:
        rvec = rng.normal(0, 0.35, 3)
        tvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.11, 0.11), rng.uniform(0.35, 0.75)])
        R = cv2.Rodrigues(rvec)[0]
        if ((objp @ R.T + tvec)[:, 2] < 0.1).any():
            continue
        pts = cv2.projectPoints(objp, rvec, tvec, K, dist)[0].reshape(-1, 2)
        if (pts < 5).any() or (pts[:, 0] > W - 5).any() or (pts[:, 1] > H - 5).any():
            continue
        objs.append(objp)
        imgs.append((pts + rng.normal(0, 0.02, pts.shape)).astype(np.float32))
    rms, K2, d2, _, _ = cv2.calibrateCamera(objs, imgs, (W, H), None, None,
                                            flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    true = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=dist[0])
    got = dict(fx=K2[0, 0], fy=K2[1, 1], cx=K2[0, 2], cy=K2[1, 2], k1=d2.ravel()[0])
    err = {k: abs(got[k] / true[k] - 1) for k in true}
    worst = max(err.values())
    return worst < 0.005, worst, f'max rel err {worst:.2e} ({max(err, key=err.get)}), rms {rms:.3f} px'


def check_s2():
    rng = np.random.default_rng(2)
    f, delta, L = 620.0, 0.035, 0.30
    ob = ruler_synth(f, delta, 0.0, (399.5, 299.5), L, np.linspace(0.4, 2.5, 20), (0, 0),
                     noise=0.05, rng=rng)
    fit = fit_ruler(ob, ('f', 'delta'))
    ef = abs(fit['params']['f'] / f - 1)
    ed = abs(fit['params']['delta'] / delta - 1)
    worst = max(ef, ed)
    return worst < 0.01, worst, (f"f {fit['params']['f']:.2f} (true {f}), delta {fit['params']['delta'] * 1e3:.2f} mm "
                                 f"(true {delta * 1e3:.1f}), rel err f {ef:.1e} delta {ed:.1e}")


def check_s3():
    w, wt = gauss_legendre(0.0, 1.0, 30)
    sq = np.sqrt(wt)
    # analytic columns w and w^2 (handoff 3.2)
    rho_a = retention(sq * w, (sq * w ** 2)[:, None])[2]
    # the same through the actual ruler model and its numerical Jacobian
    ob = ruler_synth(600.0, 0.0, 0.0, (399.5, 299.5), 0.3, 1 / w, (0, 0))
    J = ruler_jacobian(ob, dict(f=600.0, delta=0.0, k1=0.0), ('f', 'delta')) * sq[:, None]
    rho_m = retention(J[:, 0], J[:, 1:])[2]
    err = max(abs(rho_a - 1 / 16), abs(rho_m - 1 / 16))
    r05 = rho_uniform_w(4.0)
    return err < 1e-6, err, (f'rho analytic {rho_a:.9f}, via ruler model {rho_m:.9f}, 1/16 = 0.0625; '
                             f'Z in [0.5, 2] m: r^2 {1 - r05:.4f}, rho {r05:.4f}')


def check_s4():
    r, wt = gauss_legendre(0.0, 1.0, 30)
    sq = np.sqrt(wt)
    rho_a = retention(sq * r, (sq * r ** 3)[:, None])[2]
    # point model: radius_px = f r (1 + k1 r^2), numerical Jacobian at k1 = 0
    proj = lambda f, k1: f * r * (1 + k1 * r ** 2)
    f0, hf, hk = 600.0, 6e-3, 1e-6
    jf = (proj(f0 + hf, 0) - proj(f0 - hf, 0)) / (2 * hf)
    jk = (proj(f0, hk) - proj(f0, -hk)) / (2 * hk)
    rho_m = retention(sq * jf, (sq * jk)[:, None])[2]
    err = max(abs(rho_a - 4 / 25), abs(rho_m - 4 / 25))
    return err < 1e-6, err, f'rho analytic {rho_a:.9f}, via projection {rho_m:.9f}, 4/25 = 0.16'


def check_s5():
    rng = np.random.default_rng(5)
    worst = 0.0
    for n, p in [(50, 1), (50, 3), (200, 6), (12, 4)]:
        b = rng.normal(size=n)
        A = rng.normal(size=(n, p)) @ np.diag(np.logspace(0, 2, p))
        _, i_marg, _ = retention(b, A)
        schur = b @ b - b @ A @ np.linalg.solve(A.T @ A, A.T @ b)
        worst = max(worst, abs(i_marg - schur) / abs(schur))
    return worst < 1e-9, worst, f'max rel diff projector vs Schur complement {worst:.1e}'


def check_s6():
    K = np.array([[605.0, 0, 398.0], [0, 598.0, 305.0], [0, 0, 1]])
    h, theta = 0.412, 0.21
    D, X = np.meshgrid(np.linspace(0.4, 5.0, 40), np.linspace(-1.0, 1.0, 21))
    P = np.stack([X.ravel(), D.ravel(), np.full(D.size, -h)], 1)
    uv, zc = world_to_pixel(P, K, h, theta)
    ok = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 800) & (uv[:, 1] >= 0) & (uv[:, 1] < 600)
    D2, X2, _ = ground_point(uv[ok, 0], uv[ok, 1], K, h, theta)
    err_d = np.abs(D2 - P[ok, 1]).max()
    # centre column against the handoff formula D = h / tan(theta + alpha)
    Pc = np.stack([np.zeros(30), np.linspace(0.6, 4, 30), np.full(30, -h)], 1)
    uvc, _ = world_to_pixel(Pc, K, h, theta)
    alpha = np.arctan((uvc[:, 1] - K[1, 2]) / K[1, 1])
    err_f = np.abs(h / np.tan(theta + alpha) - Pc[:, 1]).max()
    # a 0.25 m tall, 0.12 m wide box face at D = 1.5, X = 0.2
    box = np.array([[0.14, 1.5, -h], [0.26, 1.5, -h], [0.2, 1.5, -h + 0.25]])
    buv, _ = world_to_pixel(box, K, h, theta)
    Db, Xb, Zc = ground_point((buv[0, 0] + buv[1, 0]) / 2, buv[0, 1], K, h, theta)
    Hb = object_height(buv[2, 0], buv[2, 1], Db, Xb, K, h, theta)
    Wb = (buv[1, 0] - buv[0, 0]) * Zc / K[0, 0]
    err = max(err_d, err_f, abs(Hb - 0.25), abs(Wb - 0.12))
    return err < 1e-3, err, (f'max |D err| {err_d:.1e} m over {ok.sum()} floor points, formula {err_f:.1e} m, '
                             f'box H err {abs(Hb - 0.25):.1e} W err {abs(Wb - 0.12):.1e}')


def check_s7():
    def source():
        i = 0
        while True:
            i += 1
            time.sleep(0.004)  # ~250 fps producer
            yield i

    rd = LatestFrame(source)
    lags, last, t_end = [], 0, time.time() + 1.0
    while time.time() < t_end:
        got = rd.get(after=last, timeout=1.0)
        if got is None:
            break
        seq, _, frame = got
        lags.append(rd.seq - seq)
        last = seq
        assert frame == seq
        time.sleep(0.03)  # slow consumer: 30 ms per frame
    produced = rd.seq
    rd.stop()
    worst = max(lags) if lags else 99
    return worst < 2, worst, (f'{len(lags)} frames consumed of {produced} produced, max lag {worst} frames '
                              f'(a FIFO queue would lag ~{produced - len(lags)})')


def synth_room(K, dist, size, C, target, panels, square=0.010):
    """Render a checkerboard room: panels = [(origin, u, v, extent_u, extent_v)] in world
    coordinates [m]; camera at C looking at target, world z up. Returns (gray image, R, t)."""
    f = (target - C) / np.linalg.norm(target - C)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(f, right), f])  # world -> camera
    W, H = size
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    n = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))] @ R  # ray directions in world coordinates
    best = np.full(len(d), np.inf)
    val = np.full(len(d), 0.35)
    for O, u, v, eu, ev, *shade in panels:  # optional shade: a plain face (an object) instead of squares
        nrm = np.cross(u, v)
        den = d @ nrm
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = ((O - C) @ nrm) / den
        P = C + d * lam[:, None]
        a, b = (P - O) @ u, (P - O) @ v
        hit = (lam > 0) & (lam < best) & (a >= 0) & (a <= eu) & (b >= 0) & (b <= ev)
        best[hit] = lam[hit]
        val[hit] = shade[0] if shade else np.where((np.floor(a[hit] / square) + np.floor(b[hit] / square)) % 2 == 0,
                                                  0.1, 0.9)
    img = cv2.GaussianBlur((val.reshape(H, W) * 255).astype(np.uint8), (0, 0), 0.7)
    return img, R, -R @ C


def check_s8():
    """Room self-calibration from one rendered frame whose panels are not square to each other."""
    K = np.array([[600.0, 0, 404.0], [0, 600.0, 296.0], [0, 0, 1]])
    dist = np.array([-0.08, 0.02, 0, 0, 0])
    a2, a3 = math.radians(2), math.radians(3)
    panels = [(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.25, 0.25),  # floor
              (np.zeros(3), np.array([0, 1.0, 0]), np.array([-math.sin(a2), 0, math.cos(a2)]), 0.25, 0.29),
              (np.zeros(3), np.array([math.cos(a3), math.sin(a3), 0]), np.array([0, 0, 1.0]), 0.25, 0.29)]
    img, _, _ = synth_room(K, dist, (800, 600), np.array([0.40, 0.36, 0.30]), np.array([0.07, 0.07, 0.07]), panels)
    planes = find_planes(img)
    c = calib_from_planes(planes, (800, 600)) if planes else None
    if c is None:
        return False, float('nan'), f'{len(planes)} planes found, no calibration'
    poses = plane_poses(planes, c['K'], c['dist'])
    nrm = [p['n'] for p in poses]
    groups = plane_groups(nrm)
    reps = [nrm[groups.index(g)] for g in sorted(set(groups))]
    got = sorted(math.degrees(math.acos(min(1, abs(float(reps[i] @ reps[j])))))
                 for i in range(len(reps)) for j in range(i + 1, len(reps)))
    true = sorted([88.0, 90.0, 87.0])  # floor-left wall, floor-right wall, wall-wall (normals, acute)
    ef = abs(c['K'][0, 0] / K[0, 0] - 1)
    ea = max(abs(x - y) for x, y in zip(got, true)) if len(got) == 3 else float('inf')
    return ef < 0.01 and ea < 0.3, ef, (f"{len(planes)} planes, f {c['K'][0, 0]:.1f} (true 600, rel err {ef:.1e}), "
                                        f"rms {c['rms']:.2f} px, angles {', '.join(f'{x:.2f}' for x in got)} "
                                        f"(true {', '.join(f'{x:.0f}' for x in true)})")


def check_s9():
    """Upright shapes on the floor: fitting the outline recovers length, width, height."""
    K = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    dist = np.zeros(5)
    rvec = np.array([[-2.1], [0.0], [0.0]])  # camera looking down at the floor at about 30 deg
    R = cv2.Rodrigues(rvec)[0]
    tvec = np.array([[0.0], [0.02], [0.32]])
    pose = dict(rvec=rvec, tvec=tvec, R=R, t=tvec.ravel())
    up = 1.0 if (-R.T @ tvec.ravel())[2] >= 0 else -1.0
    worst, detail = 0.0, []
    for shape, (L, W, Hh, yaw) in (('box', (0.120, 0.064, 0.040, 0.5)), ('round', (0.120, 0.064, 0.040, 0.9)),
                                   ('cylinder', (0.090, 0.090, 0.182, 0.3)), ('sphere', (0.042, 0.042, 0.042, 0.0)),
                                   ('revolve', (0.090, 0.090, 0.170, 0.0)),
                                   ('box', (0.080, 0.080, 0.110, 0.2))):
        p = np.array([0.0, 0.0, yaw, math.log(L), math.log(W), math.log(Hh)])
        if shape == 'revolve':  # a mic: 90 mm base tapering to 55 mm at the top
            p = np.r_[p, np.log([1.0, 0.85, 0.72, 55 / 90])]
        uv = cv2.projectPoints(shape_points(p, shape, up), rvec, tvec, K, dist)[0].reshape(-1, 2)
        fit = fit_shape(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2), pose, K, dist, up, shape)
        e = max(abs(fit['L'] / max(L, W) - 1), abs(fit['W'] / min(L, W) - 1), abs(fit['H'] / Hh - 1))
        worst = max(worst, e)
        detail.append(f"{shape} {fit['L'] * 1e3:.0f}x{fit['W'] * 1e3:.0f}x{fit['H'] * 1e3:.0f}")
    return worst < 0.02, worst, f"max rel err {worst:.1e}: " + ', '.join(detail) + ' mm'


def check_s11():
    """Size estimability (projection identity on the outline residuals): one view of a box leaves
    little information on L and H (they trade off), a second view 60 degrees round restores it; a
    circular cylinder pins its height from one view."""
    K = np.array([[600.0, 0, 400], [0, 600, 300], [0, 0, 1]])

    def view(p, shape, az, el=0.6, r=0.35):
        Cp = np.array([r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el)])
        f = -Cp / np.linalg.norm(Cp)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(f, right), f])
        ps = dict(rvec=cv2.Rodrigues(R)[0], tvec=(-R @ Cp).reshape(3, 1))
        uv = cv2.projectPoints(shape_points(p, shape, 1.0), ps['rvec'], ps['tvec'], K, np.zeros(5))[0].reshape(-1, 2)
        return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2), ps

    pb = np.r_[0, 0, 0.4, np.log([0.08, 0.05, 0.06])]
    v1 = [view(pb, 'box', 0.3)]
    one = size_information(pb, 'box', 1.0, v1, K, np.zeros(5))
    two = size_information(pb, 'box', 1.0, v1 + [view(pb, 'box', 0.3 + math.radians(60))], K, np.zeros(5))
    pc = np.r_[0, 0, 0.4, np.log([0.09, 0.09, 0.182])]
    cyl = size_information(pc, 'cylinder', 1.0, [view(pc, 'cylinder', 0.3)], K, np.zeros(5))
    ok = one['H'][0] < 0.2 and two['H'][0] > 0.5 and two['H'][1] < one['H'][1] / 3 and cyl['H'][0] > 0.8
    return ok, one['H'][0], (f"box H: rho {one['H'][0]:.3f} from 1 view -> {two['H'][0]:.3f} from 2 "
                             f"(sd {one['H'][1] * 1e3:.2f} -> {two['H'][1] * 1e3:.2f} mm at 1 px); "
                             f"circular cylinder H from 1 view: rho {cyl['H'][0]:.3f}")


def check_s10(shape='box'):
    """Rendered room with a grey box on the floor, camera moved around it: found by the hidden
    pattern, fixed room-corner coordinates, joint fit over the views recovers its size."""
    K = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    X, Y, L, W, H = 0.13, 0.12, 0.08, 0.05, 0.06
    ex, ey, ez = np.eye(3)
    a2, a3 = math.radians(2), math.radians(3)
    panels = [(np.zeros(3), ex, ey, 0.25, 0.25), (np.zeros(3), ey, np.array([-math.sin(a2), 0, math.cos(a2)]), 0.25, 0.29),
              (np.zeros(3), np.array([math.cos(a3), math.sin(a3), 0]), ez, 0.25, 0.29),
              (np.array([X - L / 2, Y - W / 2, H]), ex, ey, L, W, 0.55),
              (np.array([X - L / 2, Y - W / 2, 0]), ex, ez, L, H, 0.45), (np.array([X - L / 2, Y + W / 2, 0]), ex, ez, L, H, 0.45),
              (np.array([X - L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5), (np.array([X + L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5)]
    room = Room(argparse.Namespace(shape=shape, object_class=None, truth=None, seg='room'), (800, 600), store=False)
    room.corrector = False  # the geometry is under test, not a trained corrector
    res = None
    for az in np.linspace(25, 65, 6):
        a = math.radians(az)
        img, _, _ = synth_room(K, np.zeros(5), (800, 600), np.array([X + 0.42 * math.cos(a), Y + 0.42 * math.sin(a), 0.28]),
                               np.array([0.10, 0.10, 0.04]), panels)
        res = room.process(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None)
    if not res['objects']:
        return False, float('nan'), 'box not found'
    o = res['objects'][0]
    err = max(abs(o['L'] - L), abs(o['W'] - W), abs(o['H'] - H), math.dist(o['p'][:2], (X, Y)))
    how = f"from {o['n_views']} views"
    return err < 0.006, err, (f"{o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x {o['H'] * 1e3:.0f} mm {how} "
                              f"(true 80 x 50 x 60), centre ({o['p'][0] * 1e3:.0f}, {o['p'][1] * 1e3:.0f}) mm "
                              f"(true 130, 120), max error {err * 1e3:.1f} mm")


def check_s13():
    """A rig of two cameras with different lenses and image sizes side by side (like the phone with the ESP32-CAM
    clamped to it): each finds its pose in the room corner coordinates, each fits the box to its own outline and
    the other's, so a single frame from each already gives two views."""
    KA = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    KB = np.array([[470.0, 0, 320.0], [0, 470.0, 240.0], [0, 0, 1]])
    X, Y, L, W, H = 0.13, 0.12, 0.08, 0.05, 0.06
    ex, ey, ez = np.eye(3)
    a2, a3 = math.radians(2), math.radians(3)
    panels = [(np.zeros(3), ex, ey, 0.25, 0.25), (np.zeros(3), ey, np.array([-math.sin(a2), 0, math.cos(a2)]), 0.25, 0.29),
              (np.zeros(3), np.array([math.cos(a3), math.sin(a3), 0]), ez, 0.25, 0.29),
              (np.array([X - L / 2, Y - W / 2, H]), ex, ey, L, W, 0.55),
              (np.array([X - L / 2, Y - W / 2, 0]), ex, ez, L, H, 0.45), (np.array([X - L / 2, Y + W / 2, 0]), ex, ez, L, H, 0.45),
              (np.array([X - L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5), (np.array([X + L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5)]
    ns = lambda: argparse.Namespace(shape='box', object_class=None, truth=None, seg='room')
    A, B = Room(ns(), (800, 600), store=False), Room(ns(), (640, 480), store=False)
    A.K, A.dist, B.K, B.dist = KA, np.zeros(5), KB, np.zeros(5)
    A.corrector = B.corrector = False
    ra = rb = None
    shot = lambda K, size, az: cv2.cvtColor(synth_room(K, np.zeros(5), size, np.array(
        [X + 0.42 * math.cos(math.radians(az)), Y + 0.42 * math.sin(math.radians(az)), 0.28]), np.array([0.10, 0.10, 0.04]),
        panels)[0], cv2.COLOR_GRAY2BGR)
    for _ in range(2):
        ra = A.process(shot(KA, (800, 600), 40), None, rig_views(B, rb))
        rb = B.process(shot(KB, (640, 480), 52), None, rig_views(A, ra))
    if not ra['objects'] or not rb['objects']:
        return False, float('nan'), 'box not found by both cameras'
    o = ra['objects'][0]
    err = max(abs(o['L'] - L), abs(o['W'] - W), abs(o['H'] - H), math.dist(o['p'][:2], (X, Y)))
    return o['n_views'] >= 2 and err < 0.006, err, (f"{o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x {o['H'] * 1e3:.0f} mm from "
                                                    f"{o['n_views']} views of 2 cameras (true 80 x 50 x 60), max error {err * 1e3:.1f} mm")


def check_s14():
    """CAD export: every shape family becomes a valid STEP solid whose volume (read back by FreeCAD from the file)
    matches the analytic volume of the fitted shape."""
    if shutil.which('freecadcmd') is None:
        return True, 0.0, 'FreeCAD is not installed: CAD export skipped (the OBJ mesh is still written)'
    import tempfile
    cases = []  # (shape, parameters, analytic volume in mm3)
    L, W, H = 0.12, 0.064, 0.04
    base = lambda sh, l, w, h: np.r_[0.1, 0.2, 0.5, np.log([l, w, h])]
    cases.append(('box', base('box', L, W, H), L * W * H))
    cases.append(('cylinder', base('cylinder', 0.09, 0.09, 0.182), math.pi / 4 * 0.09 ** 2 * 0.182))
    cases.append(('sphere', base('sphere', 0.042, 0.042, 0.042), math.pi / 6 * 0.042 ** 3))
    cases.append(('dome', base('dome', L, W, H), 2 / 3 * math.pi * (L / 2) * (W / 2) * H))
    d = np.array([1.0, 0.85, 0.72, 55 / 90]) * 0.09
    Hr = 0.17
    frustum = lambda r0, r1, h: math.pi * h / 3 * (r0 * r0 + r0 * r1 + r1 * r1)
    cases.append(('revolve', np.r_[base('revolve', 0.09, 0.09, Hr), np.log([1.0, 0.85, 0.72, 55 / 90])],
                  sum(frustum(d[i] / 2, d[i + 1] / 2, Hr / 3) for i in range(3))))
    pr = base('round', L, W, H)
    Vr, Qr = shape_mesh(pr, 'round', 1.0)
    cases.append(('round', pr, mesh_volume_area(Vr, Qr)[0]))
    with tempfile.TemporaryDirectory() as dd:
        items = [(os.path.join(dd, sh), cad_spec(dict(shape=sh, p=p, up=1.0), sh), 'selfcheck') for sh, p, _ in cases]
        got = write_cad(items)
        worst, bad, detail = 0.0, [], []
        for (sh, p, vol), g in zip(cases, got):
            e = abs(g['volume'] / (vol * 1e9) - 1)
            worst = max(worst, e)
            if not g['valid'] or g['solids'] != 1 or e > 0.01:
                bad.append(sh)
            detail.append(f"{sh} {g['volume'] / 1e3:.1f} cm3")
        files = all(os.path.exists(os.path.join(dd, sh + ext)) for sh, _, _ in cases for ext in ('.step', '.FCStd'))
    return not bad and files, worst, f"max volume error {worst:.1e} against the analytic shape: " + ', '.join(detail) \
        + (f"; failed: {bad}" if bad else '')


def check_s15():
    """Denoiser: the U-Net learns, from pairs of noisy copies of the same pictures (Noise2Noise, no clean target), to remove
    Gaussian noise of 10 grey levels; on a fresh picture its error against the clean one must fall to under half."""
    try:
        torch = _torch()
    except ImportError:  # the measurement itself does not need torch: only the denoiser does
        return True, 0.0, 'torch is not installed: denoiser check skipped'
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def picture(n=64):  # blocks and a gradient: edges to keep, flat areas to smooth
        img = np.tile(np.linspace(60, 180, n, dtype=np.float32), (n, 1))
        for _ in range(6):
            y, x, h, w = rng.integers(0, n - 12, 2).tolist() + rng.integers(8, 30, 2).tolist()
            img[y:y + h, x:x + w] = rng.uniform(30, 220)
        return img
    net = denoise_net(16).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(150):
        clean = [picture() for _ in range(16)]
        f = lambda v: torch.from_numpy(np.stack(v))[:, None].to(dev) / 255.0
        a = f([c + rng.normal(0, 10, c.shape).astype(np.float32) for c in clean])
        b = f([c + rng.normal(0, 10, c.shape).astype(np.float32) for c in clean])  # a second, independent noisy copy
        loss = torch.nn.functional.l1_loss(net(a), b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    c = picture()
    noisy = c + rng.normal(0, 10, c.shape).astype(np.float32)
    with torch.no_grad():
        y = net(torch.from_numpy(noisy).to(dev).float().div(255)[None, None])[0, 0].cpu().numpy() * 255
    e_in, e_out = float(np.sqrt(np.mean((noisy - c) ** 2))), float(np.sqrt(np.mean((y - c) ** 2)))
    return e_out < 0.5 * e_in, e_out / e_in, f'rms error {e_in:.1f} -> {e_out:.1f} grey levels after 150 steps on {dev}'


def check_s12():
    """Boards scattered on a table, no calibration given: f and k1 come from the boards being coplanar,
    and a box standing away from every board is sized (its mask supplied, the geometry is under test)."""
    f_true, dist = 600.0, np.array([-0.08, 0.02, 0, 0, 0])
    K = np.array([[f_true, 0, 399.5], [0, f_true, 299.5], [0, 0, 1.0]])
    X, Y, L, W, H, yaw = 0.12, 0.10, 0.06, 0.04, 0.05, 0.4
    ex, ey, ez = np.eye(3)
    boards = []
    for ox, oy, ang in ((-0.05, -0.03, 0.1), (0.20, -0.04, -0.25), (-0.04, 0.17, 0.4), (0.21, 0.17, -0.1)):
        c, s = math.cos(ang), math.sin(ang)
        boards.append((np.array([ox, oy, 0.0]), np.array([c, s, 0.0]), np.array([-s, c, 0.0]), 0.09, 0.12))
    c, s = math.cos(yaw), math.sin(yaw)
    ux, uy = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    O = np.array([X, Y, 0.0]) - ux * L / 2 - uy * W / 2
    faces = lambda sh: [(O + ez * H, ux, uy, L, W, sh), (O, ux, ez, L, H, sh * 0.8), (O + uy * W, ux, ez, L, H, sh * 0.8),
                        (O, uy, ez, W, H, sh * 0.9), (O + ux * L, uy, ez, W, H, sh * 0.9)]
    room = Room(argparse.Namespace(shape='box', object_class=None, truth=None, seg='salient', table=True, refine_f=False),
                (800, 600), store=False)
    room.corrector, cur, res = False, {}, None
    room.salient = lambda im: cur['mask']
    for az in (-100, -80, -65, -120, -90):
        a = math.radians(az)
        C, tgt = np.array([X + 0.30 * math.cos(a), Y + 0.30 * math.sin(a), 0.26]), np.array([0.12, 0.09, 0.0])
        img, _, _ = synth_room(K, dist, (800, 600), C, tgt, boards + faces(0.6))
        m, _, _ = synth_room(K, dist, (800, 600), C, tgt, faces(1.0))
        cur['mask'] = (m > 200).astype(np.float32)
        res = room.process(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None)
    if room.K is None or not res['objects']:
        return False, float('nan'), 'no focal length or no object'
    o = res['objects'][0]
    ef = abs(room.K[0, 0] / f_true - 1)
    err = max(abs(o['L'] - L), abs(o['W'] - W), abs(o['H'] - H))
    return ef < 0.02 and err < 0.002, err, (f"f {room.K[0, 0]:.0f} (true 600, k1 {room.dist[0]:.3f} true -0.080), object "
                                           f"{o['L'] * 1e3:.1f} x {o['W'] * 1e3:.1f} x {o['H'] * 1e3:.1f} mm (true 60 x 40 x 50)")


def check_s16():
    """Live wireframe: for every shape family the silhouette of the projected mesh is made of edges that face the camera,
    and a box seen from above and the side has both visible and hidden edges."""
    C = np.array([-0.25, -0.2, 0.2])
    zc = np.array([0.1, 0.2, 0.02]) - C
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    R = np.stack([xc, np.cross(zc, xc), zc])
    rvec = cv2.Rodrigues(R)[0]
    t = -R @ C
    pose = dict(R=R, t=t, rvec=rvec, tvec=t.reshape(3, 1))
    K = np.array([[600.0, 0, 400], [0, 600.0, 300], [0, 0, 1]])
    dist = np.zeros(5)
    bad, detail = [], []
    for sh in ('box', 'cylinder', 'sphere', 'dome', 'round'):
        p = np.r_[0.1, 0.2, 0.5, np.log([0.12, 0.064, 0.04])]
        if sh == 'sphere':
            p = np.r_[0.1, 0.2, 0.5, np.log([0.05] * 3)]
        o = dict(shape=sh, p=p, up=1.0)
        vis, hid = mesh_segments(o, pose, K, dist)
        V, Q = shape_mesh(p, sh, 1.0)
        uv = cv2.projectPoints(V, pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2)
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        ends = np.vstack([vis[:, :2], vis[:, 2:]])
        on_silhouette = all(np.min(np.linalg.norm(ends - h, axis=1)) < 1e-3 for h in hull)
        n_edges = len(mesh_edges(Q))
        if not on_silhouette or len(vis) + len(hid) != n_edges or len(vis) == 0:
            bad.append(sh)
        detail.append(f'{sh} {len(vis)} visible / {len(hid)} hidden')
        if sh == 'box' and len(hid) == 0:
            bad.append('box hidden edges')
    return not bad, float(len(bad)), ', '.join(detail) + (f'; failed: {bad}' if bad else '')


def check_s17():
    """Phone and ESP32-CAM as a rig with unequal eyes. (1) The second camera's own detector finds nothing: the phone's fitted
    box, projected with the second camera's pose, still gives it an outline (GrabCut from the prediction), and the two
    cameras fitted together recover the box. (2) A camera whose outline is five times noisier counts for 1/25 of the
    information: the weighted fit of one clean and one noisy outline is closer to the truth than the equal-weight fit,
    and the reported standard error of the length follows the weights."""
    KA = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    KB = np.array([[470.0, 0, 320.0], [0, 470.0, 240.0], [0, 0, 1]])
    X, Y, L, W, H = 0.13, 0.12, 0.08, 0.05, 0.06
    ex, ey, ez = np.eye(3)
    a2, a3 = math.radians(2), math.radians(3)
    panels = [(np.zeros(3), ex, ey, 0.25, 0.25), (np.zeros(3), ey, np.array([-math.sin(a2), 0, math.cos(a2)]), 0.25, 0.29),
              (np.zeros(3), np.array([math.cos(a3), math.sin(a3), 0]), ez, 0.25, 0.29),
              (np.array([X - L / 2, Y - W / 2, H]), ex, ey, L, W, 0.55),
              (np.array([X - L / 2, Y - W / 2, 0]), ex, ez, L, H, 0.45), (np.array([X - L / 2, Y + W / 2, 0]), ex, ez, L, H, 0.45),
              (np.array([X - L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5), (np.array([X + L / 2, Y - W / 2, 0]), ey, ez, W, H, 0.5)]
    ns = lambda seg: argparse.Namespace(shape='box', object_class=None, truth=None, seg=seg)
    shot = lambda K, size, az: cv2.cvtColor(synth_room(K, np.zeros(5), size, np.array(
        [X + 0.42 * math.cos(math.radians(az)), Y + 0.42 * math.sin(math.radians(az)), 0.28]), np.array([0.10, 0.10, 0.04]),
        panels)[0], cv2.COLOR_GRAY2BGR)
    A, B = Room(ns('room'), (800, 600), store=False), Room(ns('none'), (640, 480), store=False)
    A.K, A.dist, B.K, B.dist = KA, np.zeros(5), KB, np.zeros(5)
    A.corrector = B.corrector = False
    imA, imB = shot(KA, (800, 600), 40), shot(KB, (640, 480), 52)
    ra = A.process(imA, None, ())
    alone = B.process(imB, None, ())
    B.tracks, B.fits = [], []
    rb = B.process(imB, None, rig_views(A, ra), rig_priors(A, ra))
    if not ra['objects'] or alone['objects'] or not rb['objects'] or not rb['objects'][0].get('guided'):
        return False, float('nan'), (f"phone found {len(ra['objects'])}, second camera alone {len(alone['objects'])}, "
                                     f"with the phone's help {len(rb['objects'])}")
    o = rb['objects'][0]
    err1 = max(abs(o['L'] - L), abs(o['W'] - W), abs(o['H'] - H), math.dist(o['p'][:2], (X, Y)))
    # (2) weights, on exact projected outlines of the true box with noise added to one of them
    q = np.r_[X, Y, 0.0, np.log([L, W, H])]
    psA, psB = dict(ra['floor'], K=KA, dist=np.zeros(5)), dict(rb['floor'], K=KB, dist=np.zeros(5))
    proj = lambda ps: cv2.projectPoints(shape_points(q, 'box', 1.0), ps['rvec'], ps['tvec'], ps['K'], ps['dist'])[0].reshape(-1, 2)
    hA = cv2.convexHull(proj(psA).astype(np.float32)).reshape(-1, 2)
    hB0 = outline_samples(cv2.convexHull(proj(psB).astype(np.float32)).reshape(-1, 2), 64)
    rng = np.random.default_rng(3)
    e_w, e_u = [], []
    for _ in range(6):
        hB = hB0 + rng.normal(0, 5.0, hB0.shape)
        for e, sg in ((e_w, 5.0), (e_u, 1.0)):
            f = fit_shape(hA, dict(psA, sigma=1.0), KA, np.zeros(5), 1.0, 'box', [], [(hB, dict(psB, sigma=sg))])
            e.append(max(abs(f['L'] - L), abs(f['W'] - W), abs(f['H'] - H)) if f else 1.0)
    rep = []
    hB = hB0 + rng.normal(0, 5.0, hB0.shape)
    size_information(q, 'box', 1.0, [(hA, psA), (hB, psB)], KA, np.zeros(5), sigma_px=None, report=rep)
    ok = err1 < 0.008 and np.mean(e_w) < np.mean(e_u) and rep[1] > 3 * rep[0]
    return ok, err1, (f"second camera alone finds {len(alone['objects'])}, with the phone's prediction {o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x "
                      f"{o['H'] * 1e3:.0f} mm (true 80 x 50 x 60, max error {err1 * 1e3:.1f} mm); noisy outline of 5 px: mean error "
                      f"{np.mean(e_w) * 1e3:.1f} mm weighted, {np.mean(e_u) * 1e3:.1f} mm equal; sigma per view {rep[0]:.1f} / {rep[1]:.1f} px")


CHECKS = [('S1', 'checkerboard calibration recovers f, c, k1', check_s1),
          ('S2', 'ruler joint fit recovers f, delta', check_s2),
          ('S3', 'retention of f with delta, w ~ U(0,1] = 1/16', check_s3),
          ('S4', 'retention of f with k1, r ~ U[0,1] = 4/25', check_s4),
          ('S5', 'projector form = Schur complement', check_s5),
          ('S6', 'ground-plane distance on synthetic camera', check_s6),
          ('S7', 'stream reader keeps only the newest frame', check_s7),
          ('S8', 'room: one-frame calibration, panels not square', check_s8),
          ('S9', 'room: shape fit recovers L, W, H', check_s9),
          ('S10', 'room: object found and sized from several views', check_s10),
          ('S11', 'room: size estimability, one view vs two (projection identity)', check_s11),
          ('S12', 'table: scattered boards give f and k1, object away from them sized', check_s12),
          ('S13', 'room: two cameras of a rig fitted together', check_s13),
          ('S14', 'CAD export: STEP solids match the fitted shapes', check_s14),
          ('S15', 'denoiser learns to remove noise without clean targets', check_s15),
          ('S16', 'live wireframe: silhouette from edges facing the camera, hidden edges kept', check_s16),
          ('S17', 'rig: the phone guides the second camera, outlines weighted by their noise', check_s17)]


def run_selfcheck():
    res, ok_all = [], True
    for cid, desc, fn in CHECKS:
        t0 = time.time()
        try:
            ok, val, detail = fn()
        except Exception as e:
            ok, val, detail = False, float('nan'), f'{type(e).__name__}: {e}'
        ok_all &= bool(ok)
        res.append(dict(id=cid, desc=desc, passed=bool(ok), value=float(val), detail=detail))
        print(f"{cid} {'PASS' if ok else 'FAIL'}  {desc}\n     {detail}  ({time.time() - t0:.1f} s)")
    db().execute('INSERT INTO selfcheck_runs(ts,code_hash,passed,results) VALUES(?,?,?,?)',
                 (time.time(), code_hash(), int(ok_all), jdump(res)))
    db().commit()
    print('selfcheck ' + ('PASSED' if ok_all else 'FAILED'))
    return ok_all


def require_selfcheck():
    row = db().execute('SELECT passed FROM selfcheck_runs WHERE code_hash=? ORDER BY id DESC LIMIT 1',
                       (code_hash(),)).fetchone()
    if row and row[0]:
        return
    print('no passing selfcheck for this version of camdist.py; running it first')
    if not run_selfcheck():
        raise SystemExit('selfcheck failed: other subcommands are blocked until it passes')


def cmd_selfcheck(args):
    sys.exit(0 if run_selfcheck() else 1)


# ----------------------------------------------------------------------------
# stream
# ----------------------------------------------------------------------------
VIEW_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>camdist view</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;background:#111;color:#ddd;font:14px/1.4 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:12px 16px}
img{display:block;width:100%;height:auto;background:#000;image-rendering:auto}
#bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
#st{font-variant-numeric:tabular-nums;margin-right:auto}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 10px;cursor:pointer}
button:hover{background:#383838}
</style></head><body><main>
<img id="v" src="/stream" alt="camera stream">
<div id="bar"><span id="st">connecting...</span>
<span class="esp"><button data-k="a">auto/locked</button></span>
<span class="esp">exposure <button data-k="e">-</button><button data-k="E">+</button></span>
<span class="esp">gain <button data-k="g">-</button><button data-k="G">+</button></span>
<button data-k="s">snapshot</button></div></main>
<script>
const st=document.getElementById('st');
document.querySelectorAll('button').forEach(b=>b.onclick=()=>fetch('/key?k='+b.dataset.k));
async function poll(){try{const r=await (await fetch('/status.json')).json();
st.textContent=r.size+'  '+r.fps.toFixed(1)+' fps  age '+Math.round(r.age_ms)+' ms  brightness '+r.brightness.toFixed(1)
+(r.esp?(r.auto?'  AUTO':'  aec_value '+r.aec+'  agc_gain '+r.agc):'')+(r.err?'  ('+r.err+')':'');
document.querySelectorAll('.esp').forEach(e=>e.style.display=r.esp?'':'none');}catch(e){st.textContent='camdist not responding'}}
setInterval(poll,1000);poll();
</script></body></html>"""


def serve_http(args):
    """Relay the newest camera frame to a browser page at http://127.0.0.1:PORT (no OpenCV window)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    live = Live(args, 'camdist stream', window=False)
    esp = live.cam.kind == 'esp32' and not args.no_lock
    cond = threading.Condition()
    shared = dict(jpg=None, seq=0, img=None, err='')

    def pump():
        while True:
            try:
                img = live.frame()
            except SystemExit as e:  # no frame yet: keep waiting for the camera
                shared['err'] = str(e).split(' (')[0]
                continue
            shared['err'] = ''
            jpg = jpg_bytes(img, 85)
            with cond:
                shared.update(jpg=jpg, img=img)
                shared['seq'] += 1
                cond.notify_all()

    threading.Thread(target=pump, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition('?')
            if path == '/':
                self._send(VIEW_PAGE.encode(), 'text/html; charset=utf-8')
            elif path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                last = 0
                try:
                    while True:
                        with cond:
                            cond.wait_for(lambda: shared['seq'] > last, timeout=10)
                            jpg, last = shared['jpg'], shared['seq']
                        if jpg is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif path == '/frame.jpg' and shared['jpg']:
                self._send(shared['jpg'], 'image/jpeg')
            elif path == '/status.json':
                img = shared['img']
                st = dict(size=f'{img.shape[1]}x{img.shape[0]}' if img is not None else '-', fps=live.fps,
                          age_ms=live.age * 1000, brightness=live.brightness, esp=esp,
                          aec=getattr(live, 'aec', None), agc=getattr(live, 'agc', None), auto=live.auto,
                          err=shared['err'])
                self._send(jdump(st).encode(), 'application/json')
            elif path == '/key':
                k = query.partition('k=')[2][:1]
                try:
                    if esp and k and k in 'aeEgG':
                        live.adjust(k)
                    elif k == 's' and shared['img'] is not None:
                        rid = put_result('snapshot', dict(brightness=live.brightness), blob=jpg_bytes(shared['img']))
                        print(f'snapshot stored as results #{rid}')
                    self._send(b'ok', 'text/plain')
                except OSError as e:
                    self._send(str(e).encode(), 'text/plain', 502)
            else:
                self._send(b'not found', 'text/plain', 404)

    srv = ThreadingHTTPServer((args.host, args.http), Handler)
    srv.daemon_threads = True
    print(f'camera view at http://{args.host}:{args.http}   (Ctrl+C to stop)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        live.close()
        if esp:
            live.cam.lock(live.aec, live.agc, args.framesize, live.quality)  # final settings logged


def cmd_esp_capture(args):
    """Record bursts of frames of a still scene from the ESP32-CAM (SerialCam, USB) for training the denoiser.

    The sensor noise is random from frame to frame, so the average of a burst is a nearly clean picture and any single
    frame of it is a noisy copy (Noise2Noise: no clean photograph is needed). Each scene is stored as the raw JPEG
    files the board sent (about 25 KB each, decoded exactly as the live program decodes them) in
    data/denoise/<session>/scene_NNN/, with the camera settings and the measured noise.

    A hand shakes, so the burst keeps recording until it holds `--burst` frames that sit within `--steady` pixels of one
    another (the image shift of every frame is measured by phase correlation, which does not mistake noise for motion);
    the frames that moved are dropped, not the whole scene."""
    live = Live(args, 'esp-capture')
    if live.cam.transport != 'serial':
        raise SystemExit('esp-capture needs the raw JPEGs of the SerialCam stream: --url /dev/ttyUSB0')
    session = os.path.join(DATA, 'denoise', time.strftime('%Y%m%d_%H%M%S'))
    tags, tag, n_scene, sink, armed, t_arm, saved = ['bright', 'normal', 'dim', 'dark'], 1, 0, [], False, 0.0, 0
    auto = args.script_bursts or 0  # testing aid: take this many scenes by itself, then quit
    win = cv2.createHanningWindow((800, 600), cv2.CV_32F)
    seen = dict(gray=[], shift=[], conf=[], bright=[])

    def arm():
        nonlocal armed, t_arm
        sink.clear()
        for v in seen.values():
            v.clear()
        live.cam.raw_sink, armed, t_arm = sink.append, True, time.time()

    def steady_set():
        """Indices of the largest group of frames within --steady px of one frame (and of the same brightness)."""
        sh = np.array(seen['shift'])
        if len(sh) == 0:
            return []
        d = np.linalg.norm(sh[:, None] - sh[None], axis=2)
        i = int(np.argmax((d < args.steady).sum(1)))
        idx = [j for j in range(len(sh)) if d[i, j] < args.steady]
        br = np.array([seen['bright'][j] for j in idx])
        return [j for j, b in zip(idx, br) if abs(b - np.median(br)) < 3.0]

    print(f'session {session}: space = record a scene ({args.burst} steady frames, a shaking hand is fine, '
          f'frames that moved more than {args.steady} px are dropped), l = lighting tag, e/E exposure, g/G gain, q = quit')
    while True:
        img = live.frame()
        if auto and not armed and saved < auto and live.n > 10:
            arm()
        lines = [live.status_line(img), f'LOCKED  aec_value {live.aec}  agc_gain {live.agc}  jpeg quality {live.quality}'
                 f'   lighting tag: {tags[tag]}   scenes saved: {saved}']
        if armed:
            for jp in list(sink[len(seen['gray']):]):  # new frames: grey image, shift against the first one
                g = cv2.cvtColor(cv2.imdecode(np.frombuffer(jp, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32)
                if g.shape != (600, 800):
                    g = cv2.resize(g, (800, 600))
                seen['bright'].append(float(g.mean()))
                g = cv2.GaussianBlur(g, (0, 0), 1.5)  # noise would swamp the correlation of the raw frames
                if seen['gray']:
                    (dx, dy), conf = cv2.phaseCorrelate(seen['gray'][0], g, win)
                    seen['shift'].append((dx, dy) if conf > 0.05 else (0.0, 0.0))  # a flat scene has nothing to track
                else:
                    seen['shift'].append((0.0, 0.0))
                seen['gray'].append(g)
            ok = steady_set()
            lines.append(f'recording: {len(ok)}/{args.burst} steady frames ({len(seen["gray"])} seen), keep the camera as still as you can')
            if len(ok) >= args.burst or len(seen['gray']) >= 4 * args.burst or time.time() - t_arm > 120:
                live.cam.raw_sink, armed = None, False
                if len(ok) < min(args.burst, max(5, args.burst // 2)):
                    print(f'only {len(ok)} of {len(seen["gray"])} frames were steady within {args.steady} px: discarded, '
                          'rest the camera on something and try again')
                else:
                    ok = ok[:args.burst]
                    jpgs = [sink[j] for j in ok]
                    st = np.stack([cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) for j in jpgs])
                    mean_img = st.mean(3).mean(0)
                    sigma = float(st.std(0).mean())  # temporal noise, grey levels (0 to 255)
                    d = os.path.join(session, f'scene_{n_scene:03d}')
                    os.makedirs(d, exist_ok=True)
                    for i, j in enumerate(jpgs):
                        open(os.path.join(d, f'f{i:02d}.jpg'), 'wb').write(j)
                    json.dump(dict(ts=time.time(), n=len(jpgs), n_seen=len(seen['gray']), steady_px=args.steady, aec=live.aec,
                                   agc=live.agc, quality=live.quality, lighting=tags[tag], brightness=float(mean_img.mean()),
                                   noise_sigma=sigma, size=[st.shape[2], st.shape[1]],
                                   kb_per_frame=float(np.mean([len(j) for j in jpgs]) / 1024)),
                              open(os.path.join(d, 'meta.json'), 'w'))
                    print(f'scene {n_scene} saved: {len(jpgs)} steady frames of {len(seen["gray"])}, brightness {mean_img.mean():.0f}, '
                          f'noise {sigma:.1f} grey levels, {d}')
                    n_scene += 1
                    saved += 1
        k = live.show(img.copy(), lines + ['space record a scene  l lighting tag  e/E exposure  g/G gain  q quit'])
        if auto and saved >= auto and not armed:
            k = ord('q')
        if k in (ord('q'), 27):
            break
        if k == ord(' ') and not armed:
            arm()
        if k == ord('l'):
            tag = (tag + 1) % len(tags)
        if k in (ord('e'), ord('E'), ord('g'), ord('G')):
            live.adjust(chr(k))
    live.cam.raw_sink = None
    live.close()
    print(f'{saved} scenes in {session}')


DENOISE_CKPT = os.path.join(DATA, 'checkpoints', 'denoise_gray.pt')
DENOISE_RESUME = os.path.join(DATA, 'checkpoints', 'denoise_gray_resume.pt')  # network, optimiser, step: training continues from here


def denoise_net(ch=32):
    """Small U-Net for grey frames (three scales, residual output: the picture plus a learned correction).
    Feed-forward and one pass per frame, so it is real time; any size, padded to a multiple of 4 inside."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.1, True),
                             nn.Conv2d(o, o, 3, padding=1), nn.LeakyReLU(0.1, True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = blk(1, ch), blk(ch, 2 * ch), blk(2 * ch, 4 * ch)
            self.up2, self.d2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 2, stride=2), blk(4 * ch, 2 * ch)
            self.up1, self.d1 = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2), blk(2 * ch, ch)
            self.out = nn.Conv2d(ch, 1, 1)

        def forward(self, x):
            h, w = x.shape[-2:]
            xp = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode='reflect')
            e1 = self.e1(xp)
            e2 = self.e2(F.avg_pool2d(e1, 2))
            e3 = self.e3(F.avg_pool2d(e2, 2))
            d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return (xp + self.out(d1))[..., :h, :w]

    return Net()


def load_denoiser(path=DENOISE_CKPT, device=None):
    """The trained grey denoiser as a function: grey uint8 image (or BGR) -> grey uint8 image, or None if not trained."""
    if not os.path.exists(path):
        return None
    torch = _torch()
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = denoise_net(ck['ch']).to(dev).eval()
    net.load_state_dict(ck['state'])

    def run(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        with torch.no_grad():
            y = net(torch.from_numpy(g).to(dev).float().div(255)[None, None])
        return (y[0, 0].clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    run.info = ck.get('info', {})
    return run


def load_denoise_scenes(root=None):
    """Every captured scene (esp-capture) as grey frames with an aligned average. Returns a list of dicts:
    name, frames (n x h x w uint8, as the board sent them) and target (n x h x w float16: the average of all the
    frames of the scene, moved back to each frame's own position, so the pair differs only by noise)."""
    win = None
    out = []
    for d in sorted(glob.glob(os.path.join(root or os.path.join(DATA, 'denoise'), '*', 'scene_*'))):
        files = sorted(glob.glob(os.path.join(d, 'f*.jpg')))
        if len(files) < 5:
            continue
        g = [cv2.cvtColor(cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY) for f in files]
        if len({x.shape for x in g}) != 1:
            continue
        h, w = g[0].shape
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        bl = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.5)
        ref, sh, al = bl(g[0]), [], []
        for x in g:  # shift of each frame against the first (sub-pixel), then every frame moved onto the first
            (dx, dy), conf = cv2.phaseCorrelate(ref, bl(x), win)
            dx, dy = (dx, dy) if conf > 0.05 and abs(dx) < 4 and abs(dy) < 4 else (0.0, 0.0)
            sh.append((dx, dy))
            al.append(cv2.warpAffine(x.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
        mean = np.mean(al, 0)
        tgt = np.stack([cv2.warpAffine(mean, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REFLECT) for dx, dy in sh]).astype(np.float16)
        sig = float(np.std(np.stack(al), 0).mean())
        fr = np.stack(g)
        rm = np.mean([np.sqrt(np.mean((fr[i][8:-8, 8:-8].astype(np.float32) - tgt[i][8:-8, 8:-8].astype(np.float32)) ** 2))
                      for i in range(0, len(fr), max(1, len(fr) // 6))])
        # misfit: how much more a frame differs from its scene average than the noise alone explains (1 = nothing else:
        # the target is good; much above 1 = the hand drifted or the picture changed, so the average is blurred)
        out.append(dict(name='/'.join(d.split(os.sep)[-2:]), frames=fr, target=tgt, sigma=sig, misfit=float(rm / max(sig, 1e-6))))
    return out


def psnr(a, b):
    return float(10 * np.log10(255.0 ** 2 / max(float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)), 1e-9)))


def esp_qtable():
    """The ESP32-CAM's own JPEG luminance quantisation table (read from a recorded frame), so synthetic frames get the same blocks."""
    from PIL import Image
    for f in sorted(glob.glob(os.path.join(DATA, 'denoise', '*', 'scene_*', 'f00.jpg'))):
        return list(Image.open(f).quantization[0])
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
    return [int(v) for v in np.clip(std * 1.25, 1, 255)]


def degrade_esp(clean, rng, qtable, sigma=None, blur=None):
    """A clean grey picture (float, 0 to 255, sides multiples of 8) as the ESP32-CAM would deliver it: a soft lens, signal-dependent
    sensor noise (strength `sigma`, grey levels at mid brightness), then JPEG with the board's own table. Returns
    (noisy uint8, soft float: what a perfect denoiser should give back)."""
    from PIL import Image
    import io as _io
    blur = rng.uniform(0.5, 1.4) if blur is None else blur
    sigma = rng.uniform(1.5, 14.0) if sigma is None else sigma  # the JPEG table swallows part of it: 12 here is about 7 grey levels after (real bursts: 2.5 to 7)
    soft = cv2.GaussianBlur(clean, (0, 0), blur)
    nz = rng.normal(0, 1, soft.shape).astype(np.float32)
    corr = rng.uniform(0.0, 1.5)  # the sensor's noise is not white: neighbouring pixels move together
    if corr > 0.2:
        nz = cv2.GaussianBlur(nz, (0, 0), corr)
        nz /= max(float(nz.std()), 1e-6)
    noisy = soft + nz * sigma * np.sqrt(0.35 + 0.65 * np.clip(soft, 0, 255) / 255.0)
    buf = _io.BytesIO()
    Image.fromarray(np.clip(noisy + 0.5, 0, 255).astype(np.uint8), 'L').save(buf, 'JPEG', qtables=[qtable], subsampling=0)
    return np.array(Image.open(_io.BytesIO(buf.getvalue()))), soft


def load_clean_sources(n=500, seed=0):
    """Clean grey pictures at 800 x 600 for the synthetic pairs: stored phone frames of the room (the thing to be measured), the
    phone photos and the recorded ESP burst averages."""
    rng = np.random.default_rng(seed)
    out = []
    rows = db().execute("SELECT blob FROM results WHERE kind='room_raw' ORDER BY id").fetchall()
    for i in rng.permutation(len(rows))[:n]:
        im = cv2.imdecode(np.frombuffer(rows[int(i)][0], np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is not None and im.shape[0] >= 400:
            out.append(cv2.resize(im, (800, 600), interpolation=cv2.INTER_AREA))
    for f in glob.glob(os.path.join(ROOT, '*.jpg')) + glob.glob(os.path.join(DATA, 'phone_photos', '*.jpg')):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is not None and im.shape[0] >= 400:
            out.append(cv2.resize(im, (800, 600), interpolation=cv2.INTER_AREA))
    return out


def synth_pairs(sources, n, seed, size=192, qtable=None):
    """n pairs (noisy uint8, clean uint8), size x size, from random aligned crops of the clean pictures."""
    rng = np.random.default_rng(seed)
    qtable = qtable or esp_qtable()
    X, Y = np.zeros((n, size, size), np.uint8), np.zeros((n, size, size), np.uint8)
    for k in range(n):
        g = sources[rng.integers(len(sources))]
        y0, x0 = 8 * rng.integers(0, (600 - size) // 8 + 1), 8 * rng.integers(0, (800 - size) // 8 + 1)
        c = g[y0:y0 + size, x0:x0 + size].astype(np.float32)
        if rng.random() < 0.5:
            c = c[:, ::-1]
        c = np.clip(c * rng.uniform(0.7, 1.15) + rng.uniform(-25, 25), 0, 255)  # other exposures
        noisy, soft = degrade_esp(np.ascontiguousarray(c), rng, qtable)
        X[k], Y[k] = noisy, np.clip(soft + 0.5, 0, 255).astype(np.uint8)
    return X, Y


def cmd_denoise_train(args):
    """Train the grey denoiser. Two kinds of pairs (noisy frame, clean picture):
    synthetic ones made from clean pictures of the room that are degraded the way the ESP32-CAM does it (soft lens, signal-dependent
    noise, JPEG with the board's own table), unlimited in number and with an exact clean target; and real ones from the recorded
    bursts (the average of a scene's frames at the frame's position). The scenes with the best targets (frames differ from their
    average by little more than the noise itself) are held out to test it, against two baselines, a Gaussian blur and non-local
    means, each at its best strength on those scenes."""
    torch = _torch()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenes = load_denoise_scenes()
    print('recorded scenes (misfit 1 = the average is a clean target, more = the hand drifted):')
    for s_ in scenes:
        print(f'  {s_["name"]}: {len(s_["frames"])} frames, noise {s_["sigma"]:.1f} grey levels, misfit {s_["misfit"]:.2f}')
    val = sorted([s_ for s_ in scenes if s_['misfit'] <= 1.3], key=lambda s_: s_['misfit'])[:args.val]
    real_train = [s_ for s_ in scenes if s_ not in val and s_['misfit'] <= 1.4]
    print(f'held out to test: {[s_["name"] for s_ in val]}\nreal scenes trained on: {[s_["name"] for s_ in real_train]}\ndevice {dev}')
    qt = esp_qtable()
    src = load_clean_sources(args.sources)
    if len(src) < 20:
        raise SystemExit(f'only {len(src)} clean pictures found for the synthetic pairs')
    nv = max(2, len(src) // 10)
    SX, SY = synth_pairs(src[:-nv], args.synth, 1, qtable=qt)
    VX, VY = synth_pairs(src[-nv:], 300, 2, qtable=qt)  # other pictures than the training ones
    print(f'{args.synth} synthetic training pairs from {len(src) - nv} pictures, 300 synthetic test pairs from {nv} others')
    net = denoise_net(args.ch).to(dev)
    print(f'network: {sum(p.numel() for p in net.parameters()) / 1e3:.0f} k parameters')
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    start = 0
    names = [s_['name'] for s_ in real_train] + ['|'] + [s_['name'] for s_ in val] + ['|', args.synth, args.sources, args.real_share]
    if os.path.exists(DENOISE_RESUME) and not args.fresh:
        ck = torch.load(DENOISE_RESUME, map_location=dev, weights_only=False)
        if ck['names'] == names and ck['ch'] == args.ch and ck['steps'] == args.steps and ck['step'] < args.steps:
            net.load_state_dict(ck['net'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            rng.bit_generator.state = ck['rng']
            start = ck['step']
            print(f'resuming from step {start} (saved checkpoint; --fresh starts over)')
        else:
            print('the saved checkpoint is for other scenes or settings (or finished): starting fresh')
    P, M = args.patch, 8  # patches start on the JPEG block grid (multiples of 8)

    def flip(a, b):
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if rng.random() < 0.5:
            a, b = a[::-1], b[::-1]
        if rng.random() < 0.5:
            a, b = a.T, b.T
        return np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)

    def batch():
        xs, ys = [], []
        for _ in range(args.batch):
            if real_train and rng.random() < args.real_share:
                s_ = real_train[rng.integers(len(real_train))]
                i = rng.integers(len(s_['frames']))
                h, w = s_['frames'][i].shape
                y0 = 8 * rng.integers(M // 8, (h - P - M) // 8 + 1)
                x0 = 8 * rng.integers(M // 8, (w - P - M) // 8 + 1)
                a, b = s_['frames'][i][y0:y0 + P, x0:x0 + P], s_['target'][i][y0:y0 + P, x0:x0 + P]
            else:
                i = rng.integers(len(SX))
                n = SX.shape[1]
                y0, x0 = 8 * rng.integers(0, (n - P) // 8 + 1), 8 * rng.integers(0, (n - P) // 8 + 1)
                a, b = SX[i][y0:y0 + P, x0:x0 + P], SY[i][y0:y0 + P, x0:x0 + P]
            a, b = flip(a.astype(np.float32), b.astype(np.float32))
            xs.append(a)
            ys.append(b)
        f = lambda v: torch.from_numpy(np.stack(v))[:, None].to(dev) / 255.0
        return f(xs), f(ys)

    def run_net(a):
        with torch.no_grad():
            y = net(torch.from_numpy(a).to(dev).float().div(255)[None, None])
        return (y[0, 0].clamp(0, 1) * 255).cpu().numpy()

    def evaluate(net_fn):  # real held-out scenes
        r = []
        for s_ in val:
            for i in range(0, len(s_['frames']), max(1, len(s_['frames']) // 8)):
                a, b = s_['frames'][i], s_['target'][i].astype(np.float32)
                r.append(psnr(net_fn(a)[M:-M, M:-M], b[M:-M, M:-M]))
        return float(np.mean(r))

    def evaluate_synth():
        raw = np.mean([psnr(VX[i], VY[i]) for i in range(0, len(VX), 3)])
        got = np.mean([psnr(run_net(VX[i]), VY[i]) for i in range(0, len(VX), 3)])
        return float(raw), float(got)

    def save_resume(step):
        os.makedirs(os.path.dirname(DENOISE_RESUME), exist_ok=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.bit_generator.state,
                        step=step, names=names, ch=args.ch, steps=args.steps), DENOISE_RESUME + '.tmp')
        os.replace(DENOISE_RESUME + '.tmp', DENOISE_RESUME)  # never a half-written file

    t0, step = time.time(), start
    try:
        for step in range(start + 1, args.steps + 1):
            x, y = batch()
            loss = torch.nn.functional.l1_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % args.save_every == 0:
                save_resume(step)
            if step % max(1, args.steps // 10) == 0 or step == start + 1:
                msg = f'step {step}/{args.steps}  l1 {loss.item() * 255:.2f} grey levels  {time.time() - t0:.0f} s'
                if step % max(1, args.steps // 5) == 0:
                    net.eval()
                    sr, sn = evaluate_synth()
                    msg += f'  synthetic test {sr:.2f} -> {sn:.2f} dB'
                    if val:
                        msg += f'  real held-out {evaluate(run_net):.2f} dB'
                    net.train()
                print(msg, flush=True)
    except KeyboardInterrupt:
        step = max(start, step - 1)
        save_resume(step)
        raise SystemExit(f'\nstopped at step {step}: saved, run the same command again to continue')
    net.eval()
    sr, sn = evaluate_synth()
    info = dict(steps=args.steps, real_train=[s_['name'] for s_ in real_train], val=[s_['name'] for s_ in val], ts=time.time(),
                synth=args.synth, psnr_synth_raw=sr, psnr_synth_net=sn)
    print(f'\nsynthetic test pairs (other pictures, known clean target): raw {sr:.2f} dB -> network {sn:.2f} dB')
    if val:
        raw = evaluate(lambda a: a.astype(np.float32))
        best = lambda fn, ks: max(((evaluate(lambda a, k=k: fn(a, k).astype(np.float32)), k) for k in ks))
        blur, kb = best(lambda a, k: cv2.GaussianBlur(a, (0, 0), k), (0.6, 0.8, 1.0, 1.3, 1.7, 2.2))
        nlm, kn = best(lambda a, k: cv2.fastNlMeansDenoising(a, None, k, 7, 21), (3, 5, 8, 12, 16))
        ours = evaluate(run_net)
        t1 = time.time()
        for _ in range(10):
            run_net(val[0]['frames'][0])
        ms = (time.time() - t1) / 10 * 1e3
        info.update(psnr_raw=raw, psnr_blur=blur, psnr_nlm=nlm, psnr_net=ours, ms_per_frame=ms)
        print(f'real held-out scenes, PSNR against the burst average (higher is better):\n'
              f'  raw frame                    {raw:6.2f} dB\n  Gaussian blur (sigma {kb})     {blur:6.2f} dB\n'
              f'  non-local means (h {kn})       {nlm:6.2f} dB\n  this network                 {ours:6.2f} dB   '
              f'({ms:.0f} ms per 800x600 frame on {dev})')
        rows = []
        for s_ in val:
            i = len(s_['frames']) // 2
            a, b = s_['frames'][i], s_['target'][i].astype(np.float32)
            h, w = a.shape
            cy, cx = h // 2 - 90, w // 2 - 120
            tiles = [a, cv2.fastNlMeansDenoising(a, None, kn, 7, 21), run_net(a).astype(np.uint8), np.clip(b + 0.5, 0, 255).astype(np.uint8)]
            crops = [cv2.resize(t[cy:cy + 180, cx:cx + 240], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for t in tiles]
            for c, lab in zip(crops, ('noisy frame', 'non-local means', 'network', 'burst average')):
                cv2.putText(c, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
                cv2.putText(c, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
            rows.append(np.vstack([np.hstack(crops[:2]), np.hstack(crops[2:])]))
        out = os.path.join(DATA, 'denoise', 'held_out_comparison.jpg')
        cv2.imwrite(out, np.hstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f'comparison picture: {out}')
    os.makedirs(os.path.dirname(DENOISE_CKPT), exist_ok=True)
    torch.save(dict(state=net.state_dict(), ch=args.ch, info=info), DENOISE_CKPT)
    if os.path.exists(DENOISE_RESUME):
        os.remove(DENOISE_RESUME)  # finished: the next run starts fresh
    print(f'saved {DENOISE_CKPT}')


def denoise_bgr(dn, img):
    """A colour frame with its luminance cleaned by the trained denoiser (colour kept as it is)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = dn(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def corner_jitter(corner_sets, tol=3.0):
    """Frame-to-frame jitter of the detected checkerboard corners [px] in frames of a still scene. The corners of the frame with most
    corners are the reference; every other frame's corners are matched to them (nearest, within `tol`), the shift of the whole frame
    (a hand) is removed with the median, and what is left per corner is the standard deviation over the frames, rms over the
    corners seen in at least 60 % of the frames. Returns (jitter, number of corners used) or (nan, 0)."""
    from scipy.spatial import cKDTree
    ref_i = int(np.argmax([len(c) for c in corner_sets]))
    ref = corner_sets[ref_i]
    if len(ref) < 20 or len(corner_sets) < 4:
        return float('nan'), 0
    tree = cKDTree(ref)
    D = np.full((len(corner_sets), len(ref), 2), np.nan)
    for i, c in enumerate(corner_sets):
        if i == ref_i or len(c) == 0:
            continue
        dist, j = tree.query(c, distance_upper_bound=tol)
        ok = np.isfinite(dist)
        # one match per reference corner: the closest
        order = np.argsort(dist[ok])
        taken = set()
        for k in order:
            jj = int(j[ok][k])
            if jj not in taken:
                taken.add(jj)
                D[i, jj] = c[ok][k] - ref[jj]
    idx = [i for i in range(len(corner_sets)) if i != ref_i]
    D = D[idx]
    for i in range(len(D)):  # remove each frame's overall shift
        m = np.nanmedian(D[i].reshape(-1, 2), 0) if np.isfinite(D[i]).any() else 0
        D[i] = D[i] - m
    keep = np.isfinite(D[..., 0]).mean(0) >= 0.6
    if keep.sum() < 20:
        return float('nan'), 0
    v = np.nanvar(D[:, keep, :], axis=0)  # corners x 2
    return float(np.sqrt(v.mean())), int(keep.sum())


def cmd_denoise_eval(args):
    """Does the denoiser help the measurement? On every recorded scene that shows checkerboards, find the board corners in each frame as
    it came from the ESP32-CAM, after a plain Gaussian blur and after the trained denoiser, and compare: how many corners are found,
    how much each corner jitters from frame to frame (that is the pixel noise sigma of the measurement: the sd of every size and of f
    in the Fisher / retention analysis is sigma / sqrt(I_marg), so it scales with it), and how well each board fits a plane
    (reprojection rms of its pose with the ESP32's stored calibration)."""
    dn = load_denoiser()
    if dn is None:
        raise SystemExit('no trained denoiser: run denoise-train first')
    cal = fallback_calib('esp32')
    variants = [('raw', lambda im: im), ('blur 1.0', lambda im: cv2.GaussianBlur(im, (0, 0), 1.0)), ('denoiser', lambda im: denoise_bgr(dn, im))]
    tot = {k: dict(j=[], c=[], r=[], p=[]) for k, _ in variants}
    print(f'{"scene":28s} {"variant":9s} {"planes":>6s} {"corners":>8s} {"jitter px":>10s} {"plane rms px":>13s}')
    for d in sorted(glob.glob(os.path.join(DATA, 'denoise', '*', 'scene_*'))):
        files = sorted(glob.glob(os.path.join(d, 'f*.jpg')))
        if args.last and d not in sorted(glob.glob(os.path.join(DATA, 'denoise', '*', 'scene_*')))[-args.last:]:
            continue
        frames = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in files]
        if len(frames) < 4:
            continue
        rows = {}
        for name, fn in variants:
            sets, npl, rms = [], [], []
            for im in frames:
                pl = find_planes(cv2.cvtColor(fn(im), cv2.COLOR_BGR2GRAY))
                sets.append(np.concatenate([p.reshape(-1, 2) for p in pl]) if pl else np.zeros((0, 2)))
                npl.append(len(pl))
                if cal is not None and pl:
                    K, dist = scaled_calib(cal, (im.shape[1], im.shape[0]))
                    rms += [p['rms'] for p in plane_poses(pl, K, dist) if p.get('rms') is not None]
            j, nj = corner_jitter(sets)
            rows[name] = (np.mean(npl), np.mean([len(c) for c in sets]), j, float(np.median(rms)) if rms else float('nan'), nj)
        if not np.isfinite(rows['raw'][2]):
            print(f'{"/".join(d.split(os.sep)[-2:]):28s} no boards found in enough frames: skipped')
            continue
        for name, (p_, c_, j_, r_, nj) in rows.items():
            print(f'{"/".join(d.split(os.sep)[-2:]) if name == "raw" else "":28s} {name:9s} {p_:6.1f} {c_:8.0f} {j_:10.3f} {r_:13.3f}')
            tot[name]['j'].append(j_)
            tot[name]['c'].append(c_)
            tot[name]['r'].append(r_)
    if not tot['raw']['j']:
        raise SystemExit('no recorded scene shows checkerboards: record some with esp-capture')
    print('\nmean over the scenes:')
    base = {k: np.nanmean(v) for k, v in tot['raw'].items() if k != 'p'}
    for name, _ in variants:
        j, c, r = (np.nanmean(tot[name][k]) for k in 'jcr')
        print(f'  {name:9s} corners {c:6.0f} ({100 * (c / base["c"] - 1):+.0f} %)   jitter {j:.3f} px ({100 * (j / base["j"] - 1):+.0f} %)   '
              f'plane-fit rms {r:.3f} px ({100 * (r / base["r"] - 1):+.0f} %)')
    ratio = np.nanmean(tot['denoiser']['j']) / np.nanmean(tot['raw']['j'])
    print(f'\nFisher bridge: the estimate\'s sd is sigma / sqrt(I_marg), so with the corner noise at {ratio:.2f} of its raw value the predicted sd of f,'
          f' of L, W and H (and of every other unknown) is {ratio:.2f} of the raw one, at the same geometry and number of views.')


def cmd_log_tags(args):
    """The labels of the recent room logs: camera, setup, object, usable for training, and what the corrector would use."""
    tags = log_tags()
    rows = db().execute("SELECT id, ts, calib_id, truth FROM measurements WHERE kind='room' ORDER BY id DESC LIMIT ?", (args.last,)).fetchall()
    print(f'{"id":>5} {"time":>11} {"camera":<10} {"setup":<12} {"object":<12} {"true L x W x H (mm)":<20} use  note')
    for mid, ts, cid, truth in reversed(rows):
        cam, st, obj, use, note = log_label(mid, cid, tags)
        t = json.loads(truth) if truth else None
        size = ' x '.join(f'{t[k] * 1e3:g}' for k in 'LWH') if t else 'none'
        print(f'{mid:>5} {time.strftime("%d %b %H:%M", time.localtime(ts)):>11} {cam:<10} {st:<12} {(obj or (t or {}).get("object") or "?"):<12} {size:<20} '
              f'{"yes" if use and t else "no":<4} {note}')
    labels = [log_label(m, c, tags) for m, _, c, t in rows if t]
    use = collections.Counter((cam, st) for cam, st, _, ok, _ in labels if ok)
    print('\nusable logs by camera and setup: ' + ', '.join(f'{c} / {s}: {n}' for (c, s), n in sorted(use.items())))


def cmd_stream(args):
    if args.scan:
        hosts = scan_for_esp32()
        print('found: ' + (', '.join(hosts) if hosts else 'nothing'))
        if not hosts:
            return
        args.url = args.url or f'http://{hosts[0]}'
    if args.http:
        return serve_http(args)
    live = Live(args, 'camdist stream')
    esp = live.cam.kind == 'esp32' and not args.no_lock
    if esp and args.auto:
        live.adjust('a')
    help_ = ('a auto/locked   e/E exposure -/+   g/G gain -/+   s snapshot   q quit' if esp
             else 's snapshot   q quit')
    while True:
        img = live.frame()
        lines = [live.status_line(img)]
        if esp:
            lines.append('AUTO exposure + white balance (viewing only)' if live.auto else
                         f'LOCKED  aec_value {live.aec}  agc_gain {live.agc}')
        k = live.show(img.copy(), lines + [help_])
        if k in (ord('q'), 27):
            break
        if esp and k in (ord('a'), ord('e'), ord('E'), ord('g'), ord('G')):
            live.adjust(chr(k))
        if k == ord('s'):
            rid = put_result('snapshot', dict(brightness=live.brightness, shape=img.shape), blob=jpg_bytes(img))
            print(f'snapshot stored as results #{rid}')
    live.close()
    if esp:
        live.cam.lock(live.aec, live.agc, args.framesize, live.quality)  # final settings logged


# ----------------------------------------------------------------------------
# calib-board (handoff 3.1)
# ----------------------------------------------------------------------------
def detect_board(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit).reshape(-1, 2)


def calibrate_board(corner_sets, objp, size, session):
    imgs = [c.astype(np.float32).reshape(-1, 1, 2) for c in corner_sets]
    flags = cv2.CALIB_FIX_K3
    # a flat board must fit a homography to well under a pixel (distortion of this lens is small);
    # views that miss by more were smeared by camera motion during the rolling-shutter readout
    hres = []
    for im in imgs:
        H, _ = cv2.findHomography(objp[:, :2], im.reshape(-1, 2))
        p = cv2.perspectiveTransform(objp[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
        hres.append(float(np.sqrt(((p - im.reshape(-1, 2)) ** 2).sum(1).mean())))
    keep = [i for i, r in enumerate(hres) if r < 1.0]
    if len(keep) >= 6 and len(keep) < len(imgs):
        print(f'dropped {len(imgs) - len(keep)} smeared views (homography rms '
              f'{sorted(round(r, 1) for r in hres if r >= 1.0)} px)')
        imgs = [imgs[i] for i in keep]
    dropped = []
    while True:  # drop smeared views (camera moving during the rolling-shutter readout), worst first
        objs = [objp] * len(imgs)
        out = cv2.calibrateCameraExtended(objs, imgs, size, None, None, flags=flags)
        rms, K, dist, rvecs, tvecs, std_in, _, per_view = out[:8]
        pv = np.ravel(per_view)
        worst = int(np.argmax(pv))
        if pv[worst] < max(1.0, 3 * np.median(pv)) or len(imgs) <= 8:
            break
        dropped.append(round(float(pv[worst]), 2))
        imgs.pop(worst)
    if dropped:
        print(f'dropped {len(dropped)} smeared views (per-view rms {dropped} px)')
    resid = []
    for o, im, rv, tv in zip(objs, imgs, rvecs, tvecs):
        p = cv2.projectPoints(o, rv, tv, K, dist)[0].reshape(-1, 2)
        im = im.reshape(-1, 2)
        resid.append(np.hstack([im, im - p]))
    resid = np.vstack(resid)
    names = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']
    std = dict(zip(names, np.ravel(std_in)[:9]))
    meta = dict(std=std, per_view_rms=np.ravel(per_view), n_views=len(imgs), flags='FIX_K3',
                dropped_view_rms=dropped)
    cid = save_calib('board', session, size, K, np.ravel(dist), rms, meta)
    put_result('board_residuals', dict(points=resid, rms=rms, size=size), ref=cid)
    tilt = [math.degrees(math.acos(min(1.0, abs(cv2.Rodrigues(rv)[0][2, 2])))) for rv in rvecs]
    put_result('board_tilts', dict(tilt_deg=tilt), ref=cid)
    d = np.ravel(dist)
    n_tilt = sum(t > 20 for t in tilt)
    print(f'board tilt relative to the camera: median {np.median(tilt):.0f} deg, {n_tilt} views above 20 deg'
          + ('' if n_tilt >= 5 else '  -> f is poorly determined: add views with the camera TILTED 30-45 deg'))
    print(f'board calibration #{cid}: {len(imgs)} views, reprojection rms {rms:.3f} px '
          f"({'meets' if rms < 0.5 else 'ABOVE'} the 0.5 px target)")
    print(f"  fx {K[0, 0]:.2f} +- {std['fx']:.2f}  fy {K[1, 1]:.2f} +- {std['fy']:.2f}  "
          f"cx {K[0, 2]:.2f} +- {std['cx']:.2f}  cy {K[1, 2]:.2f} +- {std['cy']:.2f}")
    print(f'  k1 {d[0]:+.4f}  k2 {d[1]:+.4f}  p1 {d[2]:+.5f}  p2 {d[3]:+.5f}')
    return cid


def list_screens():
    """Connected monitors from xrandr: name, pixel size, offset and physical size in mm."""
    import subprocess
    try:
        out = subprocess.run(['xrandr', '--query'], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    pat = r'^(\S+) connected (?:primary )?(\d+)x(\d+)\+(\d+)\+(\d+).*?(\d+)mm x (\d+)mm'
    return [dict(name=m[0], w=int(m[1]), h=int(m[2]), x=int(m[3]), y=int(m[4]), mm_w=int(m[5]), mm_h=int(m[6]))
            for m in re.findall(pat, out, re.M)]


def screen_board(scr, cols, rows, status=''):
    """Full-screen checkerboard image with (cols x rows) inner corners and a one-square white margin.

    Returns (image, square size in metres). The square size only sets the scale of the board
    poses; the intrinsics (f, c, distortion) from a planar target do not depend on it.
    """
    W, H = scr['w'], scr['h']
    s = int(min(W / (cols + 3), H / (rows + 3)))
    img = np.full((H, W), 255, np.uint8)
    x0, y0 = (W - s * (cols + 1)) // 2, (H - s * (rows + 1)) // 2
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                img[y0 + j * s:y0 + (j + 1) * s, x0 + i * s:x0 + (i + 1) * s] = 0
    if status:
        cv2.putText(img, status, (max(10, x0), H - 14), FONT, max(0.6, H / 1400), 140, 1, cv2.LINE_AA)
    mm_px = scr['mm_w'] / W if scr['mm_w'] else 0.25
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), s * mm_px / 1000


def cmd_calib_board(args):
    pattern = (args.cols, args.rows)
    scr = None
    if args.screen:
        screens = list_screens()
        if not screens:
            raise SystemExit('xrandr found no screens: print the board instead, or use --square on a paper one')
        if args.screen == 'auto':  # the physically largest screen; the camera preview stays on another
            scr = max(screens, key=lambda d: d['mm_w'] * d['mm_h'])
        else:
            scr = next((d for d in screens if d['name'] == args.screen), None)
            if scr is None:
                raise SystemExit(f"no screen {args.screen}; connected: {[d['name'] for d in screens]}")
        _, args.square = screen_board(scr, args.cols, args.rows)
        args.auto = True
        print(f"checkerboard on {scr['name']} ({scr['w']}x{scr['h']} px, {scr['mm_w']}x{scr['mm_h']} mm): "
              f"squares {args.square * 1000:.1f} mm")
    objp = board_points(args.cols, args.rows, args.square)
    if args.from_db:
        q = 'SELECT corners,width,height,session FROM board_views'
        rows = db().execute(q + ('' if args.from_db == 'all' else ' WHERE session=?') + ' ORDER BY id',
                            () if args.from_db == 'all' else (args.from_db,)).fetchall()
        if rows:  # one camera at a time: keep the views with the resolution of the newest one
            rows = [r for r in rows if (r[1], r[2]) == (rows[-1][1], rows[-1][2])]
            print(f'{len(rows)} stored views at {rows[-1][1]}x{rows[-1][2]}')
        if len(rows) < 6:
            raise SystemExit(f'only {len(rows)} stored views, need at least 6')
        calibrate_board([np.array(json.loads(r[0])) for r in rows], objp, (rows[0][1], rows[0][2]),
                        rows[-1][3])
        return
    session = new_session('screen' if scr else 'board')
    live = Live(args, 'calib-board')
    bw, shown = 'checkerboard', None
    if scr is not None:
        if live.cam.kind == 'esp32' and not args.no_lock:
            live.adjust('a')  # auto exposure is fine for intrinsics and copes with a bright screen
        cv2.namedWindow(bw, cv2.WINDOW_NORMAL)
        cv2.moveWindow(bw, scr['x'], scr['y'])
        cv2.resizeWindow(bw, scr['w'], scr['h'])
        cv2.setWindowProperty(bw, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if args.views is None:
        args.views = 12 if scr is not None else 20
    views, last_t, last_c, auto, size = [], 0.0, None, args.auto, None
    prev_c, steady_t = None, None
    gx, gy = 4, 3
    cover = np.zeros((gy, gx), int)

    def tally(c, sign):
        for u, v in c:
            cover[min(int(v * gy / size[1]), gy - 1), min(int(u * gx / size[0]), gx - 1)] += sign

    try:
        while len(views) < args.views:
            img = live.frame()
            size = (img.shape[1], img.shape[0])
            c = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
            disp = img.copy()
            for j in range(gy):
                for i in range(gx):
                    if cover[j, i] == 0:
                        x0, y0 = i * size[0] // gx, j * size[1] // gy
                        sub = disp[y0:y0 + size[1] // gy, x0:x0 + size[0] // gx]
                        sub[:] = (0.75 * sub + 0.25 * np.array([0, 0, 255])).astype(np.uint8)
            if c is not None:
                cv2.drawChessboardCorners(disp, pattern, c.reshape(-1, 1, 2).astype(np.float32), True)
            # auto capture: a new pose, held still for `steady` s, at least `interval` s after the last one
            moved = last_c is None or (c is not None and np.abs(c - last_c).mean() > 25)
            now = time.time()
            still = c is not None and (args.steady <= 0 or (prev_c is not None and np.abs(c - prev_c).mean() < 3))
            steady_t = (steady_t or now) if still else None
            prev_c = c
            wait = args.interval - (now - last_t)
            if wait > 0 and views:
                hint = f'next capture in {math.ceil(wait)} s - move to a new pose'
            elif c is None:
                hint = 'board not seen'
            elif not moved:
                hint = 'move to a new pose'
            else:
                hint = 'hold still...'
            if scr is not None and shown != (len(views), hint):
                msg = (f'{len(views)}/{args.views} views   {hint}   (TILT the camera 30-45 deg, closer, '
                       'farther, board in every corner of the camera image; q quits)')
                cv2.imshow(bw, screen_board(scr, args.cols, args.rows, msg)[0])
                shown = (len(views), hint)
            k = live.show(disp, [live.status_line(img),
                                 f'views {len(views)}/{args.views}  board {"FOUND" if c is not None else "-"}  '
                                 f'auto {"on" if auto else "off"}   {hint if auto else ""}',
                                 'space capture  a auto  c calibrate  u undo  q quit (calibrates with >= 6 views)',
                                 'tilt the camera; red cells: no corners there yet'])
            want = k == ord(' ') or (auto and moved and wait <= 0 and steady_t is not None
                                     and now - steady_t >= args.steady)
            if want and c is not None:
                db().execute('INSERT INTO board_views(session,ts,width,height,corners,brightness,jpeg) '
                             'VALUES(?,?,?,?,?,?,?)', (session, time.time(), size[0], size[1], jdump(c),
                                                        live.brightness, jpg_bytes(img)))
                db().commit()
                views.append(c)
                last_t, last_c = time.time(), c
                tally(c, 1)
                print(f'view {len(views)} captured\a')
            if k == ord('a'):
                auto = not auto
            if k == ord('u') and views:
                db().execute('DELETE FROM board_views WHERE id=(SELECT MAX(id) FROM board_views WHERE session=?)',
                             (session,))
                db().commit()
                tally(views.pop(), -1)
                last_c = views[-1] if views else None
            if k in (ord('c'), ord('q'), 27):
                break
    except (SystemExit, KeyboardInterrupt) as e:  # whatever stops the run, calibrate what was captured
        print(f'stopped: {e}' if str(e) else 'stopped')
    finally:
        live.close()
        if scr is not None:
            cv2.destroyWindow(bw)
            cv2.waitKey(1)
    if len(views) < 6:
        print(f'{len(views)} views stored in session {session}; need at least 6 to calibrate '
              '(views from several runs combine: calib-board --from-db all)')
        return
    calibrate_board(views, objp, size, session)


# ----------------------------------------------------------------------------
# calib-ruler (handoff 3.2)
# ----------------------------------------------------------------------------
def report_ruler_fits(ob, fit_k1):
    sets = [('f',), ('f', 'delta')] + ([('f', 'k1'), ('f', 'delta', 'k1')] if fit_k1 else [])
    fits = {}
    print(f"{'free parameters':<18}{'f [px]':>18}{'delta [mm]':>18}{'k1':>18}{'sigma [px]':>12}")
    for free in sets:
        ft = fit_ruler(ob, free)
        p, se = ft['params'], ft['se']
        fmt = lambda k, s=1.0, d=2: (f"{p[k] * s:.{d}f} +- {se[k] * s:.{d}f}" if k in se else 'fixed 0')
        print(f"{'+'.join(free):<18}{fmt('f'):>18}{fmt('delta', 1e3, 1):>18}{fmt('k1', 1, 4):>18}"
              f"{ft['sigma']:>12.3f}")
        fits['+'.join(free)] = dict(params=p, se=se, sigma=ft['sigma'], resid=ft['resid'])
    return fits


def cmd_calib_ruler(args):
    if args.from_db:
        ob, session = ruler_obs_from_db(None if args.from_db == 'latest' else args.from_db)
        if ob is None:
            raise SystemExit('no ruler observations stored')
    else:
        session = new_session('ruler')
        board = load_calib('board', required=False) if args.use_board else None
        if args.use_board and board is None:
            raise SystemExit('--use-board given but there is no board calibration')
        plan = [float(x) for x in args.distances.split(',')] if args.distances else []
        ranges = tape_ranges(args)
        live = Live(args, 'calib-ruler', calib=board)
        n = 0
        while True:
            img = live.frame()
            ranges = handle_click(live, img) or ranges
            blobs = find_tape_blobs(img, ranges, args.min_area)
            disp = img.copy()
            draw_blobs(disp, blobs)
            nxt = f'place ruler at {plan[n]:.3f} m' if n < len(plan) else 'any distance'
            l_now = f'{math.dist(blobs[0][:2], blobs[1][:2]):.1f} px' if len(blobs) == 2 else f'{len(blobs)} blobs'
            k = live.show(disp, [live.status_line(img), f'stored {n}   l = {l_now}   next: {nxt}',
                                 'hold the ruler square to the camera; vary distance AND image position',
                                 'space capture  u undo  c fit  q quit   click tape to pick its colour'])
            if k == ord(' '):
                cap = capture_tape(live, ranges, args.min_area)
                if cap is None:
                    print('need exactly two tape blobs in most frames; nothing stored')
                    continue
                z = plan[n] if n < len(plan) else live.ask(cap['frame'], 'measured distance Z [m]')
                if z is None or z == 'esc':
                    continue
                h, w = img.shape[:2]
                cx, cy = (board.cx, board.cy) if board else ((w - 1) / 2, (h - 1) / 2)
                db().execute('INSERT INTO ruler_obs(session,ts,L,z_meas,l_px,l_std,n_frames,u1,v1,u2,v2,cx,cy,'
                             'width,height,undistorted,brightness) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                             (session, time.time(), args.length, z, cap['l'], cap['l_std'], cap['n'],
                              *cap['P1'], *cap['P2'], cx, cy, w, h, int(board is not None), cap['brightness']))
                db().commit()
                n += 1
                print(f"#{n}: Z_meas {z:.3f} m  l {cap['l']:.2f} +- {cap['l_std']:.2f} px")
            if k == ord('u') and n:
                db().execute('DELETE FROM ruler_obs WHERE id=(SELECT MAX(id) FROM ruler_obs WHERE session=?)',
                             (session,))
                db().commit()
                n -= 1
            if k == ord('c') or k in (ord('q'), 27) or (plan and n >= len(plan)):
                break
        live.close()
        ob, _ = ruler_obs_from_db(session)
        if ob is None:
            return
    n = len(ob['z'])
    need = 3 + int(args.fit_k1)
    if n < need:
        print(f'{n} observations in session {session}; need at least {need} to fit')
        return
    print(f'ruler calibration, session {session}: {n} placements, L = {ob["L"][0]:.4f} m')
    fits = report_ruler_fits(ob, args.fit_k1)
    main = fits['f+delta+k1' if args.fit_k1 else 'f+delta']
    p, se = main['params'], main['se']
    J = ruler_jacobian(ob, p)
    tab = fisher_table(J, main['sigma'] if main['sigma'] > 0 else 1.0)
    print(f"retention of f: with delta {tab[1]['rho']:.3f}, with k1 {tab[2]['rho']:.3f}, "
          f"with both {tab[3]['rho']:.3f}  (see `fisher` for the full report)")
    size = ob['size']
    K = np.array([[p['f'], 0, ob['c'][0]], [0, p['f'], ob['c'][1]], [0, 0, 1]])
    meta = dict(delta=p['delta'], se=se, session=session, L=float(ob['L'][0]), n=n, fit_k1=args.fit_k1)
    cid = save_calib('ruler', session, size, K, [p['k1'], 0, 0, 0, 0], main['sigma'], meta)
    put_result('ruler_fit', dict(z=ob['z'], P1=ob['P1'], P2=ob['P2'], L=ob['L'], c=ob['c'],
                                 l_std=ob['l_std'], fits=fits, main='f+delta+k1' if args.fit_k1 else 'f+delta',
                                 calib_id=cid), ref=session)
    print(f"stored ruler calibration #{cid}: f = {p['f']:.2f} +- {se['f']:.2f} px, "
          f"delta = {p['delta'] * 1e3:.1f} +- {se['delta'] * 1e3:.1f} mm")


# ----------------------------------------------------------------------------
# calib-nominal: a datasheet estimate for when no calibration target is at hand
# ----------------------------------------------------------------------------
def cmd_calib_nominal(args):
    """Pinhole estimate from the lens field of view: f = (half image extent) / tan(FOV / 2).

    The OV2640 output modes up to SVGA are scaled from the full sensor, so the FOV
    is the same at every resolution. No distortion, principal point at the image centre.
    """
    W, H = args.size
    if args.hfov:
        f = (W / 2) / math.tan(math.radians(args.hfov) / 2)
        how = f'horizontal FOV {args.hfov:g} deg'
    else:
        f = (math.hypot(W, H) / 2) / math.tan(math.radians(args.dfov) / 2)
        how = f'diagonal FOV {args.dfov:g} deg'
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    cid = save_calib('nominal', new_session('nominal'), (W, H), K, [0, 0, 0, 0, 0], float('nan'),
                     dict(source=how, note='datasheet estimate, not measured'))
    hf = 2 * math.degrees(math.atan(W / 2 / f))
    print(f'nominal calibration #{cid} ({how}) at {W}x{H}: f = {f:.0f} px '
          f'(horizontal FOV {hf:.1f} deg), c = ({K[0, 2]:.1f}, {K[1, 2]:.1f}), no distortion')
    print('  the same formula for the other common ESP32-CAM lenses (diagonal FOV):')
    for d in (66, 78, 120, 160):
        fd = (math.hypot(W, H) / 2) / math.tan(math.radians(d) / 2)
        print(f'    {d:>3} deg -> f = {fd:5.0f} px' + ('   (fisheye: pinhole model is poor, calibrate)' if d >= 120 else ''))
    print('  distances scale with f, so a lens FOV that is off by a few degrees gives an error of ~10 %.\n'
          '  Quick check: `measure` with the ruler at a taped distance; if it reads Z_read for true Z_true,\n'
          '  the right f is f * Z_true / Z_read. `calib-ruler` replaces this estimate with a measured one.')


def android_intrinsics(dump):
    """Back-camera factory intrinsics from `adb shell dumpsys media.camera` text.

    Returns dict(fx, fy, cx, cy, skew, array=(w, h), distortion) in active-array pixels.
    """
    def field(block, name):
        m = re.search(re.escape(name) + r' \([0-9a-f]+\): \w+\[\d+\]\s*\n\s*\[([^\]]*)\]', block)
        return m.group(1).split() if m else None

    for block in re.split(r'== Camera device ', dump)[1:]:
        if (field(block, 'android.lens.facing') or [''])[0] != 'BACK':
            continue
        cal = field(block, 'android.lens.intrinsicCalibration')
        arr = field(block, 'android.sensor.info.preCorrectionActiveArraySize') or \
            field(block, 'android.sensor.info.activeArraySize')
        if cal and arr:
            fx, fy, cx, cy, sk = map(float, cal)
            return dict(fx=fx, fy=fy, cx=cx, cy=cy, skew=sk, array=(int(arr[2]), int(arr[3])),
                        distortion=[float(x) for x in field(block, 'android.lens.distortion') or []])
    return None


def cmd_calib_phone(args):
    """Android factory intrinsics (LENS_INTRINSIC_CALIBRATION) via adb, scaled to the stream size.

    The stream is assumed to show the whole sensor width (centre crop for other aspect ratios)
    and to be distortion-corrected by the phone's ISP (the default), so no distortion is applied.
    """
    import subprocess
    try:
        dump = subprocess.run(['adb', 'shell', 'dumpsys', 'media.camera'], capture_output=True, text=True,
                              timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f'adb failed ({e}): plug the phone in with USB debugging on')
    intr = android_intrinsics(dump)
    if intr is None:
        raise SystemExit('the phone reports no back-camera intrinsics (LENS_INTRINSIC_CALIBRATION)')
    W, H = args.size
    aw, ah = intr['array']
    s = max(W / aw, H / ah)  # fill the output, cropping the other axis symmetrically
    fx, fy = intr['fx'] * s, intr['fy'] * s
    cx, cy = intr['cx'] * s - (aw * s - W) / 2, intr['cy'] * s - (ah * s - H) / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cid = save_calib('phone', new_session('phone'), (W, H), K, [0, 0, 0, 0, 0], float('nan'),
                     dict(source='android LENS_INTRINSIC_CALIBRATION via adb', factory=intr, scale=s))
    print(f"phone factory intrinsics at {aw}x{ah}: fx {intr['fx']:.1f} fy {intr['fy']:.1f} "
          f"cx {intr['cx']:.1f} cy {intr['cy']:.1f}")
    print(f'stored calibration #{cid} for a {W}x{H} stream: fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f} '
          f'(horizontal FOV {2 * math.degrees(math.atan(W / 2 / fx)):.1f} deg)')


# ----------------------------------------------------------------------------
# fisher (handoff 3.3)
# ----------------------------------------------------------------------------
def cmd_fisher(args):
    rng = np.random.default_rng(args.seed)
    real = not args.plan
    if real:
        ob, session = ruler_obs_from_db(args.session)
        if ob is None:
            raise SystemExit('no ruler observations: run calib-ruler, or give a planned design with --plan')
        free = ('f', 'delta', 'k1') if len(ob['z']) >= 6 else ('f', 'delta')
        fit = fit_ruler(ob, free)
        prm = fit['params']
        sigma = args.sigma or fit['sigma']
        desc = f'real data, session {session}, {len(ob["z"])} placements'
    else:
        a, b, n = args.plan.split(':')
        z = np.linspace(float(a), float(b), int(n))
        if args.spacing == 'w':
            z = 1 / np.linspace(1 / float(a), 1 / float(b), int(n))
        f0 = args.f or (load_calib(required=False).fx if load_calib(required=False) else 600.0)
        W, H = 800, 600
        pos = [float(p) for p in args.positions.split(',')]
        zz = np.repeat(z, len(pos))
        xy = np.array([[p * (W / 2) / f0, 0.0] for p in pos] * len(z))
        prm = dict(f=f0, delta=args.delta, k1=0.0)
        ob = ruler_synth(f0, args.delta, 0.0, ((W - 1) / 2, (H - 1) / 2), args.length, zz, xy)
        sigma = args.sigma or 1.0
        session = None
        desc = f'planned design, Z {a}..{b} m x {n} ({args.spacing} spacing), positions {args.positions}'
    J = ruler_jacobian(ob, prm)
    tab = fisher_table(J, sigma)
    if real and args.bootstrap > 0:
        n = len(ob['z'])
        # parameters outside a nuisance set are 'known': held at their fitted values, as the
        # Fisher prediction assumes (fixing them at 0 instead adds bias, see calib-ruler)
        known = dict(prm)
        boot = {i: [] for i in range(len(NUISANCE_SETS))}
        sub = lambda idx: {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
                           for k, v in ob.items()}
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n, n)
            if len(np.unique(ob['z'][idx])) < 3:
                continue
            o2 = sub(idx)
            for i, nuis in enumerate(NUISANCE_SETS):
                try:
                    boot[i].append(fit_ruler(o2, ('f',) + nuis, base=known)['params']['f'])
                except Exception:
                    pass
        for i, row in enumerate(tab):
            row['se_boot'] = float(np.std(boot[i], ddof=1)) if len(boot[i]) > 2 else float('nan')
            row['se_lsq'] = fit_ruler(ob, ('f',) + NUISANCE_SETS[i], base=known)['se']['f']
    zs = ob['z'] + prm['delta']
    spread = float(zs.max() / zs.min())
    print(f'Fisher report for f: {desc}; sigma = {sigma:.3f} px; f = {prm["f"]:.1f} px')
    print(f"{'nuisance':<12}{'I_raw':>12}{'I_marg':>12}{'rho':>9}{'SE pred':>10}"
          + (f"{'SE boot':>10}{'SE lsq':>10}" if 'se_boot' in tab[0] else ''))
    for row in tab:
        name = '{' + ','.join(row['nuisance']) + '}'
        line = f"{name:<12}{row['I_raw']:>12.4g}{row['I_marg']:>12.4g}{row['rho']:>9.4f}{row['se_pred']:>10.3f}"
        if 'se_boot' in row:
            line += f"{row['se_boot']:>10.3f}{row['se_lsq']:>10.3f}"
        print(line)
    print(f'distance spread Zmax/Zmin = {spread:.2f}; with that spread, rho = {rho_uniform_w(spread):.4f} '
          f'uniform in 1/Z (limit 1/16), {rho_uniform_z(spread):.4f} uniform in Z (limit 1/4)')
    put_result('fisher', dict(desc=desc, real=real, session=session, sigma=sigma, params=prm, table=tab,
                              spread=spread, z=ob['z'], P1=ob['P1'], P2=ob['P2']), ref=session)
    for fn in (fig_f4, fig_f5):
        print(fn())


# ----------------------------------------------------------------------------
# measure (handoff 2.2)
# ----------------------------------------------------------------------------
def ruler_distance(P1, P2, K, L):
    """Depth of a square-on ruler from its undistorted endpoints, plus lateral offsets."""
    n1 = np.array([(P1[0] - K[0, 2]) / K[0, 0], (P1[1] - K[1, 2]) / K[1, 1]])
    n2 = np.array([(P2[0] - K[0, 2]) / K[0, 0], (P2[1] - K[1, 2]) / K[1, 1]])
    Z = L / np.linalg.norm(n2 - n1)
    mid = (n1 + n2) / 2 * Z
    return float(Z), float(mid[0]), float(mid[1])


def object_extent(bgr, ranges, min_area=200):
    """Largest blob of the colour: its leftmost and rightmost points (silhouette edges)."""
    cs, _ = cv2.findContours(tape_mask(bgr, ranges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    left, right = c[c[:, 0].argmin()], c[c[:, 0].argmax()]
    return c, left.astype(float), right.astype(float)


def object_distance(left, right, K, width, shape):
    """Distance from the silhouette edges of an object of known width.

    flat: Z = W / (x_r - x_l), depth of a face square to the camera.
    cylinder: the edge rays are tangent, so their angle phi gives the axis range R / sin(phi / 2);
    the front surface is R closer. Returns (distance to the front, lateral offset of the centre).
    """
    xl, xr = (left[0] - K[0, 2]) / K[0, 0], (right[0] - K[0, 2]) / K[0, 0]
    if shape == 'cylinder':
        R = width / 2
        phi = math.atan(xr) - math.atan(xl)
        rng = R / math.sin(phi / 2)
        mid = (math.atan(xr) + math.atan(xl)) / 2
        return rng - R, rng * math.sin(mid)
    Z = width / (xr - xl)
    return Z, (xl + xr) / 2 * Z


class Detector:
    """Pretrained COCO object detector (torchvision Faster R-CNN, MobileNetV3 backbone).

    Not trained here: it only finds and labels objects; the distances still come from geometry.
    """

    def __init__(self, min_score=0.5):
        torch = _torch()
        from torchvision.models import detection as D
        w = D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'loading object detector on {self.dev} (first run downloads ~75 MB of weights) ...')
        self.model = D.fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=min_score).eval().to(self.dev)
        self.labels = w.meta['categories']

    def __call__(self, bgr):
        torch = _torch()
        x = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.dev)
        with torch.no_grad():
            out = self.model([x])[0]
        return [dict(box=b.tolist(), label=self.labels[int(l)], score=float(s))
                for b, l, s in zip(out['boxes'].cpu(), out['labels'].cpu(), out['scores'].cpu())]


class DepthNet:
    """Pretrained metric monocular depth (Depth Anything V2, indoor, small): metres per pixel.

    Its absolute scale assumes typical camera optics, so `scene` rescales it with an object of
    known width measured by geometry; the network supplies relative depth between objects.
    """
    NAME = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'

    def __init__(self):
        torch = _torch()
        from transformers import AutoModelForDepthEstimation
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'loading depth network on {self.dev} (first run downloads ~100 MB) ...')
        self.model = AutoModelForDepthEstimation.from_pretrained(self.NAME).to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, bgr, short=518):
        torch = _torch()
        F = torch.nn.functional
        H, W = bgr.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(self.dev)
        x = x.permute(2, 0, 1)[None].float().div(255)
        h, w = (short, round(short * W / H / 14) * 14) if H <= W else (round(short * H / W / 14) * 14, short)
        x = (F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False) - self.mean) / self.std
        with torch.no_grad():
            d = self.model(pixel_values=x).predicted_depth
        return F.interpolate(d[:, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def refine_box(img, box, max_side=220, iters=3):
    """Tight outline of the object inside a detector box: GrabCut seeded with the box, then a
    rotated rectangle. Returns (w_px along the more horizontal side, h_px, corners) or None."""
    c = grabcut_contour(img, box, max_side, iters)
    if c is None:
        return None
    pts = cv2.boxPoints(cv2.minAreaRect(c))
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    a, b = np.linalg.norm(e1), np.linalg.norm(e2)
    w, h = (a, b) if abs(e1[0]) / (a + 1e-9) >= abs(e2[0]) / (b + 1e-9) else (b, a)
    return float(w), float(h), pts


def grabcut_contour(img, box, max_side=220, iters=3):
    """Outline (N x 2 px) of the object inside a detector box, by GrabCut seeded with the box."""
    Hi, Wi = img.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = 0.08 * (x1 - x0) + 2, 0.08 * (y1 - y0) + 2
    X0, Y0, X1, Y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(Wi, x1 + mx)), int(min(Hi, y1 + my))
    crop = img[Y0:Y1, X0:X1]
    if crop.size == 0:
        return None
    s = min(1.0, max_side / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    rx, ry = int((x0 - X0) * s), int((y0 - Y0) * s)
    rect = (max(1, rx), max(1, ry), max(2, min(sw - 2, int((x1 - x0) * s)) - max(0, 1 - rx)),
            max(2, min(sh - 2, int((y1 - y0) * s)) - max(0, 1 - ry)))
    mask = np.zeros((sh, sw), np.uint8)
    try:
        cv2.grabCut(small, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:  # no background around the box (it fills the image)
        return None
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * rect[2] * rect[3]:
        return None
    return c.reshape(-1, 2).astype(np.float32) / s + np.array([X0, Y0], np.float32)


def box_depth(depth, box, frac=0.5):
    """Median network depth over the central part of a box (edges hold background)."""
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * frac / 2, (y1 - y0) * frac / 2
    sub = depth[int(max(0, cy - hh)):int(cy + hh) + 1, int(max(0, cx - hw)):int(cx + hw) + 1]
    return float(np.median(sub)) if sub.size else float('nan')


def cmd_scene(args):
    """Every detected object: distance, width and height (approximate) from detector boxes,
    a depth network, and the calibration; optionally rescaled by one object of known width."""
    calib = load_calib(args.calib)
    det, net = Detector(args.min_score), DepthNet()
    scale = cfg_get('depth_scale', 1.0) if args.ref_width else 1.0
    ref = None
    live = Live(args, 'scene', calib=calib)
    show_depth, cycle = False, False
    while True:
        img = live.frame()
        K = live.calib.K
        dets, depth = det(img), net(img)
        click, live.click = live.click, None
        if args.ref_width:
            if cycle and dets:  # r: next object (left to right) becomes the reference
                order = sorted(dets, key=lambda d: d['box'][0])
                i = next((j for j, d in enumerate(order) if ref is not None and d['box'] == ref['box']), -1)
                ref, cycle = order[(i + 1) % len(order)], False
                print(f"reference: {ref['label']} at x={ref['box'][0]:.0f}..{ref['box'][2]:.0f} px")
            elif click or ref or args.ref_class:  # kept through frames without detections
                ref = pick_target(dets, ref, click, args.ref_class) or ref
        for d in dets:  # tight outline inside each box
            d['tight'] = None if args.no_refine else refine_box(img, d['box'])
        if ref is not None and any(d is ref for d in dets):  # known object's geometry sets the scale
            x0, y0, x1, y1 = ref['box']
            if ref.get('tight'):
                cxm, wpx = (x0 + x1) / 2, ref['tight'][0]
                x0, x1 = cxm - wpx / 2, cxm + wpx / 2
            zg, _ = object_distance(np.array([x0, 0]), np.array([x1, 0]), K, args.ref_width, args.ref_shape)
            zn = box_depth(depth, ref['box'])
            if np.isfinite(zn) and zn > 0:
                scale = 0.8 * scale + 0.2 * zg / zn
        disp = img.copy()
        if show_depth:
            dn = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp = cv2.addWeighted(disp, 0.4, cv2.applyColorMap(255 - dn, cv2.COLORMAP_TURBO), 0.6, 0)
        rows = []
        for d in dets:
            x0, y0, x1, y1 = d['box']
            Z = box_depth(depth, d['box']) * scale
            wpx, hpx = (d['tight'][0], d['tight'][1]) if d['tight'] else (x1 - x0, y1 - y0)
            Wm, Hm = wpx * Z / K[0, 0], hpx * Z / K[1, 1]
            rows.append(dict(label=d['label'], score=d['score'], box=d['box'], Z=Z, W=Wm, H=Hm, ref=d is ref,
                             tight=d['tight'] is not None))
            col = (0, 255, 255) if d is ref else (90, 255, 90)
            if d['tight']:  # thin detector box, thick tight outline
                cv2.rectangle(disp, (int(x0), int(y0)), (int(x1), int(y1)), (150, 150, 150), 1)
                cv2.polylines(disp, [d['tight'][2].astype(np.int32)], True, col, 3 if d is ref else 2)
            else:
                cv2.rectangle(disp, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if d is ref else 2)
            draw_text(disp, [f"{d['label']}" + ('  (reference)' if d is ref else ''),
                             f'{Z * 1e3:.0f} mm away', f'{Wm * 1e3:.0f} x {Hm * 1e3:.0f} mm'],
                      org=(int(x0) + 4, int(y0) + 22), scale=0.6, color=col)
        status = (f'scale x{scale:.3f} from reference ({args.ref_width * 1e3:.0f} mm wide)'
                  + ('' if any(d is ref for d in dets) else ' - reference not seen this frame') if ref is not None else
                  'press r until the object of known width is yellow (sets the scale)' if args.ref_width else
                  'no reference: sizes use the depth network\'s own scale (pass --ref-width to fix it)')
        k = live.show(disp, [live.status_line(img) + f'   calib #{live.calib.id}', f'{len(dets)} objects   {status}',
                             'space log all   r next reference   d depth map   q quit'])
        if k in (ord('q'), 27):
            break
        if k == ord('d'):
            show_depth = not show_depth
        if k == ord('r'):
            cycle = True
        if k == ord(' ') and rows:
            cur = db().execute('INSERT INTO measurements(ts,kind,calib_id,est,truth,brightness,lighting) '
                               'VALUES(?,?,?,?,?,?,?)', (time.time(), 'scene', live.calib.id,
                                                         jdump(dict(objects=rows, scale=scale)), None,
                                                         live.brightness, args.lighting))
            db().commit()
            put_result('scene_frame', dict(measurement=cur.lastrowid), ref=cur.lastrowid, blob=jpg_bytes(disp, 85))
            put_result('scene_raw', dict(measurement=cur.lastrowid), ref=cur.lastrowid, blob=jpg_bytes(img, 92))
            print('logged: ' + '; '.join(f"{r['label']} {r['Z'] * 1e3:.0f} mm, {r['W'] * 1e3:.0f}x{r['H'] * 1e3:.0f} mm"
                                         for r in rows))
    live.close()
    if args.ref_width and ref is not None:
        cfg_set('depth_scale', scale)


# ----------------------------------------------------------------------------
# room: checkerboard floor and walls, object size without a separate calibration (handoff 10.5)
# ----------------------------------------------------------------------------
ROOM_SQUARE = 0.010  # printed square [m]
ROOM_OBJECTS = {  # true L x W x H [m] (bounding box; H = highest point); --object NAME --truth L W H adds more
    'mouse': (0.120, 0.064, 0.040), 'clippers': (0.065, 0.026, 0.045), 'card': (0.085, 0.055, 0.001),
    'paper': (0.090, 0.085, 0.0005), 'cup': (0.090, 0.090, 0.182), 'box': (0.150, 0.060, 0.068),
    'nucleo': (0.082, 0.070, 0.015), 'player': (0.100, 0.058, 0.014), 'ball': (0.04189, 0.04189, 0.04189),
    'car': (0.050, 0.022, 0.014), 'can': (0.066, 0.066, 0.168),  # can: 500 mL, typical size, check
    'transparent_bowl': (0.094, 0.094, 0.044),  # upside down: 94 at the floor, 70 at the top
    'box_vertical': (0.068, 0.060, 0.150),  # the box standing on its end
    'hard_drive': (0.125, 0.080, 0.0115)}
JOINT = 0.012  # squares this close to another panel are not judged (paper margins, the joint) [m]
ROOM_SHAPES = {'round': 4.0, 'box': None, 'dome': 2.0, 'cylinder': 'cyl', 'sphere': 'sph', 'revolve': 'rev'}
EXTRA_PARAMS = {'revolve': 4}  # revolve: diameter at 4 heights (base, 1/3, 2/3, top), relative to L  # superellipsoid exponent; None: box
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
_UNIT_SHAPES = {}


def grid_cells_ok(gray, c, tol=0.9):
    """A found grid is real if each cell between four neighbouring corners is one uniform
    colour and the colours alternate. OpenCV sometimes grows a sheared sub-lattice on strongly
    slanted panels (every third corner); that still fits a homography, but its cells straddle
    several squares and fail here."""
    a, b, d, e = c[:-1, :-1], c[:-1, 1:], c[1:, :-1], c[1:, 1:]
    g = gray.astype(np.float32)
    vals = []
    for s, t in ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5)):
        p = (1 - s) * (1 - t) * a + s * (1 - t) * b + (1 - s) * t * d + s * t * e
        vals.append(cv2.remap(g, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR))
    dark = np.stack(vals) < np.median(vals)
    uniform = (dark.all(0) | (~dark).all(0)).mean()
    parity = np.add.outer(np.arange(c.shape[0] - 1), np.arange(c.shape[1] - 1)) % 2 == 1
    alternate = max((dark[4] == parity).mean(), (dark[4] != parity).mean())
    return uniform >= tol and alternate >= tol


def find_planes(gray, hints=(), full=True, seed=(5, 5), max_planes=6, min_side=0.3, stats=None):
    """Every checkerboard panel in view, each as its own grid (rows x cols x 2 corners, px).

    Panels are not assumed to meet at right angles or to line up: each one is its own plane,
    so a bent middle or sheets glued slightly off do not matter. A plain checkerboard has no
    origin, so grid indices start at an arbitrary corner; that shifts a plane's coordinates and
    leaves every length unchanged. OpenCV's sector detector grows the largest grid it can from
    a 5x5 seed; a found grid is painted grey and its window searched again; a window with no
    valid grid is split into two overlapping halves, down to `min_side` of the frame.
    `hints` are last frame's grids: each one's area is warped face-on with last frame's
    homography (squares 20 px, 3 squares of margin), where the detector finds it easily even on
    steeply slanted panels; the corners are mapped back and refined in the frame. `full` adds a
    search of the whole frame. stats['lost'] counts hints of 60+ corners found with under 70 %."""
    g = gray.copy()
    H, W = g.shape
    flags = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    planes = []

    def accept(c):
        if not grid_cells_ok(gray, c):
            return False
        planes.append(c)
        step = float(np.median(np.linalg.norm(np.diff(c, axis=1), axis=2)))
        m = np.zeros_like(g)
        cv2.fillPoly(m, [cv2.convexHull(c.reshape(-1, 2)).astype(np.int32)], 255)
        g[cv2.dilate(m, np.ones((int(0.6 * step) + 1,) * 2, np.uint8)) > 0] = 128
        return True

    def grid(sub, x, y):
        ok, c, meta = cv2.findChessboardCornersSBWithMeta(sub, seed, flags=flags)
        return ok and accept(c.reshape(meta.shape[0], meta.shape[1], 2) + np.float32([x, y]))

    px, mg = 20.0, 3
    lost = 0
    for prev in hints:
        r, k = prev.shape[:2]
        lost += r * k >= 60  # until found at 70 % of its corners or more (small pieces don't count)
        jj, ii = np.meshgrid(np.arange(k), np.arange(r))
        Hg, _ = cv2.findHomography(np.stack([jj.ravel(), ii.ravel()], 1).astype(np.float32), prev.reshape(-1, 2))
        if Hg is None:
            continue
        T = np.array([[px, 0, mg * px], [0, px, mg * px], [0, 0, 1]])
        A = T @ np.linalg.inv(Hg)  # frame -> face-on view
        size = (int((k - 1 + 2 * mg) * px) + 1, int((r - 1 + 2 * mg) * px) + 1)
        face = cv2.warpPerspective(g, A, size, flags=cv2.INTER_LINEAR, borderValue=128)
        ok, c, meta = cv2.findChessboardCornersSBWithMeta(face, seed, flags=flags)
        if not ok:
            continue
        c = cv2.perspectiveTransform(c.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(A))
        c = c.reshape(meta.shape[0], meta.shape[1], 2).astype(np.float32)
        while c.shape[0] >= 3 and c.shape[1] >= 3:  # trim edge rows/columns that leave the frame
            out = (c[..., 0] <= 3) | (c[..., 0] >= W - 4) | (c[..., 1] <= 3) | (c[..., 1] >= H - 4)
            if not out.any():
                break
            edges = [out[0].sum(), out[-1].sum(), out[:, 0].sum(), out[:, -1].sum()]
            c = [c[1:], c[:-1], c[:, 1:], c[:, :-1]][int(np.argmax(edges))]
        if c.shape[0] < 3 or c.shape[1] < 3:
            continue
        step = float(np.median(np.linalg.norm(np.diff(c, axis=1), axis=2)))
        win = int(np.clip(step / 3, 2, 6))
        rk = c.shape[:2]
        c = cv2.cornerSubPix(gray, np.ascontiguousarray(c.reshape(-1, 1, 2)), (win, win), (-1, -1), crit)
        if accept(c.reshape(rk[0], rk[1], 2)) and rk[0] * rk[1] >= 0.7 * r * k and r * k >= 60:
            lost -= 1
    if stats is not None:
        stats['lost'] = lost
    todo = [(0, 0, W, H)] if full else []
    while todo and len(planes) < max_planes:
        x, y, w, h = todo.pop(0)
        if w < 40 or h < 40:
            continue
        if grid(g[y:y + h, x:x + w], x, y):
            todo.insert(0, (x, y, w, h))
        elif w >= 1.6 * min_side * W or h >= 1.6 * min_side * H:
            if w >= h:
                todo += [(x, y, w // 2 + w // 8, h), (x + w // 2 - w // 8, y, w - w // 2 + w // 8, h)]
            else:
                todo += [(x, y, w, h // 2 + h // 8), (x, y + h // 2 - h // 8, w, h - h // 2 + h // 8)]
    return planes


def grid_points(c, square=ROOM_SQUARE):
    """Plane coordinates [m] (z = 0) of a rows x cols corner grid."""
    r, k = c.shape[:2]
    jj, ii = np.meshgrid(np.arange(k), np.arange(r))
    return np.stack([jj.ravel() * square, ii.ravel() * square, np.zeros(r * k)], 1).astype(np.float32)


def plane_groups(normals, tol_deg=10.0):
    """Group index per plane: pieces whose normals agree within tol_deg share an orientation
    (e.g. the floor seen in two pieces on either side of an object)."""
    groups, reps = [], []
    for n in normals:
        for gi, r in enumerate(reps):
            if abs(float(n @ r)) > math.cos(math.radians(tol_deg)):
                groups.append(gi)
                break
        else:
            groups.append(len(reps))
            reps.append(n)
    return groups


def calib_from_planes(planes, size):
    """Intrinsics from a single frame: every plane is a separate view (Zhang's method), so the
    angles between the panels never enter. Three orientations fix f and the principal point;
    with two, the principal point is held at the image centre. None with one orientation."""
    objs = [grid_points(c) for c in planes]
    imgs = [c.reshape(-1, 1, 2).astype(np.float32) for c in planes]
    base = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO

    def run(flags):
        K = cv2.initCameraMatrix2D(objs, imgs, size)
        K[0, 2], K[1, 2] = (size[0] - 1) / 2, (size[1] - 1) / 2
        return cv2.calibrateCamera(objs, imgs, size, K, None, flags=flags)

    rms, K, dist, rv, _ = run(base | cv2.CALIB_FIX_PRINCIPAL_POINT)
    n_orient = len(set(plane_groups([cv2.Rodrigues(r)[0][:, 2] for r in rv])))
    if n_orient < 2:
        return None
    if n_orient >= 3:
        rms, K, dist, rv, _ = run(base)
    return dict(K=K, dist=dist.ravel(), rms=float(rms), n_orient=n_orient, n_planes=len(planes))


def plane_poses(planes, K, dist):
    """Pose of each plane (planar PnP), its normal in camera coordinates and reprojection rms."""
    out = []
    for c in planes:
        obj, img = grid_points(c), c.reshape(-1, 1, 2).astype(np.float64)
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        rep = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2) - c.reshape(-1, 2)
        out.append(dict(rvec=rvec, tvec=tvec, R=R, t=tvec.ravel(), n=R[:, 2],
                        rms=float(np.sqrt((rep ** 2).sum(1).mean())), n_pts=len(obj)))
    return out


def merge_pieces(planes, poses, idxs, K, dist, max_rms=0.002):
    """One pose from several pieces of the same plane (e.g. the floor on both sides of an
    object). The largest piece sets the plane coordinates; every other piece is placed in them
    by its own 2D rotation (or mirror: grid indices have no handedness) and shift, fitted to
    where its corners' rays meet the plane, so sheets glued slightly off still fit. Then the
    pose is solved from all corners. Returns (pose, number of pieces used); the pose also holds
    'base' (index of the largest piece) and 'obj' (all used corners in its plane coordinates)."""
    base = max(idxs, key=lambda i: poses[i]['n_pts'])
    pb = poses[base]
    obj, img, used = [grid_points(planes[base])], [planes[base].reshape(-1, 2)], 1
    for i in idxs:
        if i == base or abs(float((poses[i]['t'] - pb['t']) @ pb['n'])) > 0.01:
            continue  # parallel but not the same plane
        dst = pixels_to_plane(planes[i].reshape(-1, 2), K, dist, pb['R'], pb['t'])[0][:, :2]
        src = grid_points(planes[i])[:, :2].astype(np.float64)
        ms, md = src.mean(0), dst.mean(0)
        U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
        moved = (src - ms) @ (U @ Vt) + md  # best rotation or mirror
        if np.sqrt(((moved - dst) ** 2).sum(1).mean()) > max_rms:
            continue
        obj.append(np.c_[moved, np.zeros(len(moved))].astype(np.float32))
        img.append(planes[i].reshape(-1, 2))
        used += 1
    if used == 1:
        return dict(pb, base=base, obj=obj[0], img=img[0]), 1
    obj, img = np.vstack(obj), np.vstack(img).reshape(-1, 1, 2).astype(np.float64)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, pb['rvec'].copy(), pb['tvec'].copy())
    R = cv2.Rodrigues(rvec)[0]
    rep = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2) - img.reshape(-1, 2)
    return dict(rvec=rvec, tvec=tvec, R=R, t=tvec.ravel(), n=R[:, 2], base=base, obj=obj, img=img.reshape(-1, 2),
                rms=float(np.sqrt((rep ** 2).sum(1).mean())), n_pts=len(obj)), used


def shade_free(gray, planes):
    """The image divided by its local mean brightness over about three squares, so a shadow (which
    dims black and white squares by the same factor) no longer looks like an object; an object
    (one colour, no pattern) still differs from the squares around it. Scaled to 0..255."""
    step = float(np.median([np.median(np.linalg.norm(np.diff(c, axis=1), axis=2)) for c in planes])) if planes else 12.0
    k = int(3 * step) | 1  # 3 squares: best of 2, 3, 4, 6 on stored frames (MAE 9.7 / 9.2 / 6.3 mm)
    g = gray.astype(np.float32)
    local = cv2.blur(g, (k, k)) + 1.0
    return np.clip(g / local * 127.5, 0, 255).astype(np.uint8)


def room_occupancy(gray, planes, gposes, K, dist, square=ROOM_SQUARE, reach=40):
    """Where the room's pattern is hidden: each panel orientation's grid (aligned to its largest
    piece) is extended `reach` squares past what was detected, over the part of the room in view
    (the hull of all detected corners). A cell is free if its middle is one uniform colour of the
    expected parity, occupied otherwise. Cells behind another panel are skipped (the room is the
    convex region between its panels, so the first panel a ray meets is the one seen).
    Returns the occupied and free masks, the occupied mask of each orientation, the square size
    in px, and each orientation's grey levels (darkest black, lightest white, parity)."""
    H, W = gray.shape
    g = gray.astype(np.float32)
    allc = np.vstack([c.reshape(-1, 2) for c in planes])
    step = float(np.median([np.median(np.linalg.norm(np.diff(c, axis=1), axis=2)) for c in planes]))
    region = np.zeros((H, W), np.uint8)  # cell middles must lie inside the detected corners' hull
    cv2.fillPoly(region, [cv2.convexHull(allc).astype(np.int32)], 255)
    occ, free, per, levels = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), {}, {}
    for c in planes:  # detected grids are visible pattern by definition
        cv2.fillPoly(free, [cv2.convexHull(c.reshape(-1, 2)).astype(np.int32)], 255)
    offs = [(0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6), (0.5, 0.5)]  # +-4 mm from the middle
    for gid, gp in gposes.items():
        # brightness threshold and colour parity from the largest detected piece
        base = planes[gp['base']]
        a, b, d, e = base[:-1, :-1], base[:-1, 1:], base[1:, :-1], base[1:, 1:]
        mid = ((a + b + d + e) / 4).astype(np.float32)
        v = cv2.remap(g, mid[..., 0], mid[..., 1], cv2.INTER_LINEAR)
        par = np.add.outer(np.arange(v.shape[0]), np.arange(v.shape[1])) % 2 == 1
        lo_v, hi_v = np.sort(v.ravel())[[len(v.ravel()) // 4, 3 * len(v.ravel()) // 4]]
        thr = (lo_v + hi_v) / 2
        dark_odd = ((v < thr) == par).mean() > 0.5  # grid_points: column index is x, row index is y
        # clearly black or clearly white: a mid-grey object (a shaded side) matches neither
        d_max, w_min = lo_v + 0.35 * (hi_v - lo_v), hi_v - 0.35 * (hi_v - lo_v)
        levels[gid] = (d_max, w_min, dark_odd)
        lo = np.floor(gp['obj'][:, :2].min(0) / square).astype(int) - reach
        hi = np.ceil(gp['obj'][:, :2].max(0) / square).astype(int) + reach
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))  # ii: x cells, jj: y cells
        ii, jj = ii.ravel(), jj.ravel()
        pts = [np.stack([ii + s, jj + t], 1) for s, t in [(0, 0), (1, 0), (1, 1), (0, 1)] + offs]
        P = np.vstack(pts) * square
        P3 = np.c_[P, np.zeros(len(P))]
        Pc = P3 @ gp['R'].T + gp['t']  # camera coordinates
        n = len(ii)
        ok = (Pc[:, 2] > 1e-3).reshape(9, n).all(0)
        uv = cv2.projectPoints(P3, gp['rvec'], gp['tvec'], K, dist)[0].reshape(9, n, 2)
        ok &= ((uv[..., 0] > 1) & (uv[..., 0] < W - 2) & (uv[..., 1] > 1) & (uv[..., 1] < H - 2)).all(0)
        ctr = Pc[8 * n:9 * n]
        for oid, op in gposes.items():
            if oid == gid:
                continue
            den = ctr @ op['n']
            with np.errstate(divide='ignore', invalid='ignore'):
                lam = float(op['n'] @ op['t']) / den
            ok &= ~((lam > 0) & (lam < 1 - 1e-3))  # another panel in front of this cell
            ok &= np.abs((ctr - op['t']) @ op['n']) > JOINT  # paper margins and joints where panels meet
        cu = np.clip(uv[8], 0, [W - 1, H - 1]).astype(int)
        ok &= region[cu[:, 1], cu[:, 0]] > 0
        quads = uv[:4].transpose(1, 0, 2)
        d1, d2 = quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1]
        area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        edge = np.linalg.norm(quads - np.roll(quads, 1, axis=1), axis=2).max(1)
        ok &= (area > 9) & (edge < 4 * step)  # long slivers: points the lens model folds back in
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        samp = uv[4:9][:, idx].astype(np.float32)  # 5 x m x 2
        vals = cv2.remap(g, samp[..., 0], samp[..., 1], cv2.INTER_LINEAR)  # 5 x m
        expect = ((ii[idx] + jj[idx]) % 2 == 1) == dark_odd
        bad = ~np.where(expect, (vals < d_max).all(0), (vals > w_min).all(0))
        q = quads[idx].astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        if bad.any():
            cv2.fillPoly(m, list(q[bad]), 255)
        if (~bad).any():
            cv2.fillPoly(free, list(q[~bad]), 255)
        per[gid] = m
        occ |= m
    occ[free > 0] = 0
    return occ, free, per, step, levels


def pattern_mismatch(gray, rect, gposes, levels, K, dist, square=ROOM_SQUARE, edge=0.15, maps=None):
    """Pixels inside rect (x, y, w, h) whose grey level is not the room pattern's colour there.
    Each pixel's ray meets the first panel in front of it, giving the expected square colour;
    pixels within `edge` of a square's border (blurred) are never counted. A one-colour object
    shows on the squares of the other colour, so it shows its edges to the pixel there.
    Also edges the pattern does not explain (away from every square border): an object's outline
    even where it has the colour of the square under it (black on black).
    Returns (off-pattern pixels, pixels on a black square still black, on a white square still
    white, unexplained edges)."""
    x, y, w, h = rect
    uu, vv = np.meshgrid(np.arange(x, x + w, dtype=np.float64), np.arange(y, y + h, dtype=np.float64))
    n = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))]
    best = np.full(len(d), np.inf)
    bad, room_b, room_w = np.zeros(len(d), bool), np.zeros(len(d), bool), np.zeros(len(d), bool)
    away, pat = np.zeros(len(d), bool), np.zeros(len(d), np.uint8)
    sub = cv2.GaussianBlur(gray[y:y + h, x:x + w], (0, 0), 1.0)
    contrast = []
    g = gray[y:y + h, x:x + w].astype(np.float32).ravel()
    for gid, gp in gposes.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = float(gp['n'] @ gp['t']) / (d @ gp['n'])
        hit = (lam > 0) & (lam < best)
        best[hit] = lam[hit]
        X = d[hit] * lam[hit, None]  # camera coordinates of the hit
        P = (X - gp['t']) @ gp['R']
        c = P[:, :2] / square
        clear = np.ones(len(X), bool)
        for oid, op in gposes.items():
            if oid != gid:
                clear &= np.abs((X - op['t']) @ op['n']) > JOINT
        fr = c - np.floor(c)
        inner = ((fr > edge) & (fr < 1 - edge)).all(1)
        d_max, w_min, dark_odd = levels[gid]
        expect = ((np.floor(c[:, 0]) + np.floor(c[:, 1])) % 2 == 1) == dark_odd
        gv = g[hit]
        bad[hit] = inner & clear & np.where(expect, gv >= d_max, gv <= w_min)
        room_b[hit] = inner & expect & (gv < d_max)
        room_w[hit] = inner & ~expect & (gv > w_min)
        away[hit] = clear
        pat[hit] = 1 + expect  # expected colour; square borders are where it changes
        contrast.append((w_min - d_max) / 0.3)  # white minus black level
    # 2 px either side of every predicted square border (in pixels: far squares are only a few px)
    border = cv2.morphologyEx(pat.reshape(h, w), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).ravel() > 0
    away &= ~border
    if contrast:  # one-pixel edges at the gradient's peak (Canny), so the outline is not widened
        c = min(contrast)
        edges = away & (cv2.Canny(sub, 0.25 * c, 0.5 * c, L2gradient=True).ravel() > 0)
    else:
        edges = away
    if maps is not None:  # for the learned outline: expected colour (0 none, 1 white, 2 black) and borders
        maps.update(pat=pat.reshape(h, w), border=border.reshape(h, w))
    return bad.reshape(h, w), room_b.reshape(h, w), room_w.reshape(h, w), edges.reshape(h, w)


def room_blobs(occ, per_floor, step, min_cells=4):
    """Objects from the occupied cells: a closing fills the checker pattern a one-colour object
    leaves (it matches every other cell). A blob needs a core thicker than about one cell (an
    opening drops lone strips: paper margins); thin parts attached to a core stay (a wire, a pen
    clip). A blob must cover occupied floor cells, i.e. stand on the floor; parts seen over a wall
    within two cells join it across the joint strip, which is not judged.
    Returns a list of (blob mask, bounding rect)."""
    k = int(round(1.5 * step)) | 1
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k2 = int(round(1.2 * step)) | 1
    core = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))) > 0
    n, lab = cv2.connectedComponents(closed)
    floor, rest = [], []
    for i in range(1, n):
        b = lab == i
        if (b & core).sum() < min_cells * step * step:
            continue
        (floor if (b & (per_floor > 0)).sum() >= 2 * step * step else rest).append(b)
    reach = np.ones((int(4 * step) | 1,) * 2, np.uint8)
    out = []
    for b in floor:
        near = cv2.dilate(b.astype(np.uint8), reach) > 0
        for r in rest:
            if (r & near).any():
                b = b | r
        x, y, w, h = cv2.boundingRect(b.astype(np.uint8))
        out.append((b, (int(x), int(y), int(w), int(h))))
    return out


def grabcut_seeded(img, fg, bg, rect, sure_fg=None, max_side=240, iters=3):
    """Outline (N x 2 px) of an object from GrabCut started with probable-object pixels `fg`,
    sure-background pixels `bg` and optional sure-object pixels (boolean masks of the rect
    (x, y, w, h))."""
    x, y, w, h = rect
    crop = img[y:y + h, x:x + w]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[fg] = cv2.GC_PR_FGD
    mask[bg] = cv2.GC_BGD
    if sure_fg is not None:
        mask[sure_fg] = cv2.GC_FGD
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not (mask == cv2.GC_PR_FGD).any() or not (mask == cv2.GC_BGD).any():
        return None
    try:
        cv2.grabCut(small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float32) / s + np.array([x, y], np.float32)


def corner_frame(gposes, fg):
    """Fixed room coordinates from one frame: origin where the floor and two walls meet, z up
    from the floor, x along the inward normal of one wall, y = z x x leaning into the other
    (which wall is x is the one choice that makes y point into the room as well). Needs no
    tracking, so views from anywhere share it. None without the floor and two walls."""
    walls = sorted((g for g in gposes if g != fg), key=lambda g: -gposes[g]['n_pts'])[:2]
    if len(walls) < 2:
        return None
    inward = {}
    for g in [fg] + walls:  # normals pointing to the camera's side (inside the room)
        n, t = gposes[g]['n'], gposes[g]['t']
        inward[g] = n if float(n @ -t) > 0 else -n
    z = inward[fg]
    for a, b in (walls, walls[::-1]):
        x = inward[a] - (inward[a] @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        if float(y @ inward[b]) > 0:
            break
    N = np.array([gposes[g]['n'] for g in [fg] + walls])
    o = np.linalg.solve(N, np.array([gposes[g]['n'] @ gposes[g]['t'] for g in [fg] + walls]))
    R = np.stack([x, y, z], 1)
    return dict(R=R, t=o, rvec=cv2.Rodrigues(R)[0], tvec=o.reshape(3, 1), n=z)


def outline_samples(hull, n=32):
    """n points evenly spaced along a closed outline."""
    h = np.asarray(hull, float).reshape(-1, 2)
    seg = np.roll(h, -1, axis=0) - h
    s = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    t = np.linspace(0, s[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(s, t, side='right') - 1, 0, len(h) - 1)
    f = (t - s[i]) / np.maximum(s[i + 1] - s[i], 1e-9)
    return h[i] + f[:, None] * seg[i]


def view_sigma(ps):
    """Outline noise [px] a view is trusted with: the camera's own measured noise, at least `sigma_min` (an outline grown from
    the other camera's prediction cannot be trusted as much as its small residual says)."""
    return max(float(ps.get('sigma', 1.0)), float(ps.get('sigma_min', 0.0)), 1.0)


def hull_iou(a, b):
    """Overlap (intersection over union) of the convex hulls of two outlines."""
    a, b = [cv2.convexHull(np.asarray(h, np.float32).reshape(-1, 2)) for h in (a, b)]
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / max(cv2.contourArea(a) + cv2.contourArea(b) - inter, 1e-9))


def size_information(p, shape, up, views, K, dist, sigma_px=1.0, report=None):
    """How well the views pin down each size (handoff 3, gravity-paper projection identity).

    Residuals: points along each view's observed outline, signed pixel distance to the fitted
    shape's projected outline, noise sigma_px (None: each view's own residual rms, at least 1 px, so a noisy camera counts
    for less: the information is the sum of J_v^T J_v / sigma_v^2 over the views; `report` gets the sigma_v). Columns: the sensitivities to every parameter, by
    central differences (sigma_px None: the fit's own residual rms). For each size (in log units) b is its column and A all the others
    (position, angle, other sizes, profile): rho = I_marg / I_raw is the share of information left
    after the nuisances are estimated, and I_marg^-1/2 its standard error. Returns
    {'L'|'W'|'H': (rho, sd [m])}."""
    samples = [(outline_samples(h), ps) for h, ps in views]
    counts = [len(s) for s, _ in samples]

    def resid(q):
        pts, out = shape_points(q, shape, up), []
        for s, ps in samples:
            uv = cv2.projectPoints(pts, ps['rvec'], ps['tvec'], ps.get('K', K), ps.get('dist', dist))[0].reshape(-1, 2)
            c = cv2.convexHull(uv.astype(np.float32))
            out += [cv2.pointPolygonTest(c, (float(x), float(y)), True) for x, y in s]
        return np.array(out)

    r0 = resid(p)  # in px
    if sigma_px is None:  # the outline noise is what the fit leaves, per view: its residual rms (at least 1 px)
        cuts = np.cumsum(counts)[:-1]
        sig = np.array([max(float(np.sqrt(np.mean(r ** 2))), float(ps.get('sigma_min', 0.0)), 1.0)
                        for r, (_, ps) in zip(np.split(r0, cuts), samples)])
    else:
        sig = np.full(len(samples), float(sigma_px))
    if report is not None:
        report[:] = [float(v) for v in sig]
    scale = np.repeat(sig, counts)
    steps = np.r_[5e-4, 5e-4, 5e-3, 5e-3, 5e-3, 5e-3, np.full(len(p) - 6, 5e-3)]
    J = np.stack([(resid(p + np.eye(len(p))[k] * h) - resid(p - np.eye(len(p))[k] * h)) / (2 * h * scale)
                  for k, h in enumerate(steps)], 1)
    norms = np.linalg.norm(J, axis=0)
    live = norms > 1e-9 * max(norms.max(), 1e-12)  # tied sizes have no column of their own
    sizes = np.exp(tied(p, shape)[3:6])
    li = 3 if sizes[0] >= sizes[1] else 4  # which parameter is the longer side
    out = {}
    for d, k in (('L', li), ('W', 7 - li), ('H', 5)):
        k = k if live[k] else 3  # tied to L (cylinder, sphere, revolve)
        others = [j for j in range(len(p)) if j != k and live[j]]
        A = J[:, others] / norms[others]  # column-scaled, as the handoff asks
        i_raw, i_marg, rho = retention(J[:, k], A)
        size = float(revolve_profile(p).max()) if shape == 'revolve' and d != 'H' else float(sizes[min(k, 5) - 3])
        out[d] = (rho, size / math.sqrt(i_marg) if i_marg > 0 else float('inf'))
    return out


def view_ious(p, shape, up, views, K, dist):
    """Overlap of the fitted shape's outline with each view's outline."""
    pts, out = shape_points(p, shape, up), []
    for h, ps in views:
        hh = np.asarray(h, np.float32).reshape(-1, 2)
        q = cv2.convexHull(cv2.projectPoints(pts, ps['rvec'], ps['tvec'], ps.get('K', K), ps.get('dist', dist))[0]
                           .reshape(-1, 2).astype(np.float32))
        inter, _ = cv2.intersectConvexConvex(q, hh)
        out.append(inter / max(cv2.contourArea(q) + cv2.contourArea(hh) - inter, 1e-9))
    return np.array(out)


def pixels_to_plane(uv, K, dist, R, t):
    """Plane coordinates of the points where the pixels' rays meet the plane, and the ray
    parameter (negative: the ray points away from the plane)."""
    n = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))]
    lam = float(R[:, 2] @ t) / (d @ R[:, 2])
    return (d * lam[:, None] - t) @ R, lam


def unit_shape(shape):
    """Surface samples of a shape of length, width and height 1 standing on z = 0."""
    if shape not in _UNIT_SHAPES:
        e = ROOM_SHAPES[shape]
        if e is None:
            a = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
            u = np.vstack([np.c_[a, np.zeros(4)], np.c_[a, np.ones(4)]])
        elif e == 'sph':  # ellipsoid resting on the floor (a ball): widest at half height
            th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 32, endpoint=False), np.linspace(-np.pi / 2, np.pi / 2, 13))
            u = np.stack([0.5 * np.cos(th) * np.cos(ph), 0.5 * np.sin(th) * np.cos(ph), 0.5 + 0.5 * np.sin(ph)],
                         -1).reshape(-1, 3)
            u = np.vstack([u[6 * 32:7 * 32], u])  # equator first: the drawn outline
        elif e == 'cyl':  # upright (elliptic) cylinder: base ring then top ring
            th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            ring = np.c_[0.5 * np.cos(th), 0.5 * np.sin(th)]
            u = np.vstack([np.c_[ring, np.zeros(32)], np.c_[ring, np.ones(32)]])
        else:  # upper half of a superellipsoid; ring 0 is the base outline
            th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 32, endpoint=False), np.linspace(0, np.pi / 2, 9))
            sp = lambda x: np.sign(x) * np.abs(x) ** (2 / e)
            u = np.stack([0.5 * sp(np.cos(th)) * sp(np.cos(ph)), 0.5 * sp(np.sin(th)) * sp(np.cos(ph)),
                          sp(np.sin(ph))], -1).reshape(-1, 3)
        _UNIT_SHAPES[shape] = u
    return _UNIT_SHAPES[shape]


ROUND_TIES = {'cylinder': (0, 0, 2), 'sphere': (0, 0, 0), 'revolve': (0, 0, 2)}  # round objects: W = L (and H = L for a ball)


def tied(p, shape):
    """Parameters with the round shapes' sizes tied: a cup's depth is its width, a ball is a ball.
    One view then determines them (diameter from the outline's width, height from its top)."""
    t = ROUND_TIES.get(shape)
    if t is None:
        return p
    q = np.array(p, float)
    q[3:6] = [p[3 + i] for i in t]
    return q


def revolve_profile(p):
    """Diameters [m] at the revolve knots (base, 1/3, 2/3, top) from its parameters."""
    return math.exp(p[3]) * np.exp(np.asarray(p[6:10]))


def shape_points(p, shape, up):
    """World points (floor coordinates) of a shape: p = (cx, cy, yaw, log L, log W, log H, extra...)."""
    p = tied(p, shape)
    if shape == 'revolve':  # round, diameter varying linearly between 4 heights; base ring first
        d = revolve_profile(p)
        zf = np.linspace(0, 1, 13)
        dz = np.interp(zf, [0, 1 / 3, 2 / 3, 1], d)
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        Hh = math.exp(p[5])
        pts = [np.c_[p[0] + 0.5 * dk * np.cos(th), p[1] + 0.5 * dk * np.sin(th), np.full(32, up * z * Hh)]
               for z, dk in zip(zf, dz)]
        return np.vstack(pts)
    u = unit_shape(shape)
    L, W, H = np.exp(p[3:6])
    c, s = math.cos(p[2]), math.sin(p[2])
    x, y = u[:, 0] * L, u[:, 1] * W
    return np.stack([p[0] + x * c - y * s, p[1] + x * s + y * c, up * u[:, 2] * H], 1)


def fit_shape(hull, pose, K, dist, up, shape='round', near=(), views=()):
    """Upright shape standing on the floor whose projected outline best overlaps the object's
    outline (intersection over union of the two convex hulls).

    The floor fixes where the object stands and its scale; one view leaves height and the depth
    across the line of sight partly traded against each other, so the height is the least sure
    number, best from a low camera. `near`: last frame's fitted parameters; one standing within
    3 cm replaces the cold starts (live: much faster, and steadier). `views`: more (hull, pose)
    pairs of the same object from other camera positions, all in the same floor coordinates; the
    cost is the mean over all views, which separates height from depth.
    Returns dict(L, W, H [m], yaw, IoU, params) or None."""
    hull = np.asarray(hull, np.float32).reshape(-1, 2)
    A = cv2.contourArea(hull)
    F, lam = pixels_to_plane(hull, K, dist, pose['R'], pose['t'])
    if (lam > 0).sum() < 3 or A < 20:
        return None
    # starts: (1) the outline projected onto the floor, shrunk (good for flat objects); (2) the
    # outline point whose ray meets the floor nearest the camera touches the floor: its depth
    # turns the outline's pixel size into metres (good for tall objects, whose projection runs off)
    F, lam_ok = F[lam > 0, :2], lam[lam > 0]
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(F.astype(np.float32))
    rw, rh = max(rw, 1e-3), max(rh, 1e-3)
    cam = -pose['R'].T @ pose['t']
    i0 = int(np.argmin(np.linalg.norm(F - cam[:2], axis=1)))
    Zc = float(lam_ok[i0])  # depth along the optical axis of the contact point
    w_px, h_px = np.ptp(hull[:, 0]), np.ptp(hull[:, 1])
    s0 = max(w_px * Zc / K[0, 0], 1e-3)
    away = (F[i0] - cam[:2]) / (np.linalg.norm(F[i0] - cam[:2]) + 1e-9)
    c2 = F[i0] + away * s0 / 2
    h0 = max(h_px * Zc / K[1, 1] * 0.7, 1e-3)
    starts = [(cx, cy, math.radians(ang) + dy, 0.8 * rw, 0.8 * rh, hf * max(rw, rh))
              for dy in (0, math.pi / 2) for hf in (0.25, 0.6)]
    starts += [(c2[0], c2[1], math.atan2(away[1], away[0]) + dy, s0, s0, h0) for dy in (0, math.pi / 4)]
    warm = [p for p in near if len(p) == 6 + EXTRA_PARAMS.get(shape, 0)
            and min(math.dist(p[:2], (cx, cy)), math.dist(p[:2], c2)) < 0.03]
    if warm:
        starts = [(p[0], p[1], p[2], *np.exp(p[3:6]), None, *p[6:]) for p in warm[:1]]

    obs = [(hull, pose, A)] + [(np.asarray(h, np.float32).reshape(-1, 2), ps, cv2.contourArea(np.asarray(h, np.float32)))
                               for h, ps in views]

    wts = np.array([1.0 / view_sigma(ps) ** 2 for _, ps, _ in obs])  # a noisy camera pulls the fit less
    wts /= wts.sum()

    def cost(p):
        pts, c = shape_points(p, shape, up), 0.0
        for (hh, ps, aa), w in zip(obs, wts):
            uv = cv2.projectPoints(pts, ps['rvec'], ps['tvec'], ps.get('K', K), ps.get('dist', dist))[0]
            q = cv2.convexHull(uv.reshape(-1, 2).astype(np.float32))
            inter, _ = cv2.intersectConvexConvex(q, hh)
            c += w * (1 - inter / (cv2.contourArea(q) + aa - inter))
        return c

    from scipy.optimize import minimize
    best, nx = None, EXTRA_PARAMS.get(shape, 0)
    for st in starts:
        sx, sy, yaw0, l0, w0, hh0 = st[:6]
        x0 = np.r_[sx, sy, yaw0, math.log(l0), math.log(w0), math.log(hh0), np.zeros(nx)]
        if len(st) == 6 + nx + 1:  # warm start carries its own extra parameters
            x0[6:] = st[7:]
        simplex = np.vstack([x0, x0 + np.diag(np.r_[0.015, 0.015, 0.4, 0.3, 0.3, 0.4, np.full(nx, 0.2)])])
        r = minimize(cost, x0, method='Nelder-Mead',
                     options=dict(initial_simplex=simplex, maxiter=1500, xatol=1e-4, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    for _ in range(3):  # restart from the best point: Nelder-Mead stalls in a collapsed simplex
        simplex = np.vstack([best.x, best.x + np.diag(np.r_[0.005, 0.005, 0.15, 0.1, 0.1, 0.1, np.full(nx, 0.07)])])
        r = minimize(cost, best.x, method='Nelder-Mead',
                     options=dict(initial_simplex=simplex, maxiter=1500, xatol=1e-5, fatol=1e-8))
        if r.fun >= best.fun - 1e-6:
            break
        best = r
    best.x = tied(best.x, shape)
    L, W, H = np.exp(best.x[3:6])
    out = dict(L=float(max(L, W)), W=float(min(L, W)), H=float(H), yaw=float(best.x[2]),
               iou=float(1 - best.fun), p=best.x)
    if shape == 'revolve':
        prof = revolve_profile(best.x)
        out.update(L=float(prof.max()), W=float(prof.max()), profile=[float(v) for v in prof])
    return out


def coplanar_cost(planes, f, size, dist):
    """How badly boards that lie on one table disagree about the table when the focal length is f:
    rms [m] of the board centres' distance from their common plane plus the boards' normals' tilt
    (over a 5 cm lever). A wrong f bends the boards' poses differently depending on where they
    sit in the image, so only the true f makes them coplanar."""
    K = np.array([[f, 0, (size[0] - 1) / 2], [0, f, (size[1] - 1) / 2], [0, 0, 1.0]])
    poses = plane_poses(planes, K, dist)
    N = np.array([p['n'] for p in poses])
    N = N * np.where(N @ N[0] >= 0, 1.0, -1.0)[:, None]
    nbar = np.linalg.eigh(N.T @ N)[1][:, -1]
    C = np.array([p['R'] @ np.array([(c.shape[1] - 1) * ROOM_SQUARE / 2, (c.shape[0] - 1) * ROOM_SQUARE / 2, 0.0]) + p['t']
                  for p, c in zip(poses, planes)])
    off = (C - C.mean(0)) @ nbar
    tilt = np.arccos(np.clip(np.abs(N @ nbar), 0, 1))
    return float(np.sqrt(np.mean(off ** 2) + np.mean((0.05 * tilt) ** 2)))


def refine_focal(planes, f0, size, dist, fit_k1=False):
    """Focal length [px] (and, with fit_k1, the radial distortion k1) that makes three or more boards
    coplanar, and whether it is trustworthy: the cost must rise clearly on both sides (a camera
    looking straight down cannot tell). Returns (f, ok, cost, dist)."""
    from scipy.optimize import minimize, minimize_scalar
    dist = np.array(dist, float)
    r = minimize_scalar(lambda f: coplanar_cost(planes, f, size, dist), bounds=(0.4 * f0, 2.5 * f0), method='bounded',
                        options=dict(xatol=0.05))
    f, c0 = float(r.x), float(r.fun)
    if fit_k1:
        def cost(q):
            d = dist.copy()
            d[0] = q[1]
            return coplanar_cost(planes, q[0], size, d)
        r2 = minimize(cost, [f, dist[0]], method='Nelder-Mead',
                      options=dict(initial_simplex=[[f, dist[0]], [f * 1.03, dist[0]], [f, dist[0] - 0.05]], xatol=0.05, fatol=1e-7,
                                   maxiter=120))
        if r2.fun < c0 and abs(r2.x[1]) < 0.5:
            f, c0, dist[0] = float(r2.x[0]), float(r2.fun), float(r2.x[1])
    lo, hi = coplanar_cost(planes, 0.85 * f, size, dist), coplanar_cost(planes, 1.15 * f, size, dist)
    ok = min(lo, hi) > max(4 * c0, 0.002) and 0.45 * f0 < f < 2.4 * f0
    return f, ok, c0, dist


def scaled_calib(calib, size):
    """(K, dist) of a stored calibration for frames of another size with the same shape (the
    distortion coefficients live in normalised coordinates, so they carry over)."""
    s = size[0] / calib.width
    if abs(size[1] / calib.height - s) > 0.01 * s:
        raise SystemExit(f'calibration #{calib.id} is for {calib.width}x{calib.height}, frames are {size[0]}x{size[1]}: '
                         f'a different shape; leave out --calib')
    K = np.array([[calib.fx * s, 0, (calib.cx + 0.5) * s - 0.5], [0, calib.fy * s, (calib.cy + 0.5) * s - 0.5], [0, 0, 1.0]])
    return K, calib.dist


class SalientNet:
    """Object masks for anything (BiRefNet, dichotomous segmentation): no class list, so objects the
    detector never saw are cut out as well; the table needs no pattern around the object."""
    NAME = 'ZhengPeng7/BiRefNet'

    def __init__(self):
        torch = _torch()
        from transformers import AutoModelForImageSegmentation
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'loading segmentation network on {self.dev} ...')
        self.model = AutoModelForImageSegmentation.from_pretrained(self.NAME, trust_remote_code=True).eval().to(self.dev)
        self.half = self.dev == 'cuda'
        if self.half:
            self.model = self.model.half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, bgr, size=1024):
        """Foreground probability per pixel (float32, frame size)."""
        torch = _torch()
        F = torch.nn.functional
        x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        x = (F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False) - self.mean) / self.std
        with torch.no_grad():
            y = self.model(x.half() if self.half else x)[-1].sigmoid().float()
        return F.interpolate(y, size=bgr.shape[:2], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


_SHARED, _GPU = {}, threading.Lock()


def shared(cls):
    """One copy of a network for every room (a rig with two cameras has two), called one at a time."""
    with _GPU:
        if cls not in _SHARED:
            _SHARED[cls] = cls()
        net = _SHARED[cls]

    def call(*a, **k):
        with _GPU:
            return net(*a, **k)
    return call


def fallback_calib(kind):
    """Newest stored calibration for this kind of camera, or None: what table mode starts from until the
    boards give a better f. The ESP32 streams 800 px wide or less, a USB phone 1280 or more; the phone's browser
    stream has its own field of view, so only a calibration made from that stream (meta source) counts."""
    rows = db().execute("SELECT id, meta FROM calibrations WHERE method IN ('board', 'phone', 'ruler', 'room') "
                        "AND rms IS NOT NULL AND rms < 1.0 AND fx < 5 * width ORDER BY id DESC").fetchall()
    for i, meta in rows:
        c = load_calib(i)
        src = json.loads(meta or '{}').get('source')
        if kind == 'webphone':
            if src == 'webphone':
                return c
        elif src != 'webphone' and (kind == 'esp32') == (c.width <= 1024) and c.method != 'room':
            return c
    return None


class Room:
    """Checkerboard room state across frames: intrinsics (self-calibrated once from the panels,
    or a stored calibration), last planes (search hints), and the measured objects."""

    def __init__(self, args, size, calib=None, store=True):
        self.args, self.size, self.store = args, size, store
        self.K = self.dist = None
        self.calib_id, self.calib_note, self.prev, self.n = None, '', [], 0
        self.floor_pick, self.floor_n, self.shape = 0, None, args.shape
        self.namer, self.pool = None, []
        self.want_calib, self.fits = False, []
        self.tracks = []  # each object's views from different camera positions
        self.noise = 1.0  # this camera's outline noise [px]: what its outlines leave over after the fit (running mean)
        self.corrector = None if getattr(args, 'corrector', False) else False  # opt-in; loaded on first use
        self.sizer = None if getattr(args, 'sizer', False) else False  # opt-in, like the corrector
        self.outline_net = None if getattr(args, 'learned_outline', False) else False
        self.want_inputs = False  # keep the outline network's inputs with each object (for labels)
        self.table = bool(getattr(args, 'table', False))  # boards scattered on a table, objects anywhere on it
        self.salient, self.f_hist, self.K0, self.dist0 = None, [], None, None
        if calib is not None:
            if (calib.width, calib.height) == tuple(size):
                self.K, self.dist = calib.K, calib.dist
            else:  # same shape, other size (the phone's 1600 x 1200 measured at 1000 x 750)
                self.K, self.dist = scaled_calib(calib, size)
            self.calib_id = calib.id
            self.calib_note = f'calib #{calib.id} ({calib.method})' + ('' if (calib.width, calib.height) == tuple(size)
                                                                      else f' scaled to {size[0]}x{size[1]}')
            if self.table and getattr(args, 'refine_f', False):
                self.K0, self.dist0, self.K = self.K, self.dist, None  # f refined from the boards, distortion kept

    def calibrate(self, planes):
        """Self-calibrate from this frame if it shows all three panel orientations sharply."""
        c = calib_from_planes(planes, self.size)
        if c is None or c['n_orient'] < 3 or c['rms'] > 1.0:
            self.calib_wait = ('need the floor and both walls in view' if c is None or c['n_orient'] < 3 else
                               f"frame not sharp enough (rms {c['rms']:.2f} px > 1): hold the camera still")
            return False
        self.K, self.dist = c['K'], c['dist']
        self.calib_note = (f"self-calibrated: f {c['K'][0, 0]:.0f} px, rms {c['rms']:.2f} px, "
                           f"{c['n_planes']} planes / {c['n_orient']} orientations")
        if self.store:
            self.calib_id = save_calib('room', new_session('room'), self.size, c['K'], c['dist'], c['rms'],
                                       dict(n_planes=c['n_planes'], n_orient=c['n_orient'], square=ROOM_SQUARE))
            self.calib_note += f' (saved #{self.calib_id})'
        print(self.calib_note)
        return True

    def pool_sizes(self, objs):
        """Table mode has no fixed frame to pool views in, so each frame's fit stands alone; the size shown
        is the median of the last 12 fits of the same object (same name, sizes within about 30 %). The
        camera moving changes the single-view bias, so the median also averages that."""
        for o in objs:
            v = np.log([max(o['L'], 1e-3), max(o['W'], 1e-3), max(o['H'], 1e-3)])
            nm = (o.get('name') or {}).get('name')
            e = next((e for e in self.pool if e['name'] == nm and np.abs(v - np.median(e['v'], 0)).max() < 0.3), None)
            if e is None:
                e = dict(name=nm, v=[])
                self.pool.append(e)
            e['v'] = (e['v'] + [v])[-12:]
            if len(e['v']) > 1:
                med = np.exp(np.median(e['v'], 0))
                o['single'] = (o['L'], o['W'], o['H'])
                o['L'], o['W'], o['H'] = (float(m) for m in med)
                o['pooled'] = len(e['v'])
        self.pool = self.pool[-8:]

    def table_calibrate(self, planes):
        """f from boards lying on one table (coplanarity), median over frames; distortion from the stored
        calibration if one was given (--calib with --refine-f), else none."""
        boards = [c for c in planes if c.shape[0] * c.shape[1] >= 30]
        dist = self.dist0 if self.dist0 is not None else np.zeros(5)
        f0 = float(self.K0[0, 0]) if self.K0 is not None else 0.9 * max(self.size)
        if len(boards) >= 3:
            try:
                f, ok, c0, d = refine_focal(boards, f0, self.size, dist, fit_k1=self.dist0 is None)
            except (cv2.error, np.linalg.LinAlgError, ValueError):
                ok = False
            if ok:
                self.f_hist = (self.f_hist + [f])[-15:]
                self.k1_hist = (getattr(self, 'k1_hist', []) + [float(d[0])])[-15:]
        if len(self.f_hist) < 3:
            self.calib_wait = ('lay three or more boards on the table and look at them from an angle (not straight down)'
                               if len(boards) < 3 or not self.f_hist else 'holding still for the focal length')
            return False
        f = float(np.median(self.f_hist))
        self.K = np.array([[f, 0, (self.size[0] - 1) / 2], [0, f, (self.size[1] - 1) / 2], [0, 0, 1.0]])
        self.dist = dist.copy()
        if self.dist0 is None:
            self.dist[0] = float(np.median(self.k1_hist))
        self.calib_note = f'f {f:.0f} px' + (f', k1 {self.dist[0]:.3f}' if self.dist0 is None else '') + f' from {len(boards)} coplanar boards ({len(self.f_hist)} frames, spread ' \
                          f'{100 * np.std(self.f_hist) / f:.1f} %)'
        return True

    def guided(self, img, pose, cands, prior):
        """Outlines the other camera's objects predict in this picture, where this camera's own segmentation found no
        match: the fitted shape is projected with this camera's pose, and GrabCut, started from the projection (inside
        it the object, far outside it the room), finds where the image itself puts the edge. An outline that then
        disagrees with the prediction is dropped (nothing seen there), so a view only counts when the image confirms it."""
        Hh, Ww = img.shape[:2]
        for pr in prior:
            pts = shape_points(pr['p'], pr['shape'], pr['up'])
            if ((pose['R'] @ pts.T).T + pose['t'])[:, 2].min() <= 0:
                continue  # part of it behind the camera
            uv = cv2.projectPoints(pts, pose['rvec'], pose['tvec'], self.K, self.dist)[0].reshape(-1, 2)
            if not np.isfinite(uv).all() or np.abs(uv).max() > 1e5:
                continue
            pred = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            cx, cy = pred.mean(0)
            if cv2.contourArea(pred) < 30 or not (0 < cx < Ww and 0 < cy < Hh):
                continue
            near = [c for c in cands if hull_iou(pred, c[0]) > 0.05]
            if any(hull_iou(pred, c[0]) >= 0.5 for c in near):
                continue  # this camera found it itself
            x0, y0 = np.floor(pred.min(0)).astype(int)
            x1, y1 = np.ceil(pred.max(0)).astype(int)
            m = int(max(10, 0.35 * max(x1 - x0, y1 - y0)))
            x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(Ww, x1 + m), min(Hh, y1 + m)
            rect = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            sl = (slice(rect[1], rect[1] + rect[3]), slice(rect[0], rect[0] + rect[2]))
            pm = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(pm, [pred.astype(np.int32)], 1)
            pm = pm[sl]
            k = lambda f: np.ones((max(3, int(f * min(np.ptp(pred[:, 0]), np.ptp(pred[:, 1]))) | 1),) * 2, np.uint8)
            sure_fg = cv2.erode(pm, k(0.3)) > 0
            far = cv2.dilate(pm, k(0.6)) == 0
            cnt = grabcut_seeded(img, pm > 0, far, rect, sure_fg=sure_fg if sure_fg.any() else None)
            if cnt is None:
                continue
            iou = hull_iou(pred, cnt)
            if iou < 0.6:
                continue
            cands[:] = [c for c in cands if not any(c is n for n in near)]  # junk of its own at the same place gives way to the confirmed outline
            cands.append((cnt, 'object', iou, [int(x0), int(y0), int(x1), int(y1)], dict(guided=True)))

    def process(self, img, det, partner=(), prior=()):
        """Planes, floor pose and every object on the floor in one frame. `partner`: (hull, pose with its own K and
        dist, centre) of objects the other camera of a rig sees at the same time, in the same room coordinates.
        `prior`: the other camera's fitted objects (see rig_priors): where this camera looks when its own segmentation fails."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.n += 1
        full = not self.prev or self.n % 15 == 0
        st = {}
        planes = find_planes(gray, self.prev, full, stats=st)
        if not full and st['lost']:
            planes = find_planes(gray)  # a panel shrank or vanished: look everywhere
        self.prev = planes
        out = dict(planes=planes, poses=[], groups=[], floor=None, objects=[], angles=[], occ=None)
        if self.want_calib and planes:
            self.want_calib = False
            if not self.calibrate(planes):
                print('not recalibrated: ' + self.calib_wait)
        if self.table and planes and (self.K is None or (self.K0 is not None or self.calib_id is None) and self.n % 5 == 0):
            if not self.table_calibrate(planes) and self.K is None:
                fb = getattr(self, 'fallback', None)
                if fb is None:
                    return out
                try:  # not enough boards at an angle for f (e.g. straight above): the camera's stored calibration
                    self.K, self.dist = scaled_calib(fb, self.size)
                    self.calib_note = f'calib #{fb.id} ({fb.method}) as fallback: view the boards from an angle for f from the boards'
                except SystemExit:
                    return out
        elif self.K is None and planes and not self.calibrate(planes):
            return out
        if self.K is None or not planes:
            return out
        poses = plane_poses(planes, self.K, self.dist)
        groups = plane_groups([p['n'] for p in poses])
        out.update(poses=poses, groups=groups)
        # floor: the orientation closest to last frame's floor; at the start (and after f) the one
        # lying lowest in the image, f cycling through the others
        gids = sorted(set(groups), key=lambda g: -np.mean([planes[i][..., 1].mean() for i in range(len(planes))
                                                           if groups[i] == g]))
        normal = {g: poses[groups.index(g)]['n'] for g in gids}
        if self.floor_n is None:
            fg = gids[0]
        else:
            fg = max(gids, key=lambda g: abs(float(normal[g] @ self.floor_n)))
            if self.floor_pick:  # f: the next orientation becomes the floor
                fg, self.floor_pick = gids[(gids.index(fg) + 1) % len(gids)], 0
            elif abs(float(normal[fg] @ self.floor_n)) < math.cos(math.radians(25)):
                return out  # floor not in view this frame
        self.floor_n = normal[fg]
        merged = {g: merge_pieces(planes, poses, [i for i in range(len(planes)) if groups[i] == g], self.K, self.dist,
                                  max_rms=0.004 if self.table else 0.002)
                  for g in gids}
        gposes = {g: m[0] for g, m in merged.items()}
        pose, n_used = merged[fg]
        fixed = corner_frame(gposes, fg)
        if fixed is not None:  # room corner coordinates: the same in every frame, so views can be pooled
            pose = dict(pose, **fixed)
        pose = dict(pose, sigma=float(self.noise))  # the views of this camera count by its outline noise
        cam = -pose['R'].T @ pose['t']  # camera centre in floor coordinates
        up = 1.0 if cam[2] >= 0 else -1.0
        out.update(floor=pose, floor_pieces=n_used, floor_group=fg, cam_height=float(abs(cam[2])))
        for a in range(len(gids)):
            for b in range(a + 1, len(gids)):
                ang = math.degrees(math.acos(min(1.0, abs(float(normal[gids[a]] @ normal[gids[b]])))))
                out['angles'].append((gids[a] == fg or gids[b] == fg, ang))
        Hh, Ww = gray.shape
        cands = []  # (outline, label, score, box)
        if self.args.seg == 'room':
            flat = shade_free(gray, planes)  # shadows removed: the pattern tests run on this
            occ, free, per, step, levels = room_occupancy(flat, planes, gposes, self.K, self.dist)
            out['occ'] = occ
            for blob, (x, y, w, h) in room_blobs(occ, per.get(fg, np.zeros_like(occ)), step):
                mx, my = int(0.25 * w + step), int(0.25 * h + step)
                x0, y0 = max(0, x - mx), max(0, y - my)
                rect = (x0, y0, min(Ww, x + w + mx) - x0, min(Hh, y + h + my) - y0)
                # pixels off the pattern inside the blob (it holds every part-covered cell) are the
                # object for sure; black squares still black are the room for sure. GrabCut settles
                # the rest (the object's own colour, and the joints, which are not judged); the
                # outline is the hull of both.
                sl = (slice(rect[1], rect[1] + rect[3]), slice(rect[0], rect[0] + rect[2]))
                maps = {}
                bad, room_b, room_w, edges = pattern_mismatch(flat, rect, gposes, levels, self.K, self.dist, maps=maps)
                bad = cv2.morphologyEx((bad & blob[sl]).astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((2, 2), np.uint8)) > 0
                if bad.sum() < 20:
                    continue
                # a light object hides on white squares, a dark one on black: only squares of the other
                # colour that still show it are the room for sure
                gl = levels[fg]
                dark_obj = np.median(flat[sl][bad]) < (gl[0] + gl[1]) / 2
                sure_room = room_w if dark_obj else room_b
                near = cv2.dilate(blob[sl].astype(np.uint8), np.ones((int(step) | 1,) * 2, np.uint8)) > 0
                pts = [np.stack(np.nonzero(bad | (edges & near))[::-1], 1).astype(np.float32) + np.float32(rect[:2])]
                cnt = grabcut_seeded(img, blob[sl], sure_room | ((free[sl] > 0) & ~blob[sl]), rect, sure_fg=bad)
                if cnt is not None:
                    pts.append(cnt)
                extra = {}
                if self.want_inputs or self.outline_net:
                    cur = np.zeros((rect[3], rect[2]), np.uint8)
                    cv2.fillPoly(cur, [cv2.convexHull(np.vstack(pts)).astype(np.int32) - np.int32(rect[:2])], 1)
                    X = outline_inputs(img[sl], flat[sl], occ[sl] > 0, maps['pat'], maps['border'], cur)
                    extra = dict(inputs=X, rect=rect)
                    if self.outline_net:  # the learned outline replaces the hand-made one
                        prob = cv2.resize(self.outline_net[0](X), (rect[2], rect[3]), interpolation=cv2.INTER_LINEAR)
                        m = (prob > 0.5).astype(np.uint8)
                        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cs and cv2.contourArea(max(cs, key=cv2.contourArea)) > 50:
                            pts = [max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) + np.float32(rect[:2])]
                            extra['learned'] = True
                cands.append((np.vstack(pts), 'object', 1.0, [x, y, x + w, y + h], extra))
        elif self.args.seg == 'salient':
            if self.salient is None:
                self.salient = shared(SalientNet)
            prob = self.salient(img)
            m = cv2.morphologyEx((prob > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n_c, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            lat = [cv2.convexHull(c.reshape(-1, 2)).astype(np.int32) for c in planes]
            walls = [lat[i] for i in range(len(planes)) if groups[i] != fg]
            for i in range(1, n_c):
                if stats[i, 4] < 0.0015 * Ww * Hh:
                    continue
                comp = (lab == i).astype(np.uint8)
                is_board = False
                for hb in lat:  # a board that came out salient itself: its mask is the board
                    bm = np.zeros_like(comp)
                    cv2.fillPoly(bm, [hb], 1)
                    if (comp & bm).sum() / max(float((comp | bm).sum()), 1.0) > 0.5:
                        is_board = True
                        break
                x, y, w, h = [int(v) for v in stats[i, :4]]
                if is_board or any(cv2.pointPolygonTest(wl, (x + w / 2, y + h / 2), False) > 0 for wl in walls):
                    continue
                cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cands.append((max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32), 'object',
                              float(prob[comp > 0].mean()), [x, y, x + w, y + h], {}))
        elif self.args.seg == 'detect' and det is not None:
            walls = [cv2.convexHull(planes[i].reshape(-1, 2)) for i in range(len(planes)) if groups[i] != fg]
            for d in det(img):
                x0, y0, x1, y1 = d['box']
                if (x1 - x0) * (y1 - y0) > 0.5 * Ww * Hh:
                    continue
                if self.args.object_class and d['label'] != self.args.object_class:
                    continue
                if any(cv2.pointPolygonTest(w, ((x0 + x1) / 2, (y0 + y1) / 2), False) > 0 for w in walls):
                    continue  # a pattern on a wall, not an object on the floor
                cnt = grabcut_contour(img, d['box'])
                if cnt is not None:
                    cands.append((cnt, d['label'], d['score'], d['box'], {}))
        if prior and fixed is not None:
            self.guided(img, pose, cands, prior)
        if self.outline_net is None:  # --learned-outline: load on first use
            got = load_outline_net()
            if got is None or not got[1]:
                print('no accepted outline network: using the built-in outline (run room-labels, train-outline)')
                self.outline_net = False
            else:
                self.outline_net = got
        for cnt, label, score, box, extra in cands:
            hull = cv2.convexHull(cnt).reshape(-1, 2)
            nm = None
            if self.shape == 'auto':  # the name says which form to fit; CLIP looks at the object cut out of the frame
                if self.namer is None:
                    try:
                        self.namer = shared(ObjectNamer)
                    except (Exception, SystemExit) as e:  # no name model: the generic shape, the work goes on
                        print(f'object namer unavailable ({e}); fitting the generic shape')
                        self.namer = lambda im, hl: None
                nm = self.namer(img, hull)
            shapes = [shape_for(nm) if self.shape == 'auto' else self.shape]
            near_of = lambda sh: [f['cand_ps'][sh] for f in self.fits if sh in f.get('cand_ps', {})]
            vpose = dict(pose, sigma_min=2.5, guided=True) if extra.get('guided') else pose  # grown from a prediction: trusted less
            base = fit_shape(hull, vpose, self.K, self.dist, up, shapes[0], near_of(shapes[0]))
            if base is None:
                continue
            n_views, others, tr = 1, [], None
            if fixed is not None:  # views of the same object (same place) from camera positions 3 cm apart
                tr = next((t for t in self.tracks if math.dist(t['p'][:2], base['p'][:2]) < 0.03), None)
                if tr is None:
                    tr = dict(p=base['p'], views=[], ps={})
                    self.tracks.append(tr)
                cut_now = bool((hull[:, 0] < 2).any() or (hull[:, 1] < 2).any() or (hull[:, 0] > Ww - 3).any()
                               or (hull[:, 1] > Hh - 3).any())
                if not cut_now and all(np.linalg.norm(cam - c) > 0.03 for _, _, c in tr['views']):
                    tr['views'].append((hull, vpose, cam))
                    if len(tr['views']) > 8:  # drop the view closest to another one: keep the spread
                        cs = np.array([c for _, _, c in tr['views']])
                        dd = np.linalg.norm(cs[:, None] - cs[None], axis=2) + np.eye(len(cs))
                        tr['views'].pop(int(np.argmin(dd.min(1))))
                others = [(h, ps) for h, ps, c in tr['views'] if np.linalg.norm(cam - c) > 0.03]
                others += [(h, ps) for h, ps, c in partner if math.dist(c, base['p'][:2]) < 0.08]
                n_views = 1 + len(others)
            all_views = [(hull, vpose)] + others
            fit_views, fit_up = all_views, up
            tried = {}
            for sh in shapes:
                warm = near_of(sh) + ([tr['ps'][sh]] if tr is not None and sh in tr.get('ps', {}) else [])
                f = base if sh == shapes[0] and not others else \
                    fit_shape(hull, vpose, self.K, self.dist, up, sh, warm or [base['p']], others)
                if f is None:
                    continue
                if others and tr is not None:  # a stored view that disagrees (hand in frame, object moved,
                    iou_v = view_ious(f['p'], sh, up, [(hull, vpose)] + others, self.K, self.dist)  # blur): drop it
                    bad = [j for j in range(1, len(iou_v)) if iou_v[j] < max(0.8, np.median(iou_v) - 0.08)]
                    if bad:
                        drop = {id(others[j - 1][0]) for j in bad}
                        tr['views'] = [v for v in tr['views'] if id(v[0]) not in drop]
                        others = [v for v in others if id(v[0]) not in drop]
                        all_views = [(hull, vpose)] + others
                        fit_views, n_views = all_views, 1 + len(others)
                        f = fit_shape(hull, vpose, self.K, self.dist, up, sh, [f['p']], others) or f
                try:
                    sig = []
                    f['info'] = size_information(f['p'], sh, up, all_views, self.K, self.dist, sigma_px=None, report=sig)
                    if sig and sh == shapes[0] and not extra.get('guided'):
                        self.noise = float(np.clip(0.8 * self.noise + 0.2 * sig[0], 1.0, 8.0))
                    f['sigmas'] = sig
                except (cv2.error, np.linalg.LinAlgError, ValueError):
                    f['info'] = None
                f['shape'] = sh
                tried[sh] = f
            if not tried:
                continue
            fit = tried[shapes[0]]
            fit['cand_ps'] = {sh: f['p'] for sh, f in tried.items()}
            fit['views'], fit['up'], fit['track'] = fit_views, fit_up, tr
            fit['n_views'] = n_views
            if tr is not None:
                tr['p'], tr['ps'] = fit['p'], fit['cand_ps']
            fit.update(extra)
            fit['name'] = nm
            fit.update(label=label, score=score, box=box, hull=hull, cam=cam, fixed=fixed is not None,
                       cut=bool((hull[:, 0] < 2).any() or (hull[:, 1] < 2).any() or (hull[:, 0] > Ww - 3).any()
                                or (hull[:, 1] > Hh - 3).any()))
            fit['uv'] = cv2.projectPoints(shape_points(fit['p'], fit['shape'], up), pose['rvec'], pose['tvec'],
                                          self.K, self.dist)[0].reshape(-1, 2)
            if max(fit['L'], fit['W'], fit['H']) > 0.8 or not np.isfinite(fit['uv']).all() \
                    or np.abs(fit['uv']).max() > 1e5:
                continue  # a runaway fit (an outline the shape cannot explain), not an object in the room
            out['objects'].append(fit)
        if self.table:
            self.pool_sizes(out['objects'])
        if self.corrector is None:
            self.corrector = load_room_corrector() or False
        if self.corrector:
            apply, acc = self.corrector
            for o in out['objects']:
                mu, sd = apply(dict(o, centre=o['p'][:2]))
                if not np.isfinite(sd).all():
                    continue
                o['geo'] = (o['L'], o['W'], o['H'])
                for i, d in enumerate('LWH'):
                    if acc.get(d):
                        o[d] += float(mu[i])
                        o['sd_' + d] = float(sd[i])
        if self.sizer is None:
            self.sizer = load_sizer() or False
            if not self.sizer:
                print('no sizer trained yet: run `sizer-features` and `train-sizer`')
        if self.sizer:
            apply, acc = self.sizer
            for o in out['objects']:
                tr = o.get('track') if o.get('track') is not None else {}
                try:
                    x, ps = sizer_features(o['views'], self.K, self.dist, o['up'], tr.get('sizer_ps'))
                except (cv2.error, np.linalg.LinAlgError, ValueError):
                    continue
                if x is None:
                    continue
                if isinstance(tr, dict):
                    tr['sizer_ps'] = ps
                got = apply(x)
                if got is None:
                    continue
                o.setdefault('geo', (o['L'], o['W'], o['H']))
                for i, d in enumerate('LWH'):
                    if acc.get(d):
                        o[d] = float(got[0][i])
                        o['sd_' + d] = float(got[1][i])
                o['sized'] = True
        self.fits = out['objects']
        return out

    def draw(self, img, res, show_occ=False, sel=None, has_sel=False):
        disp = img.copy()
        if show_occ and res.get('occ') is not None and res['occ'].shape == disp.shape[:2]:
            o = res['occ'] > 0
            disp[o] = (0.5 * disp[o] + np.array([0, 0, 127])).astype(np.uint8)
        cols = [(80, 220, 80), (255, 160, 60), (60, 160, 255), (220, 80, 220), (80, 220, 220)]
        for i, c in enumerate(res['planes']):
            g = res['groups'][i] if res['groups'] else i
            col = (80, 255, 80) if g == res.get('floor_group') else cols[1 + g % 4]
            for u, v in c.reshape(-1, 2):
                cv2.circle(disp, (int(u), int(v)), 1, col, -1)
        for oi, o in enumerate(res['objects']):
            if has_sel and oi != sel:  # another object is selected: this one only as a faint outline
                cv2.polylines(disp, [o['hull'].astype(np.int32)], True, (150, 150, 150), 1, cv2.LINE_AA)
                continue
            if has_sel:
                tint = disp.copy()
                cv2.fillPoly(tint, [o['hull'].astype(np.int32)], (0, 220, 255))
                disp = cv2.addWeighted(tint, 0.25, disp, 0.75, 0)
                cv2.polylines(disp, [o['hull'].astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
            uv = o['uv']
            if ROOM_SHAPES[o['shape']] is None:
                for a, b in BOX_EDGES:
                    cv2.line(disp, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.polylines(disp, [cv2.convexHull(uv.astype(np.float32)).astype(np.int32)], True, (0, 0, 255), 1,
                              cv2.LINE_AA)
                cv2.polylines(disp, [uv[:32].astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.polylines(disp, [o['hull'].astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
            x0, y0 = int(o['box'][0]), int(o['box'][1])
            lines = [f"{o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x {o['H'] * 1e3:.0f} mm" + ('  (cut off)' if o['cut'] else '')
                     + ('  sizer' if o.get('sized') else '  corrected' if 'geo' in o else '')
                     + ('  learned outline' if o.get('learned') else '')
                     + (f"  median of {o['pooled']}" if o.get('pooled') else ''),
                     f"{(o.get('name') or {}).get('general') or o['label']} [{o['shape']}]  fit {o['iou']:.2f}  {o.get('n_views', 1)} view(s)"
                     + ('  diameter ' + ' / '.join(f'{v * 1e3:.0f}' for v in o['profile']) + ' mm up'
                        if o.get('profile') else '')]
            if o.get('info'):
                sd = [o['info'][d][1] * 1e3 for d in 'LWH']
                lines.insert(1, '+- ' + ' / '.join(f'{v:.0f}' if v < 999 else '?' for v in sd) + ' mm'
                             + ('   H weak: view from lower or the side' if sd[2] > max(5.0, 0.1 * o['H'] * 1e3) else ''))
            if self.args.truth:
                tl, tw, th = self.args.truth
                lines.append(f"err {(o['L'] - tl) * 1e3:+.0f} {(o['W'] - tw) * 1e3:+.0f} {(o['H'] - th) * 1e3:+.0f} mm")
            draw_text(disp, lines, org=(x0, max(18, y0 - 8 - 18 * (len(lines) - 1))), scale=0.5, color=(0, 255, 255))
        return disp

    def status(self, res):
        if self.K is None:
            return ['calibrating: ' + getattr(self, 'calib_wait', 'show the floor and both walls')]
        s = [self.calib_note + f'   shape {self.shape} (s)']
        if res['floor'] is None:
            return s + [f"{len(res['planes'])} planes, no floor"]
        fw = [a for f, a in res['angles'] if f]
        ww = [a for f, a in res['angles'] if not f]
        s.append(f"{len(res['planes'])} planes   camera {res['cam_height'] * 1e3:.0f} mm above floor   "
                 f"floor {res['floor_pieces']} piece(s), rms {res['floor']['rms']:.2f} px   angles: floor-wall "
                 + '/'.join(f'{a:.1f}' for a in fw) + ('  wall-wall ' + '/'.join(f'{a:.1f}' for a in ww) if ww else ''))
        return s


def room_record(room, res):
    return dict(objects=[dict(label=o['label'], score=o['score'], **dict(zip('LWH', o.get('geo', (o['L'], o['W'], o['H'])))),
                              corrected=[o['L'], o['W'], o['H']] if 'geo' in o else None, iou=o['iou'],
                              n_views=o.get('n_views', 1), centre=o['p'][:2], cam=o['cam'], profile=o.get('profile'),
                              info=o.get('info'), up=o.get('up', 1.0), sigmas=o.get('sigmas'), guided=bool(o.get('guided')),
                              views=[dict(hull=np.round(h, 1).tolist(), rvec=np.ravel(ps['rvec']).tolist(),
                                          tvec=np.ravel(ps['tvec']).tolist(),
                                          **({} if 'K' not in ps else dict(K=np.asarray(ps['K']).tolist(),
                                                                          dist=np.ravel(ps['dist']).tolist())))
                                     for h, ps in o.get('views', [])],
                              shape=o['shape'], cut=o['cut'], box=o['box']) for o in res['objects']],
                cam_height=res.get('cam_height'), angles=res['angles'], n_planes=len(res['planes']),
                K=None if room.K is None else np.asarray(room.K).tolist(),
                dist=None if room.dist is None else np.ravel(room.dist).tolist())


OBJECT_VOCAB = {  # category -> names CLIP chooses from; the category is the generalised answer
    'computer accessory': ['computer mouse', 'computer keyboard', 'USB flash drive', 'external hard drive', 'webcam',
                           'USB cable', 'power adapter', 'phone charger', 'memory card', 'portable SSD', 'portable hard disk', 'USB hub', 'card reader', 'microphone', 'handheld microphone', 'bluetooth speaker', 'headphones', 'earbuds',
                           'wireless earbuds case', 'game controller', 'router', 'laptop', 'tablet', 'speaker'],
    'electronic board': ['circuit board', 'microcontroller board', 'development board', 'ESP32 camera module',
                         'breadboard', 'camera module', 'sensor module', 'battery pack', 'power bank', 'lithium battery',
                         'electronic component', 'electrical outlet', 'power strip', 'remote control', 'MP3 player'],
    'phone or camera': ['smartphone', 'mobile phone', 'camera', 'action camera', 'calculator', 'smartwatch', 'walkie talkie'],
    'drink container': ['cup', 'coffee mug', 'drinking glass', 'water bottle', 'soda can', 'beverage can', 'thermos',
                        'tea cup', 'paper cup', 'plastic cup', 'bowl', 'transparent bowl', 'jar', 'wine glass', 'tin can'],
    'food item': ['apple', 'orange', 'banana', 'lemon', 'egg', 'bread', 'chocolate bar', 'candy', 'snack package',
                  'box of cereal', 'tomato', 'potato', 'onion', 'cucumber', 'biscuit pack'],
    'kitchen item': ['spoon', 'fork', 'knife', 'plate', 'saucer', 'cutting board', 'teapot', 'kettle', 'pan', 'pot',
                     'salt shaker', 'sponge', 'bottle opener', 'scissors'],
    'stationery': ['pen', 'pencil', 'marker', 'eraser', 'ruler', 'notebook', 'sheet of paper', 'sticky notes', 'stapler',
                   'tape dispenser', 'roll of tape', 'paper clip', 'binder clip', 'glue stick', 'highlighter', 'sharpener',
                   'envelope', 'book', 'playing card', 'credit card', 'id card', 'business card'],
    'hand tool': ['screwdriver', 'pliers', 'wrench', 'hammer', 'tweezers', 'nail clippers', 'hair clip', 'clothespin', 'binder clip', 'padlock', 'coin', 'magnet', 'tape measure', 'utility knife',
                  'soldering iron', 'multimeter', 'allen key', 'drill bit', 'caliper', 'flashlight', 'hot glue gun'],
    'personal care item': ['toothbrush', 'toothbrush case', 'toothpaste tube', 'cream jar', 'lip balm', 'perfume bottle',
                           'soap bar', 'shampoo bottle', 'comb', 'hairbrush', 'razor', 'deodorant', 'sunglasses',
                           'glasses case', 'hand sanitizer bottle', 'tissue box', 'wallet', 'keys', 'lighter'],
    'toy or model': ['toy car', 'toy robot', 'toy figure', 'stuffed animal', 'ball', 'tennis ball', 'ping pong ball',
                     'dice', 'building block', 'model crane', 'toy truck', 'puzzle', 'rubber duck', 'balloon'],
    'storage item': ['tin box', 'storage tin', 'cosmetic jar', 'lotion bottle', 'toothbrush holder', 'cardboard box', 'plastic container', 'tupperware', 'pencil case', 'small box', 'gift box',
                     'shoe box', 'bag', 'pouch', 'tin box', 'basket', 'jar with lid', 'matchbox'],
    'household item': ['clock', 'alarm clock', 'candle', 'vase', 'picture frame', 'lamp', 'light bulb', 'coaster',
                       'remote', 'pillow', 'towel', 'cable spool', 'extension cord', 'toy', 'plant pot', 'small plant'],
    'clothing item': ['sock', 'glove', 'hat', 'shoe', 'scarf', 'watch', 'bracelet', 'belt'],
    'medical item': ['pill bottle', 'medicine box', 'syringe', 'bandage box', 'thermometer', 'mask'],
}


SHAPE_PRIOR = {  # what CLIP's name says about the form: the shape fitted to the outline
    'cylinder': ['cup', 'coffee mug', 'drinking glass', 'soda can', 'beverage can', 'tea cup', 'paper cup', 'plastic cup',
                 'tin can', 'roll of tape', 'cream jar', 'jar', 'jar with lid', 'tupperware', 'candle', 'plant pot', 'coaster',
                 'cable spool', 'paper clip', 'wine glass', 'bowl', 'transparent bowl', 'hand sanitizer bottle', 'tin box', 'storage tin', 'cosmetic jar', 'toothbrush holder', 'coin'],
    'revolve': ['microphone', 'handheld microphone', 'lotion bottle', 'water bottle', 'thermos', 'perfume bottle', 'shampoo bottle', 'vase', 'deodorant', 'toothpaste tube',
                'lip balm', 'pill bottle', 'light bulb', 'bottle opener', 'kettle', 'teapot', 'toothbrush case', 'flashlight'],
    'sphere': ['ball', 'tennis ball', 'ping pong ball', 'apple', 'orange', 'lemon', 'tomato', 'potato', 'onion', 'balloon'],
    'round': ['computer mouse', 'game controller', 'soap bar', 'egg', 'toy figure', 'stuffed animal', 'shoe', 'hat', 'sponge',
              'toy car', 'toy robot', 'rubber duck', 'model crane', 'toy truck', 'keys', 'lighter', 'banana', 'cucumber',
              'earbuds', 'wireless earbuds case', 'walkie talkie', 'action camera', 'glasses case', 'sunglasses', 'pouch',
              'bag', 'pillow', 'sock', 'glove', 'scissors', 'pliers', 'tweezers', 'hammer', 'drill bit'],
}
SHAPE_OF = {n: sh for sh, ns in SHAPE_PRIOR.items() for n in ns}  # every other name: a box


def shape_for(name):
    """Fitted shape for a CLIP name; unsure names (p < 0.3) get the generic rounded box."""
    if name is None or name['p'] < 0.30:
        return 'round'
    return SHAPE_OF.get(name['name'], 'box')


class ObjectNamer:
    """Open-vocabulary object name (CLIP ViT-L/14, zero-shot): any object, not only the 80 COCO
    classes. The object is cut out on a neutral background first (the checkerboard would otherwise
    drag every answer toward 'checkerboard'). Returns the fine name, its category and the runners-up;
    when the fine name is unsure the category is the generalised answer."""
    NAMES = ('openai/clip-vit-large-patch14', 'openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32')
    TEMPLATES = ('a photo of a {}.', 'a close-up photo of a {} on a table.', 'a {}, product photo.')

    def __init__(self):
        torch = _torch()
        from transformers import CLIPModel, CLIPProcessor
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        err = None
        for nm in self.NAMES:
            try:
                self.model = CLIPModel.from_pretrained(nm, torch_dtype=torch.float16 if self.dev == 'cuda' else torch.float32)
                self.proc, self.name = CLIPProcessor.from_pretrained(nm), nm
                break
            except Exception as e:  # not downloaded and no internet: try the next smaller one
                err = e
        else:
            raise SystemExit(f'no CLIP model available ({err})')
        print(f'object namer: {self.name} on {self.dev}')
        self.model = self.model.to(self.dev).eval()
        self.items = [(n, c) for c, ns in OBJECT_VOCAB.items() for n in dict.fromkeys(ns)]
        with torch.no_grad():
            embs = []
            for t in self.TEMPLATES:
                tok = self.proc(text=[t.format(n) for n, _ in self.items], return_tensors='pt', padding=True).to(self.dev)
                e = self.model.text_projection(self.model.text_model(**tok).pooler_output)
                embs.append(e / e.norm(dim=-1, keepdim=True))
            e = torch.stack(embs).mean(0)
            self.text = e / e.norm(dim=-1, keepdim=True)

    def __call__(self, bgr, hull, k=4):
        torch = _torch()
        H, W = bgr.shape[:2]
        hull = np.asarray(hull, np.float32).reshape(-1, 2)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [hull.astype(np.int32)], 255)
        x0, y0 = hull.min(0)
        x1, y1 = hull.max(0)
        s = 0.5 * max(x1 - x0, y1 - y0) * 1.15 + 4
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b, c, d = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
        img = np.where(m[b:d, a:c, None] > 0, bgr[b:d, a:c], np.uint8(118))
        pad = np.full((int(2 * s), int(2 * s), 3), 118, np.uint8)  # square, object centred
        oy, ox = int(b - (cy - s)), int(a - (cx - s))
        pad[max(0, oy):max(0, oy) + img.shape[0], max(0, ox):max(0, ox) + img.shape[1]] = \
            img[:pad.shape[0] - max(0, oy), :pad.shape[1] - max(0, ox)]
        px = self.proc(images=pad[..., ::-1].copy(), return_tensors='pt')['pixel_values'].to(self.dev, self.model.dtype)
        with torch.no_grad():
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            pr = (100.0 * e @ self.text.T).float().softmax(-1)[0].cpu().numpy()
        order = np.argsort(-pr)
        cats = {}
        for (n, c), p in zip(self.items, pr):
            cats[c] = cats.get(c, 0.0) + float(p)
        cat = max(cats, key=cats.get)
        top = [(self.items[i][0], float(pr[i])) for i in order[:k]]
        general = top[0][0] if top[0][1] >= 0.30 else cat
        return dict(name=top[0][0], p=top[0][1], top=top, category=cat, p_category=cats[cat], general=general,
                    categories=sorted(cats.items(), key=lambda kv: -kv[1])[:3])


def mesh_parts(shape):
    """Surface grids (rows x cols x 3, unit shape of length, width and height 1 standing on z = 0)."""
    e = ROOM_SHAPES[shape]
    th = np.linspace(0, 2 * np.pi, 25)
    parts = []
    if e is None:  # box: six faces, 4 x 4 cells each
        g, z = np.linspace(-0.5, 0.5, 5), np.linspace(0, 1, 5)
        zz, gg = np.meshgrid(z, g, indexing='ij')
        a, b = np.meshgrid(g, g)
        h = np.full_like(gg, 0.5)
        parts = [np.stack([gg, -h, zz], -1), np.stack([gg, h, zz], -1), np.stack([-h, gg, zz], -1),
                 np.stack([h, gg, zz], -1), np.stack([a, b, np.zeros_like(a)], -1), np.stack([a, b, np.ones_like(a)], -1)]
    elif e == 'cyl':
        z = np.linspace(0, 1, 6)
        T, Z = np.meshgrid(th, z)
        parts.append(np.stack([0.5 * np.cos(T), 0.5 * np.sin(T), Z], -1))
        for zc in (0.0, 1.0):
            S, T = np.meshgrid(np.linspace(0, 1, 4), th)
            parts.append(np.stack([0.5 * S * np.cos(T), 0.5 * S * np.sin(T), np.full_like(S, zc)], -1))
    elif e == 'sph':
        T, P = np.meshgrid(th, np.linspace(-np.pi / 2, np.pi / 2, 13))
        parts.append(np.stack([0.5 * np.cos(T) * np.cos(P), 0.5 * np.sin(T) * np.cos(P), 0.5 + 0.5 * np.sin(P)], -1))
    elif e != 'rev':  # superellipsoid, upper half
        sp = lambda x: np.sign(x) * np.abs(x) ** (2 / e)
        T, P = np.meshgrid(th, np.linspace(0, np.pi / 2, 10))
        parts.append(np.stack([0.5 * sp(np.cos(T)) * sp(np.cos(P)), 0.5 * sp(np.sin(T)) * sp(np.cos(P)), sp(np.sin(P))], -1))
        S, T = np.meshgrid(np.linspace(0, 1, 4), th)
        parts.append(np.stack([0.5 * S * sp(np.cos(T)), 0.5 * S * sp(np.sin(T)), np.zeros_like(S)], -1))
    return parts


def shape_mesh(p, shape, up):
    """Closed quad mesh of a fitted shape in floor coordinates: (vertices n x 3, quads k x 4 indices).
    The quads' normals point outward (the sign of the volume is then positive)."""
    p = tied(p, shape)
    if shape == 'revolve':
        d, Hh = revolve_profile(p), math.exp(p[5])
        zf = np.linspace(0, 1, 13)
        th = np.linspace(0, 2 * np.pi, 25)
        T, Z = np.meshgrid(th, zf)
        R = np.interp(Z, [0, 1 / 3, 2 / 3, 1], d) / 2
        parts = [np.stack([p[0] + R * np.cos(T), p[1] + R * np.sin(T), up * Z * Hh], -1)]
        for zc, r in ((0.0, d[0] / 2), (1.0, d[3] / 2)):
            S, T = np.meshgrid(np.linspace(0, 1, 4), th)
            parts.append(np.stack([p[0] + r * S * np.cos(T), p[1] + r * S * np.sin(T), np.full_like(S, up * zc * Hh)], -1))
        world = True
    else:
        parts, world = mesh_parts(shape), False
    L, W, H = np.exp(p[3:6])
    c, s = math.cos(p[2]), math.sin(p[2])
    V, Q = [], []
    for g in parts:
        r, k = g.shape[:2]
        pts = g.reshape(-1, 3)
        if not world:
            x, y = pts[:, 0] * L, pts[:, 1] * W
            pts = np.stack([p[0] + x * c - y * s, p[1] + x * s + y * c, up * pts[:, 2] * H], 1)
        off = sum(len(v) for v in V)
        V.append(pts)
        Q += [(off + i * k + j, off + i * k + j + 1, off + (i + 1) * k + j + 1, off + (i + 1) * k + j)
              for i in range(r - 1) for j in range(k - 1)]
    V, Q = np.vstack(V), np.array(Q)
    ctr = V.mean(0)
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    flip = np.einsum('ij,ij->i', n, V[Q].mean(1) - ctr) < 0
    Q[flip] = Q[flip][:, ::-1]
    return V, Q


def mesh_edges(Q):
    """Unique edges of a quad mesh (degenerate ones at the poles dropped) -> {(a, b): [faces]}."""
    ed = {}
    for f, q in enumerate(Q):
        for a, b in zip(q, np.roll(q, -1)):
            if a != b:
                ed.setdefault((min(a, b), max(a, b)), []).append(f)
    return ed


def mesh_volume_area(V, Q):
    tri = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    vol = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
    area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    return abs(vol), area


def mesh_segments(o, pose, K, dist):
    """The fitted wireframe of one object in the frame's pixels: (edges facing the camera, hidden edges), each an n x 4 array
    of x0, y0, x1, y1."""
    V, Q = shape_mesh(o['p'], o['shape'], o.get('up', 1.0))
    ed = mesh_edges(Q)
    cam = -pose['R'].T @ pose['t']
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    front = np.einsum('ij,ij->i', n, V[Q].mean(1) - cam) < 0  # outward normals: facing the camera
    uv = cv2.projectPoints(V, pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2)
    seen = [np.r_[uv[i], uv[j]] for (i, j), fs in ed.items() if any(front[f] for f in fs)]
    hid = [np.r_[uv[i], uv[j]] for (i, j), fs in ed.items() if not any(front[f] for f in fs)]
    return np.array(seen).reshape(-1, 4), np.array(hid).reshape(-1, 4)


def draw_mesh(img, res, K, dist):
    """The wireframe of every measured object on a picture, in place: visible edges bright, hidden ones thin and dim."""
    for o in res['objects']:
        vis, hid = mesh_segments(o, res['floor'], K, dist)
        for seg, col, th in ((hid, (89, 122, 255), 1), (vis, (176, 240, 0), 1)):
            for x0, y0, x1, y1 in np.round(seg).astype(int):
                cv2.line(img, (x0, y0), (x1, y1), col, th, cv2.LINE_AA)
    return img


def object_analysis(o, pose, K, dist, res=None):
    """Everything the report states about one fitted object (SI units, floor coordinates)."""
    shape, up = o['shape'], o.get('up', 1.0)
    p = tied(o['p'], shape)
    V, Q = shape_mesh(o['p'], shape, up)
    vol, area = mesh_volume_area(V, Q)
    L, W, H = o['L'], o['W'], o['H']
    cam = -pose['R'].T @ pose['t']
    ctr = np.array([p[0], p[1], 0.5 * H * up])
    d_cam = float(np.linalg.norm(cam - ctr))
    Zc = float((pose['R'] @ ctr + pose['t'])[2])
    uv = cv2.projectPoints(V, pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2).astype(np.float32)
    proj = cv2.convexHull(uv)
    hull = np.asarray(o['hull'], np.float32).reshape(-1, 2)
    res_px = np.array([abs(cv2.pointPolygonTest(proj, (float(x), float(y)), True)) for x, y in outline_samples(hull, 64)])
    views = o.get('views') or [(hull, pose)]
    ious = view_ious(o['p'], shape, up, views, K, dist)
    base = V[np.abs(V[:, 2]) < 1e-9][:, :2]
    foot = float(cv2.contourArea(cv2.convexHull(base.astype(np.float32)))) if len(base) >= 3 else float('nan')
    info = o.get('info') or {}
    out = dict(
        shape=shape, L=L, W=W, H=H, yaw_deg=float(math.degrees(p[2]) % 180), volume=vol, surface_area=area,
        footprint_area=foot, bbox_diagonal=float(math.sqrt(L * L + W * W + H * H)),
        fill=float(vol / max(L * W * H, 1e-12)), aspect_LW=float(L / max(W, 1e-9)), aspect_HL=float(H / max(L, 1e-9)),
        compactness=float(36 * math.pi * vol ** 2 / max(area ** 3, 1e-18)),  # 1 for a sphere
        sd={d: float(info[d][1]) for d in 'LWH' if info.get(d)}, retention={d: float(info[d][0]) for d in 'LWH' if info.get(d)},
        cam_distance=d_cam, cam_height=float(abs(cam[2])), cam_elevation_deg=float(math.degrees(math.asin(min(1, abs(cam[2] - ctr[2]) / max(d_cam, 1e-9))))),
        depth_z=Zc, mm_per_px=float(Zc / K[0, 0] * 1e3), n_views=int(o.get('n_views', 1)),
        iou=float(o['iou']), view_iou=[float(v) for v in ious], outline_rms_px=float(np.sqrt((res_px ** 2).mean())),
        outline_max_px=float(res_px.max()), cut_by_frame=bool(o.get('cut')), profile=o.get('profile'),
        plane_rms_px=float(pose.get('rms', float('nan'))), f_px=float(K[0, 0]), n_vertices=int(len(V)), n_faces=int(len(Q)))
    return out


def cad_spec(o, name):
    """What the CAD script needs for one fitted object, in millimetres in the object frame (footprint centre at the
    origin, z up, the fitted length along x), like the OBJ: the sizes of the analytic shapes, or the mesh of a
    superellipsoid ('round') as triangles."""
    shape, up = o['shape'], o.get('up', 1.0)
    p = tied(o['p'], shape)
    L, W, H = (float(v) * 1e3 for v in np.exp(p[3:6]))
    spec = dict(name=name, shape=shape, L=L, W=W, H=H)
    if shape == 'revolve':
        spec['d'] = [float(v) * 1e3 for v in revolve_profile(p)]
    elif shape == 'round':
        V, Q = shape_mesh(o['p'], shape, up)
        c, s_ = math.cos(-p[2]), math.sin(-p[2])
        X = V - np.array([p[0], p[1], 0.0])
        X = np.stack([X[:, 0] * c - X[:, 1] * s_, X[:, 0] * s_ + X[:, 1] * c, X[:, 2] * up], 1) * 1e3
        key, idx, keep = {}, [], []  # the grid repeats the pole and the base centre: one vertex each
        for x in np.round(X, 4):
            k = tuple(x)
            if k not in key:
                key[k] = len(keep)
                keep.append(x)
            idx.append(key[k])
        T = []
        for q in Q:
            u = list(dict.fromkeys(idx[int(i)] for i in q))  # a quad collapsed at a pole is a triangle
            if len(u) == 4:
                T += [[u[0], u[1], u[2]], [u[0], u[2], u[3]]]
            elif len(u) == 3:
                T.append(u)
        spec.update(V=np.array(keep).tolist(), T=T)
    return spec


CAD_SCRIPT = r'''
import json, os, FreeCAD, Part
from FreeCAD import Vector as Vec
job = json.load(open(os.environ['CAMDIST_CAD']))
out = []
for it in job['items']:
    sp, stem = it['spec'], it['stem']
    L, W, H, shape = sp['L'], sp['W'], sp['H'], sp['shape']
    if shape == 'box':
        sol = Part.makeBox(L, W, H, Vec(-L / 2, -W / 2, 0))
    elif shape == 'cylinder':
        sol = Part.makeCylinder(L / 2, H, Vec(0, 0, 0), Vec(0, 0, 1))
    elif shape == 'sphere':
        sol = Part.makeSphere(L / 2, Vec(0, 0, L / 2), Vec(0, 0, 1))
    elif shape == 'revolve':
        d = sp['d']
        pts = [(0, 0), (d[0] / 2, 0), (d[1] / 2, H / 3), (d[2] / 2, 2 * H / 3), (d[3] / 2, H), (0, H)]
        wire = Part.makePolygon([Vec(x, 0, z) for x, z in pts] + [Vec(0, 0, 0)])
        sol = Part.Face(wire).revolve(Vec(0, 0, 0), Vec(0, 0, 1), 360)
    elif shape == 'dome':  # upper half of an ellipsoid standing on its flat base
        sol = Part.makeSphere(1.0, Vec(0, 0, 0), Vec(0, 0, 1), 0, 90, 360)
        m = FreeCAD.Matrix()
        m.A11, m.A22, m.A33 = L / 2, W / 2, H
        sol = sol.transformGeometry(m)
        sol = Part.Solid(Part.Shell(sol.Faces)) if not sol.Solids else sol.Solids[0]
    else:  # round: the fitted superellipsoid as a faceted solid
        V = [Vec(*v) for v in sp['V']]
        faces = []
        for a, b, c in sp['T']:
            try:
                faces.append(Part.Face(Part.makePolygon([V[a], V[b], V[c], V[a]])))
            except Exception:
                pass
        comp = Part.Compound(faces)
        comp.sewShape(1e-3)
        sol = Part.Solid(comp.Shells[0])
        if sol.Volume < 0:
            sol.reverse()
    doc = FreeCAD.newDocument('camdist')
    doc.Comment = it.get('comment', '')
    obj = doc.addObject('Part::Feature', 'Body')
    obj.Shape = sol
    obj.Label = sp['name']
    doc.recompute()
    Part.export([obj], stem + '.step')
    doc.saveAs(stem + '.FCStd')
    back = Part.read(stem + '.step')
    bb = back.BoundBox
    out.append(dict(stem=stem, volume=back.Volume, valid=bool(back.isValid()), solids=len(back.Solids),
                    bbox=[bb.XLength, bb.YLength, bb.ZLength], z0=bb.ZMin))
    FreeCAD.closeDocument(doc.Name)
json.dump(out, open(job['result'], 'w'))
'''


def write_cad(items):
    """Solid CAD models of fitted objects with FreeCAD (headless): items = [(stem, spec, comment)] -> STEP and
    FreeCAD files next to each stem. Returns FreeCAD's re-read of every STEP file (volume, validity, box), or None
    when FreeCAD is not installed."""
    exe = shutil.which('freecadcmd')
    if exe is None:
        return None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        job, script, res = (os.path.join(d, n) for n in ('job.json', 'cad.py', 'result.json'))
        json.dump(dict(items=[dict(stem=st, spec=sp, comment=cm) for st, sp, cm in items], result=res), open(job, 'w'))
        open(script, 'w').write(CAD_SCRIPT)
        r = subprocess.run([exe, script], env=dict(os.environ, CAMDIST_CAD=job), capture_output=True, text=True,
                           timeout=180, stdin=subprocess.DEVNULL)
        if not os.path.exists(res):
            raise RuntimeError('FreeCAD did not write a result: ' + (r.stderr or r.stdout)[-300:])
        return json.load(open(res))


def cad_for_objects(objs, stem, namer=None, img=None, note=''):
    """STEP + FreeCAD files for each object of a result (stem_1, stem_2 ... when several); prints what was written."""
    items = []
    for i, o in enumerate(objs):
        nm = o.get('name') or (namer(img, o['hull']) if namer is not None and img is not None else None)
        name = (nm['general'] if nm else o['label']).replace('/', '-')
        sp = cad_spec(o, name)
        items.append((stem if len(objs) == 1 else f'{stem}_{i + 1}', sp,
                      f"camdist {sp['shape']} {sp['L']:.1f} x {sp['W']:.1f} x {sp['H']:.1f} mm; {note}"))
    got = write_cad(items)
    if got is None:
        print('CAD: FreeCAD (freecadcmd) is not installed, so no STEP file; the OBJ mesh is written with the report')
        return []
    for it, g in zip(items, got):
        print(f"CAD {it[0]}.step / .FCStd: {it[1]['name']}, volume {g['volume'] / 1e3:.1f} cm3, "
              f"{'valid solid' if g['valid'] and g['solids'] == 1 else 'CHECK THE SOLID'}")
    return got


def write_obj(path, V, Q, p, up):
    """Wavefront OBJ in millimetres, object frame (footprint centre at the origin, long side on x)."""
    c, s = math.cos(-p[2]), math.sin(-p[2])
    X = V - np.array([p[0], p[1], 0.0])
    X = np.stack([X[:, 0] * c - X[:, 1] * s, X[:, 0] * s + X[:, 1] * c, X[:, 2] * up], 1) * 1e3
    with open(path, 'w') as f:
        f.write('# camdist fitted mesh, millimetres, z up\n')
        f.writelines(f'v {x:.3f} {y:.3f} {z:.3f}\n' for x, y, z in X)
        f.writelines(f'f {a + 1} {b + 1} {c_ + 1} {d + 1}\n' for a, b, c_, d in Q if len({a, b, c_, d}) == 4)


def render_report(path, img, o, pose, K, dist, name=None, calib_note='', source='', seg=''):
    """One JPEG: the photo with the fitted wireframe (visible edges bright, hidden ones dotted,
    the object's bounding box with its three sizes), top / front / side drawings, and the numbers.
    Also writes <path>.json (the analysis) and <path>.obj (the mesh). Returns the analysis."""
    plt = _plt()
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe
    shape, up = o['shape'], o.get('up', 1.0)
    an = object_analysis(o, pose, K, dist)
    V, Q = shape_mesh(o['p'], shape, up)
    ed = mesh_edges(Q)
    cam = -pose['R'].T @ pose['t']
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    vis = np.einsum('ij,ij->i', n, V[Q].mean(1) - cam) < 0  # facing the camera (outward normals)
    uv = cv2.projectPoints(V, pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2)
    hull = np.asarray(o['hull'], float).reshape(-1, 2)
    H_, W_ = img.shape[:2]
    # crop around the object, with room around it
    allp = np.vstack([hull, uv])
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, (y1 - y0) / 1.0) / 2 * 1.5 + 30
    hx, hy = half, half * 0.95
    a, b, c_, d_ = int(max(0, cx - hx)), int(max(0, cy - hy)), int(min(W_, cx + hx)), int(min(H_, cy + hy))
    crop = img[b:d_, a:c_, ::-1]
    halo = [pe.withStroke(linewidth=2.6, foreground='black')]

    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor='#f4f4f1')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1.15], left=0.015, right=0.985, top=0.855,
                          bottom=0.03, wspace=0.04, hspace=0.1)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(crop)
    ax.set_axis_off()
    ax.plot(*(np.vstack([hull, hull[:1]]) - [a, b]).T, color='#ffd400', lw=1.4, label='observed outline')
    hid = [(uv[i] - [a, b], uv[j] - [a, b]) for (i, j), fs in ed.items() if not any(vis[f] for f in fs)]
    shown = [(uv[i] - [a, b], uv[j] - [a, b]) for (i, j), fs in ed.items() if any(vis[f] for f in fs)]
    ax.add_collection(LineCollection(hid, colors='#ff7a59', linewidths=0.6, linestyles=(0, (2, 3)), alpha=0.8))
    lc = LineCollection(shown, colors='#00f0b0', linewidths=0.9)
    lc.set_path_effects([pe.withStroke(linewidth=2.4, foreground=(0, 0, 0, 0.55))])
    ax.add_collection(lc)
    # oriented bounding box with the three sizes
    cyaw, syaw = math.cos(o['p'][2]), math.sin(o['p'][2])
    ex, ey = np.array([cyaw, syaw, 0.0]), np.array([-syaw, cyaw, 0.0])
    ez = np.array([0.0, 0.0, up])
    ctr = np.array([o['p'][0], o['p'][1], 0.0])
    Ls, Ws, Hs = ((np.exp(tied(o['p'], shape)[3 + i])) for i in range(3))
    if shape == 'revolve':
        Ls = Ws = float(revolve_profile(tied(o['p'], shape)).max())
    corners = np.array([ctr + sx * Ls / 2 * ex + sy * Ws / 2 * ey + z * Hs * ez
                        for z in (0, 1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    cu = cv2.projectPoints(corners, pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2) - [a, b]
    ax.add_collection(LineCollection([(cu[i], cu[j]) for i, j in BOX_EDGES], colors='#ffffff', linewidths=0.7,
                                     linestyles=(0, (4, 3)), alpha=0.9))
    # label the three edges meeting at the base corner nearest the camera
    near = int(np.argmin(np.linalg.norm(corners[:4] - cam, axis=1)))
    for j, txt in (((near + 1) % 4, None), ((near - 1) % 4, None), (near + 4, f'H {Hs * 1e3:.0f} mm')):
        m = (cu[near] + cu[j]) / 2
        txt = txt or f'{np.linalg.norm(corners[near] - corners[j]) * 1e3:.0f} mm'
        ax.text(m[0], m[1], txt, color='white', fontsize=10, fontweight='bold', ha='center', va='center', path_effects=halo)
    ax.set_xlim(0, crop.shape[1])
    ax.set_ylim(crop.shape[0], 0)
    ax.plot([], [], color='#00f0b0', lw=1.5, label='fitted mesh (front faces)')
    ax.plot([], [], color='#ff7a59', lw=1.2, ls=':', label='hidden edges')
    ax.plot([], [], color='white', lw=1, ls='--', label='bounding box')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.55, facecolor='black', labelcolor='white')
    ins = ax.inset_axes([0.0, 0.8, 0.2, 0.2])
    ins.imshow(img[..., ::-1])
    ins.add_patch(plt.Rectangle((a, b), c_ - a, d_ - b, fill=False, ec='#ffd400', lw=1))
    ins.set_xticks([])
    ins.set_yticks([])

    # drawings in the object's own frame
    swap = Ls < Ws  # drawings: the long side along x
    yd = o['p'][2] + (math.pi / 2 if swap else 0.0)
    cyaw, syaw = math.cos(yd), math.sin(yd)
    Ls, Ws = max(Ls, Ws), min(Ls, Ws)
    loc = np.stack([(V[:, 0] - o['p'][0]) * cyaw + (V[:, 1] - o['p'][1]) * syaw,
                    -(V[:, 0] - o['p'][0]) * syaw + (V[:, 1] - o['p'][1]) * cyaw, V[:, 2] * up], 1) * 1e3
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.28)
    titles = ('top', 'front (long side)', 'side (short side)')
    axes_ = ((0, 1, Ls, Ws), (0, 2, Ls, Hs), (1, 2, Ws, Hs))
    for k, (ia, ib, da, db) in enumerate(axes_):
        axk = fig.add_subplot(sub[k])
        axk.add_collection(LineCollection([(loc[i][[ia, ib]], loc[j][[ia, ib]]) for (i, j) in ed], colors='#6b7280',
                                          linewidths=0.5))
        pts = loc[:, [ia, ib]]
        hh = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        axk.plot(*np.vstack([hh, hh[:1]]).T, color='#0b7a5a', lw=1.6)
        if k == 0:  # the outline seen from this camera, put on the floor: stretched away from the camera
            F, lam = pixels_to_plane(hull.astype(np.float32), K, dist, pose['R'], pose['t'])
            ok = lam > 0
            if ok.sum() >= 3:
                fl = np.stack([(F[ok, 0] - o['p'][0]) * cyaw + (F[ok, 1] - o['p'][1]) * syaw,
                               -(F[ok, 0] - o['p'][0]) * syaw + (F[ok, 1] - o['p'][1]) * cyaw], 1) * 1e3
                axk.plot(*np.vstack([fl, fl[:1]]).T, color='#c99a00', lw=0.9, ls=':', label='outline on floor')
            cl = np.array([(cam[0] - o['p'][0]) * cyaw + (cam[1] - o['p'][1]) * syaw,
                           -(cam[0] - o['p'][0]) * syaw + (cam[1] - o['p'][1]) * cyaw]) * 1e3
            for _, ps in (o.get('views') or []):
                cm = -ps['R'].T @ ps['t'] if 'R' in ps else None
                if cm is not None:
                    q = np.array([(cm[0] - o['p'][0]) * cyaw + (cm[1] - o['p'][1]) * syaw,
                                  -(cm[0] - o['p'][0]) * syaw + (cm[1] - o['p'][1]) * cyaw]) * 1e3
                    axk.plot(*q, marker='v', ms=4, color='#2563eb', ls='')
            rr = 0.75 * max(Ls, Ws, 0.03) * 1e3
            u = cl / max(np.linalg.norm(cl), 1e-9)
            axk.annotate('camera', xy=u * 0.95 * rr, xytext=u * 0.5 * rr, fontsize=7, color='#2563eb', ha='center',
                         arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.2))
            axk.legend(fontsize=6.5, loc='lower right', framealpha=0.7)
        r = 0.75 * max(da, db, 0.03) * 1e3
        axk.set_xlim(-r, r)
        axk.set_ylim(-r * 0.3 if k else -r, r * 1.5 if k else r)
        axk.set_aspect('equal')
        axk.set_title(titles[k], fontsize=8.5)
        axk.tick_params(labelsize=6.5)
        axk.annotate('', xy=(da * 5e2, pts[:, 1].min() - 0.1 * r), xytext=(-da * 5e2, pts[:, 1].min() - 0.1 * r),
                     arrowprops=dict(arrowstyle='<->', lw=0.8, color='#111'))
        axk.text(0, pts[:, 1].min() - 0.2 * r, f'{da * 1e3:.0f}', ha='center', va='top', fontsize=7.5)
        axk.annotate('', xy=(pts[:, 0].max() + 0.1 * r, pts[:, 1].max()), xytext=(pts[:, 0].max() + 0.1 * r, pts[:, 1].min()),
                     arrowprops=dict(arrowstyle='<->', lw=0.8, color='#111'))
        axk.text(pts[:, 0].max() + 0.2 * r, (pts[:, 1].max() + pts[:, 1].min()) / 2, f'{db * 1e3:.0f}', va='center',
                 fontsize=7.5, rotation=90)
        axk.spines[['top', 'right']].set_visible(False)

    # numbers
    axt = fig.add_subplot(gs[1, 1])
    axt.set_axis_off()
    sd = an['sd']
    sdt = lambda d: f" ± {sd[d] * 1e3:.0f}" if d in sd and sd[d] < 0.2 else ''
    rows = [('SIZE', None),
            ('length  L', f"{an['L'] * 1e3:.1f} mm{sdt('L')}"), ('width  W', f"{an['W'] * 1e3:.1f} mm{sdt('W')}"),
            ('height  H', f"{an['H'] * 1e3:.1f} mm{sdt('H')}"), ('bounding-box diagonal', f"{an['bbox_diagonal'] * 1e3:.0f} mm"),
            ('GEOMETRY', None),
            ('fitted shape', f"{shape}  ({an['n_faces']} quads)"),
            ('volume / surface area', f"{an['volume'] * 1e6:.1f} cm³  /  {an['surface_area'] * 1e4:.1f} cm²"),
            ('footprint area', f"{an['footprint_area'] * 1e4:.1f} cm²"),
            ('fills its bounding box', f"{an['fill'] * 100:.0f} %  (box 100, ball 52)"),
            ('aspect  L:W  /  H:L', f"{an['aspect_LW']:.2f}  /  {an['aspect_HL']:.2f}"),
            ('orientation on the plane', f"{an['yaw_deg']:.0f}°"),
            ('FIT AND VIEW', None),
            ('outline overlap (IoU)', f"{an['iou']:.3f}" + (f"   per view {', '.join(f'{v:.2f}' for v in an['view_iou'][:6])}" if an['n_views'] > 1 else '')),
            ('outline residual rms / max', f"{an['outline_rms_px']:.1f} px / {an['outline_max_px']:.1f} px  ({an['outline_rms_px'] * an['mm_per_px']:.1f} mm)"),
            ('views pooled', f"{an['n_views']}" + ('   ·   cut by the frame edge' if an['cut_by_frame'] else '')),
            ('camera: distance / height / elevation', f"{an['cam_distance'] * 1e3:.0f} mm / {an['cam_height'] * 1e3:.0f} mm / {an['cam_elevation_deg']:.0f}°"),
            ('image scale at the object', f"{an['mm_per_px']:.2f} mm per px"),
            ('CALIBRATION', None),
            ('focal length  f', f"{an['f_px']:.0f} px   ·   plane fit rms {an['plane_rms_px']:.2f} px"),
            ('source', (calib_note or source)[:46])]
    rows = [(k, v) for k, v in rows if v is not None or k.isupper()]
    y = 1.0
    for k, v in rows:
        if v is None:
            y -= 0.012
            axt.text(0.0, y, k, fontsize=8.5, fontweight='bold', color='#0b7a5a', va='top', transform=axt.transAxes)
            axt.plot([0.17 if len(k) < 6 else 0.27, 1], [y - 0.02] * 2, color='#c9c9c4', lw=0.6, transform=axt.transAxes)
            y -= 0.052
        else:
            axt.text(0.0, y, k, fontsize=8.8, color='#444', va='top', transform=axt.transAxes)
            axt.text(0.47, y, v, fontsize=8.8, color='#111', va='top', transform=axt.transAxes, family='monospace')
            y -= 0.047
    # header
    fig.text(0.02, 0.945, name['general'].capitalize() if name else 'Object', fontsize=26, fontweight='bold', color='#111')
    if name:
        alt = ',  '.join(f"{n} {p * 100:.0f}%" for n, p in name['top'][1:3])
        fig.text(0.02, 0.905, f"{name['category']} ({name['p_category'] * 100:.0f}%)  ·  {name['name']} {name['p'] * 100:.0f}%"
                              f"  ·  next: {alt}", fontsize=10, color='#555')
    else:
        fig.text(0.02, 0.905, 'object name not requested (--name)', fontsize=10, color='#555')
    big = f"{an['L'] * 1e3:.0f} × {an['W'] * 1e3:.0f} × {an['H'] * 1e3:.0f} mm"
    fig.text(0.985, 0.945, big, fontsize=26, fontweight='bold', color='#0b7a5a', ha='right')
    fig.text(0.985, 0.905, 'L × W × H  ·  ' + (f'outline: {seg}  ·  ' if seg else '') + 'single camera',
             fontsize=10, color='#555', ha='right')
    fig.savefig(path, format='jpg', dpi=140, pil_kwargs=dict(quality=93))
    plt.close(fig)
    base = os.path.splitext(path)[0]
    an['name'] = name
    with open(base + '.json', 'w') as f:
        json.dump(an, f, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
    write_obj(base + '.obj', V, Q, tied(o['p'], shape), up)
    return an


def shrink(img, max_side):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def info_text(info):
    sd = ' / '.join(f'{info[d][1] * 1e3:.1f}' for d in 'LWH')
    rho = ' / '.join(f'{info[d][0]:.2f}' for d in 'LWH')
    return f'\n    +- {sd} mm (L / W / H), retention {rho}'


def select_at(res, pt):
    """Index of the measured object whose outline contains the clicked pixel, else None."""
    for i, o in enumerate(res['objects']):
        if cv2.pointPolygonTest(np.asarray(o['hull'], np.float32).reshape(-1, 2), (float(pt[0]), float(pt[1])), False) >= 0:
            return i
    return None


def follow_selected(sel, res, diag):
    """Keep the selection on the same object in a newer measurement: the outline whose centre lies nearest
    to where the selected one was (within a quarter of the frame diagonal). Dropped after 15 misses."""
    cs = [np.asarray(o['hull'], float).reshape(-1, 2).mean(0) for o in res['objects']]
    if cs:
        d = [float(np.linalg.norm(c - sel['cen'])) for c in cs]
        i = int(np.argmin(d))
        if d[i] < 0.25 * diag:
            return dict(cen=cs[i], idx=i, lost=0)
    return None if sel['lost'] >= 15 else dict(sel, idx=None, lost=sel['lost'] + 1)


def save_reports(args, img, res, room, namer, path, source=''):
    """Report JPEG (+ .json, .obj) for every object of one result; `namer` is created on first use."""
    if namer is None and args.name and not any(o.get('name') for o in res['objects']):
        namer = ObjectNamer()
    objs = [o for o in res['objects'] if not o.get('cut')] or res['objects']
    stem, ext = os.path.splitext(path)
    for i, o in enumerate(objs):
        nm = o.get('name') or (namer(img, o['hull']) if namer else None)
        out = path if len(objs) == 1 else f'{stem}_{i + 1}{ext or ".jpg"}'
        an = render_report(out, img, o, res['floor'], room.K, room.dist, nm, room.calib_note, source=source, seg=args.seg)
        print(f"report {out}: {nm['general'] + ', ' if nm else ''}{an['L'] * 1e3:.0f} x {an['W'] * 1e3:.0f} x "
              f"{an['H'] * 1e3:.0f} mm, volume {an['volume'] * 1e6:.0f} cm3")
    if getattr(args, 'cad', False):
        cad_for_objects(objs, stem, namer, img, note=source)
    return namer


def rig_priors(room, res):
    """What the better camera of a rig tells the other where to look: the fitted shape (parameters, shape, up) of its well fitted,
    uncut objects in room-corner coordinates."""
    if not res or res.get('floor') is None:
        return ()
    return [dict(p=np.asarray(o['p'], float), shape=o['shape'], up=o['up']) for o in res['objects']
            if o.get('fixed') and not o['cut'] and o.get('iou', 0) >= 0.8]


def rig_views(room, res):
    """What one camera of a rig hands the other: (outline, floor pose with this camera's own K and dist, centre)
    of its objects, all in room-corner coordinates, so the other camera can fit them together with its own."""
    if not res or res.get('floor') is None or room.K is None:
        return ()
    ps = dict(res['floor'], K=np.asarray(room.K, float), dist=np.asarray(room.dist, float))
    return [(o['hull'], ps, np.asarray(o['p'][:2])) for o in res['objects'] if o.get('fixed') and not o['cut']]


class SecondRoom:
    """The rig's second camera (the ESP32-CAM on the phone's USB cable, relayed by the app): its own Room with the
    ESP's stored calibration and a worker thread. Both cameras see the checkerboard room, each finds its own pose
    in the same room coordinates, so no stereo calibration and no clock are needed: every object is fitted
    to the outlines of both cameras together."""

    def __init__(self, args, det, img, partner, prior=lambda: ()):
        self.room = Room(args, (img.shape[1], img.shape[0]), fallback_calib('esp32'), store=False)
        self.cond, self.job, self.res, self.img, self.err, self.dt, self.stop = threading.Condition(), None, None, None, None, 0.0, False
        self.partner, self.prior, self.det = partner, prior, det
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def push(self, img):
        with self.cond:
            self.job = img
            self.cond.notify()

    def get(self):
        with self.cond:
            return self.res, self.img

    def _work(self):
        while True:
            with self.cond:
                while self.job is None and not self.stop:
                    self.cond.wait()
                if self.stop:
                    return
                img, self.job = self.job, None
            t0 = time.time()
            try:
                res = self.room.process(img, self.det, self.partner(), self.prior())
            except (Exception, SystemExit) as e:
                self.err = f'{type(e).__name__}: {e}'
                continue
            with self.cond:
                self.res, self.img, self.dt, self.err = res, img, time.time() - t0, None

    def close(self):
        with self.cond:
            self.stop = True
            self.cond.notify()
        self.thread.join(timeout=10)


def cmd_room(args):
    """Size of objects standing in the checkerboard room (floor and walls of 10 mm squares).

    The panels calibrate the camera from one frame and the floor gives the camera pose in every
    frame, so the camera may move. Objects are found where the room's pattern is hidden
    (--seg room) or by the COCO detector (--seg detect); each outline is fitted with an upright
    shape standing on the floor, and its length, width and height are read off the fit.
    Live, measuring runs in a background thread on frames shrunk to --max-side, while the window
    shows every camera frame with the newest result."""
    if args.table:
        args.seg = 'salient'
    calib = load_calib(args.calib) if args.calib else None
    det = Detector(args.min_score) if args.seg == 'detect' else None
    if args.object:  # known objects carry their true size (remembered for new names)
        known = dict(ROOM_OBJECTS, **cfg_get('room_objects', {}))
        if args.truth:
            known[args.object] = tuple(args.truth)
            cfg_set('room_objects', {k: v for k, v in known.items() if ROOM_OBJECTS.get(k) != tuple(v)})
        if args.object not in known:
            raise SystemExit(f'unknown object {args.object!r}: give its size once with --truth L W H [m]')
        args.truth = list(known[args.object])
        print(f'object {args.object}: true size {" x ".join(f"{v * 1e3:.0f}" for v in args.truth)} mm')
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f'cannot read {args.image}')
        img = shrink(img, args.max_side)
        if args.denoise:
            dn = load_denoiser() or sys.exit('no trained denoiser: run denoise-train first')
            img = denoise_bgr(dn, img)
        room = Room(args, (img.shape[1], img.shape[0]), calib, store=False)
        t0 = time.time()
        res = room.process(img, det)
        for ln in room.status(res):
            print(ln)
        for o in res['objects']:
            print(f"{o['label']}: {o['L'] * 1e3:.0f} x {o['W'] * 1e3:.0f} x {o['H'] * 1e3:.0f} mm "
                  f"(L x W x H), {o['shape']} fit IoU {o['iou']:.3f}" + ('  cut off by the frame edge' if o['cut'] else '')
                  + (info_text(o['info']) if o.get('info') else ''))
        print(f"{len(res['objects'])} objects, {time.time() - t0:.1f} s")
        disp = room.draw(img, res, show_occ=args.show_occ)
        draw_text(disp, room.status(res), scale=0.45)
        if args.report and res['objects']:
            save_reports(args, img, res, room, None, args.report, source=os.path.basename(args.image))
        if args.out:
            cv2.imwrite(args.out, disp)
            print(f'annotated image: {args.out}')
        elif not args.report:
            cv2.namedWindow('room', cv2.WINDOW_NORMAL)
            cv2.imshow('room', disp)
            cv2.waitKey(0)
        return
    dn = None
    if args.denoise:
        dn = load_denoiser() or sys.exit('no trained denoiser: run denoise-train first')
        print(f'denoiser on: every frame is cleaned before measuring ({dn.info.get("ms_per_frame", 0):.0f} ms each)')
    live = Live(args, 'room')
    cond, box = threading.Condition(), dict(job=None, res=None, img=None, dt=0.0, stop=False, err=None)
    room = None

    def work():
        while True:
            with cond:
                while box['job'] is None and not box['stop']:
                    cond.wait()
                if box['stop']:
                    return
                img, box['job'] = box['job'], None
            t0 = time.time()
            try:
                res = room.process(img, det, rig_views(rig[0].room, rig[0].get()[0]) if rig else ())
            except (Exception, SystemExit) as e:  # keep the window alive; show the error instead
                box['err'] = f'{type(e).__name__}: {e}'
                continue
            with cond:
                box.update(res=res, img=img, dt=time.time() - t0, err=None)

    worker = threading.Thread(target=work, daemon=True)
    show_occ, last_logged, namer, sel, sel_res, k_prev = args.show_occ, None, None, None, None, -1
    cmd_keys, pub_res, pub_res2, show_mesh = [], None, None, False
    rig, seq2, img2, w1, s2 = [], 0, None, None, 1.0  # rig: the second camera, once it streams
    link = live.cam.link if live.cam.kind == 'webphone' else None
    stack = args.stack if args.stack is not None else (6 if live.cam.kind == 'esp32' else 1)
    acc, n_acc, prev_small = None, 0, None
    try:
        while True:
            img = shrink(live.frame(), args.max_side)
            if dn is not None and live.cam.kind != 'webphone':
                img = denoise_bgr(dn, img)
            if stack > 1:  # average frames while the camera is still: random sensor noise drops by sqrt(n)
                small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25).astype(np.float32)
                still = prev_small is not None and prev_small.shape == small.shape and \
                    float(np.mean(np.abs(small - prev_small))) < 4.0
                prev_small = small
                if still and acc is not None and acc.shape == img.shape:
                    n_acc = min(n_acc + 1, stack)
                    acc += (img.astype(np.float32) - acc) / n_acc
                else:  # moved (or first frame): start over
                    acc, n_acc = img.astype(np.float32), 1
                img = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
            if room is None:
                room = Room(args, (img.shape[1], img.shape[0]), calib)
                if args.table and calib is None:
                    room.fallback = fallback_calib(live.cam.kind)
                worker.start()
            with cond:  # the worker always takes the newest frame
                box['job'] = img
                cond.notify()
                res, rimg, dt, err = box['res'], box['img'], box['dt'], box['err']
            if link is not None:  # the second camera (ESP32-CAM on the phone's cable), when the app relays it
                got = link.second()
                if got is None:
                    img2 = None
                elif got[0] != seq2:
                    seq2, img2 = got[0], shrink(got[1], args.max_side)
                    if not rig:
                        rig.append(SecondRoom(args, det, img2, lambda: rig_views(room, box['res']), lambda: rig_priors(room, box['res'])))
                        print('second camera: the ESP32-CAM relayed by the phone app (both views are fitted together)')
                    rig[0].push(img2)
            res2, rimg2 = rig[0].get() if rig else (None, None)
            sidx2 = None

            def from_second(pt):
                """A point clicked in the second camera's image -> a point of the same object in the first one's
                (outside every outline when nothing there: clears); None when the first camera does not see it."""
                if not (res and res2):
                    return None
                i2 = select_at(res2, pt)
                if i2 is None:
                    return (-1.0, -1.0)
                d = [math.dist(o['p'][:2], res2['objects'][i2]['p'][:2]) for o in res['objects']]
                if d and min(d) < 0.1:
                    return tuple(np.asarray(res['objects'][int(np.argmin(d))]['hull'], float).reshape(-1, 2).mean(0))
                return None
            if link is not None:  # taps and buttons on the phone act like the click and the keys here
                for tx, ty, cam in link.take_taps():
                    if cam == 2:
                        pt = from_second((tx * img2.shape[1], ty * img2.shape[0])) if img2 is not None else None
                        live.click = pt if pt is not None else live.click
                    else:
                        live.click = (tx * img.shape[1], ty * img.shape[0])
                cmd_keys += [dict(log=ord(' '), report=ord('m'), cad=ord('k'), clear=ord('x'), reset=ord('r'), shape=ord('s'), mesh=ord('w')).get(c, -1)
                             for c in link.take_cmds()]
            if live.click is not None and w1 is not None and live.click[0] >= w1:  # clicked in the right half of the window
                live.click = from_second(((live.click[0] - w1) / s2, live.click[1] / s2))
            if res is not None and res is not sel_res:  # a new measurement: the selection follows its object
                sel_res = res
                if sel is not None:
                    sel = follow_selected(sel, res, math.hypot(*img.shape[:2]))
            if live.click is not None:  # click an object to select it (click elsewhere: clear)
                pt, live.click = live.click, None
                i = select_at(res, pt) if res else None
                sel = None if i is None else dict(cen=np.asarray(res['objects'][i]['hull'], float).reshape(-1, 2).mean(0),
                                                  idx=i, lost=0)
            if res and k_prev in (9, ord('n')) and res['objects']:  # Tab / n: next object (the mouse may not reach the window)
                i = 0 if sel is None or sel['idx'] is None else (sel['idx'] + 1) % len(res['objects'])
                sel = dict(cen=np.asarray(res['objects'][i]['hull'], float).reshape(-1, 2).mean(0), idx=i, lost=0)
            if k_prev == ord('x'):
                sel = None
            sidx = sel['idx'] if sel is not None else None
            view = res  # what space / m log: the selected object only, else everything measured
            if res and sel is not None:
                view = dict(res, objects=[res['objects'][sidx]] if sidx is not None else [])
            disp = room.draw(img, res, show_occ, sidx, sel is not None) if res else img.copy()
            if show_mesh and res:
                draw_mesh(disp, res, room.K, room.dist)
            w1 = None
            if rig and img2 is not None:  # split view: the second camera to the right, the selected object in both
                if res and res2 and sidx is not None and sidx < len(res['objects']):
                    d = [math.dist(o['p'][:2], res['objects'][sidx]['p'][:2]) for o in res2['objects']]
                    if d and min(d) < 0.1:
                        sidx2 = int(np.argmin(d))
                disp2 = rig[0].room.draw(img2, res2, show_occ, sidx2, sel is not None) if res2 else img2.copy()
                if show_mesh and res2:
                    draw_mesh(disp2, res2, rig[0].room.K, rig[0].room.dist)
                s2 = disp.shape[0] / disp2.shape[0]
                disp2 = cv2.resize(disp2, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
                w1 = disp.shape[1]
                disp = np.hstack([disp, disp2])
            lines = [live.status_line(img) + (f'   measuring {1 / dt:.1f}/s' if dt else '   measuring ...')
                     + (f'   stack {n_acc}/{stack}' if stack > 1 else '')]
            lines += room.status(res) if res else ['finding the panels ...']
            if rig and img2 is not None:
                lines += ['ESP32-CAM (right): ' + (rig[0].room.status(res2)[0] if res2 else 'finding the panels ...')
                          + (f"   {sum(1 for o in res2['objects'] if o.get('guided'))} found with the phone's help"
                             if res2 and any(o.get('guided') for o in res2['objects']) else '')
                          + f'   outline noise phone {room.noise:.1f} px, ESP {rig[0].room.noise:.1f} px'
                          + (f'   measuring {1 / rig[0].dt:.1f}/s' if rig[0].dt else '')]
                if rig[0].err:
                    lines.append(rig[0].err[:100])
            lines += ([err[:100]] if err else []) + \
                     ['move the camera around the object: more views, better W and H',
                      'click / Tab: select an object (x: clear)  space log  m report  w mesh  k CAD model  r forget views  c recalibrate  f floor  s shape  q quit']
            if live.cam.kind == 'webphone' and res is not None and res is not pub_res:
                pub_res = res
                live.cam.link.publish(phone_payload(rimg.shape, res, sidx, room.status(res), (room.K, room.dist) if show_mesh else None))
            if rig and res2 is not None and res2 is not pub_res2:
                pub_res2 = res2
                link.publish2(phone_payload(rimg2.shape, res2, sidx2, rig[0].room.status(res2),
                                            (rig[0].room.K, rig[0].room.dist) if show_mesh else None))
            k = k_prev = live.show(disp, lines)
            while k < 0 and cmd_keys:
                k = k_prev = cmd_keys.pop(0)
            if k in (ord('q'), 27):
                break
            if live.locked and k > 0 and chr(k) in 'eEgGa':
                live.adjust(chr(k))
            if k == ord('c'):
                room.want_calib = True
            if k == ord('f'):
                room.floor_pick = 1
                if rig:
                    rig[0].room.floor_pick = 1
            if k == ord('o'):
                show_occ = not show_occ
            if k == ord('w'):  # the fitted wireframe over the live view (and on the phone)
                show_mesh = not show_mesh
                pub_res = pub_res2 = None  # send the phone a new payload now
            if k == ord('r'):
                room.tracks, room.pool = [], []
                if rig:
                    rig[0].room.tracks, rig[0].room.pool = [], []
            if k == ord('m') and view and view['objects']:
                os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
                try:
                    namer = save_reports(args, rimg, view, room, namer, os.path.join(ROOT, 'reports', time.strftime('%Y%m%d_%H%M%S') + '.jpg'),
                                         source=live.cam.kind)
                except Exception as e:
                    print(f'report failed: {type(e).__name__}: {e}')
            if k == ord('k') and view and view['objects']:  # solid CAD model (STEP + FreeCAD) of the selected object, else all
                def cad_job(objs, stem):
                    try:
                        cad_for_objects(objs, stem, note=live.cam.kind)
                    except Exception as e:
                        print(f'CAD failed: {type(e).__name__}: {e}')
                os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
                threading.Thread(target=cad_job, daemon=True, args=(
                    [o for o in view['objects'] if not o.get('cut')] or view['objects'],
                    os.path.join(ROOT, 'reports', time.strftime('%Y%m%d_%H%M%S') + '_cad'))).start()
                print('CAD model: working ...')
            if k == ord('s'):
                names = ['auto'] + list(ROOM_SHAPES)
                room.shape = names[(names.index(room.shape) + 1) % len(names)]
                if rig:
                    rig[0].room.shape = room.shape
            if k == ord(' ') and view and view['objects'] and res is last_logged:
                print('same measurement as the last log (it updates about once a second): wait or move the camera')
            elif k == ord(' ') and view and view['objects']:
                last_logged = res
                truth = dict(zip('LWH', args.truth), object=args.object) if args.truth else None
                cur = db().execute('INSERT INTO measurements(ts,kind,calib_id,est,truth,brightness,lighting) '
                                   'VALUES(?,?,?,?,?,?,?)', (time.time(), 'room', room.calib_id,
                                                             jdump(room_record(room, view)),
                                                             jdump(truth) if truth else None,
                                                             live.brightness, args.lighting))
                db().commit()
                put_result('room_raw', dict(measurement=cur.lastrowid), ref=cur.lastrowid, blob=jpg_bytes(rimg, 92))
                if rig and rimg2 is not None:  # the ESP32-CAM's frame of the same moment: the pair can be refitted offline
                    put_result('room_raw2', dict(measurement=cur.lastrowid, calib=rig[0].room.calib_id,
                                                 K=np.asarray(rig[0].room.K).tolist(), dist=np.ravel(rig[0].room.dist).tolist(),
                                                 est=room_record(rig[0].room, res2) if res2 else None), ref=cur.lastrowid,
                               blob=jpg_bytes(rimg2, 92))
                print('logged: ' + '; '.join(f"{o['label']} {o['L'] * 1e3:.0f}x{o['W'] * 1e3:.0f}x{o['H'] * 1e3:.0f} mm"
                                             for o in view['objects']))
    finally:
        with cond:
            box['stop'] = True
            cond.notify()
        if worker.is_alive():
            worker.join(timeout=10)  # let it finish its frame: OpenCV aborts if a thread dies mid-call
        for r in rig:
            r.close()
        live.close()
        db().commit()
        # the camera reader thread may still be blocked inside OpenCV; letting the interpreter tear it
        # down aborts the process ("exception not rethrown"). Everything is saved: leave directly.
        sys.stdout.flush()
        os._exit(0)


def pick_target(dets, prev=None, click=None, label=None):
    """The detection to measure: the clicked one, else the one tracking the previous target,
    else (optionally of a given class) the largest."""
    if not dets:
        return None
    area = lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])
    centre = lambda d: ((d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2)
    if click is not None:
        hit = [d for d in dets if d['box'][0] <= click[0] <= d['box'][2] and d['box'][1] <= click[1] <= d['box'][3]]
        if hit:
            return min(hit, key=area)
    if prev is not None:  # follow by position: detector labels flicker between frames
        diag = math.hypot(prev['box'][2] - prev['box'][0], prev['box'][3] - prev['box'][1])
        near = [d for d in dets if math.dist(centre(d), centre(prev)) < 0.5 * diag]
        if near:
            return min(near, key=lambda d: math.dist(centre(d), centre(prev)) * (1 if d['label'] == prev['label'] else 1.5))
    pool = [d for d in dets if d['label'] == label] if label else dets
    return max(pool, key=area) if pool else None


def measure_object(args, calib):
    """measure --width: distance to one object of known width (e.g. a can), found by its colour
    or, with --detect, by a pretrained object detector."""
    ranges = tape_ranges(args)
    delta = ruler_delta()
    det = Detector(args.min_score) if args.detect else None
    target = None
    live = Live(args, 'measure', calib=calib)

    def find(img):
        nonlocal target
        if det is None:
            return object_extent(img, ranges, args.min_obj_area)
        dets = det(img)
        click, live.click = live.click, None
        target = pick_target(dets, target, click, args.object_class)
        if target is None:
            return None
        x0, y0, x1, y1 = target['box']
        ym = (y0 + y1) / 2
        return dets, np.array([x0, ym]), np.array([x1, ym])

    while True:
        img = live.frame()
        if det is None:
            ranges = handle_click(live, img) or ranges
        got = find(img)
        disp = img.copy()
        lines = [live.status_line(img) + f'   calib #{calib.id} ({calib.method})']
        if got and det is not None:
            for d in got[0]:
                x0, y0, x1, y1 = map(int, d['box'])
                col = (0, 255, 255) if d is target else (180, 180, 180)
                cv2.rectangle(disp, (x0, y0), (x1, y1), col, 3 if d is target else 1)
                draw_text(disp, [f"{d['label']} {d['score']:.2f}"], org=(x0 + 4, max(18, y0 - 6)), scale=0.6,
                          color=col)
        if got:
            c, left, right = got
            if det is None:
                cv2.drawContours(disp, [c.reshape(-1, 1, 2)], -1, (0, 255, 255), 2)
            for p in (left, right):
                cv2.circle(disp, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            Zf, X = object_distance(left, right, calib.K, args.width, args.shape)
            w_px = right[0] - left[0]
            y0, y1 = (target['box'][1], target['box'][3]) if det is not None else (c[:, 1].min(), c[:, 1].max())
            # height at the depth of the silhouette sides: the axis for a cylinder, the face for a flat object
            Hmm = (y1 - y0) * (Zf + (args.width / 2 if args.shape == 'cylinder' else 0)) / calib.fy * 1e3
            lines.append(f'distance to front = {Zf * 1e3:.0f} mm   height = {Hmm:.0f} mm   '
                         f'(width {args.width * 1e3:.0f} mm given)   X = {X * 1e3:+.0f} mm')
            lines.append(f'width {w_px:.0f} px; +-1 px changes the distance by {Zf * Zf / (calib.fx * args.width) * 1e3:.0f} mm')
            draw_text(disp, [f'{Zf * 1e3:.0f} mm away', f'H {Hmm:.0f} mm'], org=(int(left[0]) + 4, int(y1) + 24),
                      scale=0.8, color=(120, 255, 255))
        else:
            lines.append('no object detected' if det else 'object colour not found: click on the object to pick its colour')
        lines.append('space log' + (' with truth' if args.truth else '') + '   q quit   '
                     + ('click an object to measure it' if det else 'click object to pick colour'))
        if det is not None and target is not None:
            lines.append(f"target: {target['label']} (its true width --width {args.width} m)")
        k = live.show(disp, lines)
        if k in (ord('q'), 27):
            break
        if k == ord(' ') and got:
            Zs, frame = [], img
            for _ in range(20):
                frame = live.frame()
                g2 = find(frame)
                if g2:
                    Zs.append(object_distance(g2[1], g2[2], calib.K, args.width, args.shape))
                if len(Zs) >= 8:
                    break
            if len(Zs) < 3:
                print('object lost while averaging; nothing logged')
                continue
            Zs = np.array(Zs)
            Zf, X = np.median(Zs, 0)
            truth = None
            if args.truth:
                zm = live.ask(frame, 'true distance lens -> object front [mm]')
                zm = zm / 1e3 if isinstance(zm, float) else zm
                if zm == 'esc':
                    continue
                if zm is not None:
                    truth = dict(z_meas=zm, delta=delta, Z=zm + delta)
            est = dict(Z=Zf, X=X, Z_std=float(Zs[:, 0].std()), n=len(Zs), width=args.width, shape=args.shape,
                       f=calib.fx)
            db().execute('INSERT INTO measurements(ts,kind,calib_id,est,truth,brightness,lighting) '
                         'VALUES(?,?,?,?,?,?,?)', (time.time(), 'object', calib.id, jdump(est),
                                                   jdump(truth) if truth else None, live.brightness,
                                                   args.lighting))
            db().commit()
            print(f'logged {Zf * 1e3:.0f} mm (+- {Zs[:, 0].std() * 1e3:.0f} mm over {len(Zs)} frames)'
                  + (f"  true {truth['Z'] * 1e3:.0f} mm  err {(Zf - truth['Z']) * 1e3:+.0f} mm "
                     f"({(Zf / truth['Z'] - 1) * 100:+.1f} %)" if truth else ''))
    live.close()


def cmd_measure(args):
    calib = load_calib(args.calib)
    if args.width:
        return measure_object(args, calib)
    if not args.length:
        raise SystemExit('give --length (two tape marks) or --width (one object of known width)')
    L = args.length
    f = (calib.fx + calib.fy) / 2
    delta = ruler_delta()
    ranges = tape_ranges(args)
    live = Live(args, 'measure', calib=calib)
    while True:
        img = live.frame()
        ranges = handle_click(live, img) or ranges
        blobs = find_tape_blobs(img, ranges, args.min_area)
        disp = img.copy()
        draw_blobs(disp, blobs)
        lines = [live.status_line(img) + f'   calib #{calib.id} ({calib.method})']
        if len(blobs) == 2:
            Z, X, Y = ruler_distance(blobs[0][:2], blobs[1][:2], calib.K, L)
            l = math.dist(blobs[0][:2], blobs[1][:2])
            lines.append(f'Z = {Z:.3f} m   X = {X:+.3f} m   l = {l:.1f} px   '
                         f'dZ per px = {Z * Z / (f * L) * 1e3:.1f} mm')
            lines.append(f'from reference edge: {Z - delta:.3f} m (delta {delta * 1e3:.1f} mm)')
        else:
            lines.append(f'{len(blobs)} tape blobs (need 2)')
        lines.append('space log' + (' with truth' if args.truth else '') + '   q quit   click tape to pick colour')
        k = live.show(disp, lines)
        if k in (ord('q'), 27):
            break
        if k == ord(' '):
            cap = capture_tape(live, ranges, args.min_area)
            if cap is None:
                print('two tape blobs not found; nothing logged')
                continue
            Z, X, Y = ruler_distance(cap['P1'], cap['P2'], calib.K, L)
            est = dict(Z=Z, X=X, Y=Y, l_px=cap['l'], l_std=cap['l_std'], L=L, f=f, P1=cap['P1'], P2=cap['P2'])
            truth = None
            if args.truth:
                zm = live.ask(cap['frame'], 'true distance from reference edge [m]')
                if zm == 'esc':
                    continue
                if zm is not None:
                    truth = dict(z_meas=zm, delta=delta, Z=zm + delta)
            db().execute('INSERT INTO measurements(ts,kind,calib_id,est,truth,brightness,lighting) '
                         'VALUES(?,?,?,?,?,?,?)', (time.time(), 'ruler', calib.id, jdump(est),
                                                   jdump(truth) if truth else None, cap['brightness'],
                                                   args.lighting))
            db().commit()
            print(f"logged Z {Z:.4f} m" + (f"  true {truth['Z']:.4f} m  err {(Z - truth['Z']) * 1e3:+.1f} mm"
                                           if truth else ''))
    live.close()


# ----------------------------------------------------------------------------
# ground (handoff 2.3, 3.4)
# ----------------------------------------------------------------------------
def fit_ground(obs, K, h, sigma_h):
    """theta from floor points of known forward distance, with h measured; also joint (h, theta)."""
    from scipy.optimize import least_squares
    u, v, D = obs[:, 0], obs[:, 1], obs[:, 2]

    def res(th, hh):
        return ground_point(u, v, K, hh, th)[0] - D

    alpha = np.arctan((v - K[1, 2]) / K[1, 1])
    th0 = float(np.median(np.arctan2(h, D) - alpha))
    r1 = least_squares(lambda x: res(x[0], h), [th0])
    dof = len(D) - 1
    s2 = float(r1.fun @ r1.fun / dof) if dof > 0 else float('nan')
    se_th = math.sqrt(s2 / float(r1.jac[:, 0] @ r1.jac[:, 0])) if dof > 0 else float('nan')
    out = dict(h=h, theta=float(r1.x[0]), se_theta=se_th, sigma_h=sigma_h, resid=r1.fun,
               rms=math.sqrt(float(np.mean(r1.fun ** 2))))
    if len(D) >= 3:
        r2 = least_squares(lambda x: res(x[1], x[0]), [h, r1.x[0]])
        s2b = float(r2.fun @ r2.fun / (len(D) - 2)) if len(D) > 2 else float('nan')
        cov = np.linalg.pinv(r2.jac.T @ r2.jac) * s2b
        _, _, rho_th = retention(r2.jac[:, 1], r2.jac[:, :1])
        out['joint'] = dict(h=float(r2.x[0]), theta=float(r2.x[1]), se_h=math.sqrt(abs(cov[0, 0])),
                            se_theta=math.sqrt(abs(cov[1, 1])), rho_theta_given_h_unknown=rho_th)
    return out


def ground_setup(args, calib):
    if args.from_db:
        sess = args.from_db
        if sess == 'latest':
            row = db().execute('SELECT session FROM ground_obs ORDER BY id DESC LIMIT 1').fetchone()
            sess = row[0] if row else None
        rows = db().execute('SELECT u,v,d_true,h FROM ground_obs WHERE session=?', (sess,)).fetchall()
        if not rows:
            raise SystemExit('no stored ground observations')
        obs, h = np.array([r[:3] for r in rows], float), rows[0][3]
    else:
        if args.height is None:
            raise SystemExit('--height H is required: optical-centre height above the floor [m]')
        h, sess = args.height, new_session('ground')
        ranges = tape_ranges(args)
        live = Live(args, 'ground setup', calib=calib)
        pts = []
        while True:
            img = live.frame()
            ranges = handle_click(live, img) or ranges
            blobs = find_tape_blobs(img, ranges, args.min_area)
            disp = img.copy()
            draw_blobs(disp, blobs)
            k = live.show(disp, [live.status_line(img), f'h = {h:.3f} m   floor points {len(pts)}',
                                 'ruler FLAT on the floor, pointing straight away from the camera',
                                 'space capture (enter forward distance of the NEAR tape)  u undo  c compute  q quit'])
            if k == ord(' '):
                cap = capture_tape(live, ranges, args.min_area)
                if cap is None:
                    print('two tape blobs not found')
                    continue
                d = live.ask(cap['frame'], 'forward distance of near tape from camera foot [m]')
                if d is None or d == 'esc':
                    continue
                near, far = sorted([cap['P1'], cap['P2']], key=lambda p: -p[1])
                new = [(near[0], near[1], d)] + ([] if args.near_only else [(far[0], far[1], d + args.length)])
                for u, v, dd in new:
                    db().execute('INSERT INTO ground_obs(session,ts,calib_id,h,d_true,u,v,brightness) '
                                 'VALUES(?,?,?,?,?,?,?,?)', (sess, time.time(), calib.id, h, dd, u, v,
                                                             cap['brightness']))
                db().commit()
                pts += new
                print(f'stored {len(new)} floor points (near tape at {d:.3f} m)')
            if k == ord('u') and pts:
                m = 1 if args.near_only else 2
                db().execute('DELETE FROM ground_obs WHERE id IN (SELECT id FROM ground_obs WHERE session=? '
                             'ORDER BY id DESC LIMIT ?)', (sess, m))
                db().commit()
                pts = pts[:-m]
            if k in (ord('c'), ord('q'), 27):
                break
        live.close()
        obs = np.array(pts, float)
        if len(obs) < 2:
            print('need at least 2 floor points')
            return
    g = fit_ground(obs, calib.K, h, args.sigma_h)
    tab = ground_row_table(calib.K, calib.height, h, g['theta'], args.sigma_h, g['se_theta'])
    g.update(calib_id=calib.id, session=sess, obs=obs, rows=tab['rows'], v_horizon=tab['v_horizon'])
    put_result('ground_cal', g, ref=calib.id)
    print(f"ground calibration (calib #{calib.id}): h = {h:.3f} m (measured, +- {args.sigma_h * 1e3:.0f} mm), "
          f"theta = {math.degrees(g['theta']):.3f} +- {math.degrees(g['se_theta']):.3f} deg, "
          f"residual rms {g['rms'] * 1e3:.1f} mm over {len(obs)} points")
    if 'joint' in g:
        j = g['joint']
        print(f"  joint fit: h = {j['h']:.3f} +- {j['se_h']:.3f} m, theta = {math.degrees(j['theta']):.3f} "
              f"+- {math.degrees(j['se_theta']):.3f} deg, retention of theta with h unknown "
              f"{j['rho_theta_given_h_unknown']:.3f}")
    print(f"  horizon row v = {tab['v_horizon']:.0f}. Distance uncertainty by image row:")
    print(f"  {'row v':>7}{'D [m]':>8}{'dD/dh':>8}{'dD/dth [m/deg]':>16}{'sigma_D [mm]':>14}{'rel':>7}")
    for r in tab['rows'][::6]:
        print(f"  {r['v']:7.0f}{r['D']:8.2f}{r['dD_dh']:8.2f}{r['dD_dtheta'] * math.pi / 180:16.4f}"
              f"{r['sigma_D'] * 1e3:14.1f}{r['rel'] * 100:6.1f}%")


def load_background(size):
    r = latest_result('background')
    if r is None:
        return None
    bg = decode_img(r['blob'])
    return bg if bg is not None and (bg.shape[1], bg.shape[0]) == tuple(size) else None


def capture_background(live, n=15):
    acc = None
    for _ in range(n):
        f = live.frame().astype(np.float32)
        acc = f if acc is None else acc + f
    bg = (acc / n).astype(np.uint8)
    put_result('background', dict(brightness=gray_mean(bg)), blob=png_bytes(bg))
    print('background (empty floor) captured')
    return bg


class CorrectorRuntime:
    def __init__(self):
        r = latest_result('corrector')
        self.ok = r is not None and any(r['data']['accepted'].values())
        if not self.ok:
            return
        torch = _torch()
        d = r['data']
        self.acc, self.norm = d['accepted'], d['norm']
        self.model = make_corrector()
        self.model.load_state_dict(torch.load(io.BytesIO(r['blob']), weights_only=True))
        self.model.eval()

    def __call__(self, feats):
        torch = _torch()
        x = (np.asarray(feats, np.float32) - self.norm['x_mean']) / self.norm['x_std']
        with torch.no_grad():
            out = self.model(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
        ys = np.array(self.norm['y_std'])
        mu, sd = out[:2] * ys, np.exp(0.5 * np.clip(out[2:], -8, 6)) * ys
        return mu, sd


def ground_loop(args, calib, mode):
    """Shared live loop of `ground` (mode 'measure') and `collect` (mode 'collect')."""
    g = load_ground(calib)
    sp = SegParams(args.tau_chroma, args.dark, args.bright, args.shadow, args.min_obj_area)
    corr = CorrectorRuntime() if mode == 'measure' else None
    rowtab = np.array([[r['v'], r['sigma_D']] for r in g['rows']])
    live = Live(args, 'ground' if mode == 'measure' else 'collect', calib=calib)
    img = live.frame()
    size = (img.shape[1], img.shape[0])
    bg = load_background(size)
    plan = [float(x) for x in args.distances.split(',')] if getattr(args, 'distances', None) else []
    n_saved = 0
    session = f"{time.strftime('%Y%m%d')}_{args.lighting}"
    while True:
        img = live.frame()
        disp = img.copy()
        lines = [live.status_line(img) + f"   h {g['h']:.3f} m  theta {math.degrees(g['theta']):.2f} deg"]
        objs = []
        if bg is None:
            lines.append('press b with the floor EMPTY to store the background')
        else:
            objs, shadow = detect_objects(img, bg, sp)
            disp[shadow] = (0.6 * disp[shadow] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
            for i, o in enumerate(objs):
                geo = object_geometry(o, calib.K, g)
                o['geo'] = geo
                x, y, w, hh = o['bbox']
                col = (0, 255, 0) if i == 0 else (0, 180, 255)
                cv2.drawContours(disp, [o['contour']], -1, col, 2)
                cv2.circle(disp, (int(o['u_b']), int(o['v_b'])), 4, (0, 0, 255), -1)
                txt = [f"D {geo['D']:.3f} m", f"W {geo['W'] * 100:.1f} cm  H {geo['H'] * 100:.1f} cm"]
                sig = np.interp(o['v_b'], rowtab[:, 0], rowtab[:, 1]) if len(rowtab) else float('nan')
                txt[0] += f" +- {sig * 1e3:.0f} mm"
                if corr is not None and corr.ok:
                    mu, sd = corr(object_features(o, geo, size))
                    if corr.acc.get('D'):
                        txt.append(f"D corr {geo['D'] + mu[0]:.3f} +- {sd[0]:.3f}")
                    if corr.acc.get('W'):
                        txt.append(f"W corr {(geo['W'] + mu[1]) * 100:.1f} cm")
                    o['corr'] = (mu, sd)
                draw_text(disp, txt, org=(x, max(16, y - 8 - 20 * (len(txt) - 1))), scale=0.5,
                          color=col[::-1] if i else (180, 255, 180))
            lines.append(f'{len(objs)} objects (largest in green; shadows tinted blue)')
        if mode == 'collect':
            nxt = f'   next D {plan[n_saved]:.3f} m' if n_saved < len(plan) else ''
            lines.append(f"object '{args.object}'  lighting '{args.lighting}'  saved {n_saved} frames{nxt}")
        lines.append('b background   space ' + ('log' if mode == 'measure' else 'save burst') + '   q quit')
        k = live.show(disp, lines)
        if k in (ord('q'), 27):
            break
        if k == ord('b'):
            bg = capture_background(live)
        if k == ord(' ') and objs:
            if mode == 'measure':
                o = objs[0]
                truth = None
                if args.truth:
                    d = live.ask(disp, 'true forward distance D [m] (Enter to skip)')
                    if d == 'esc':
                        continue
                    truth = dict(D=d, W=args.width, H=args.height) if d is not None else None
                est = dict(o['geo'], v_b=o['v_b'], u_b=o['u_b'], bbox=o['bbox'])
                if 'corr' in o:
                    est['corr_mu'], est['corr_sd'] = o['corr']
                db().execute('INSERT INTO measurements(ts,kind,calib_id,est,truth,brightness,lighting) '
                             'VALUES(?,?,?,?,?,?,?)', (time.time(), 'ground', calib.id, jdump(est),
                                                       jdump(truth) if truth else None, live.brightness,
                                                       args.lighting))
                db().commit()
                print(f"logged D {o['geo']['D']:.3f} W {o['geo']['W']:.3f} H {o['geo']['H']:.3f}"
                      + (f"  true D {truth['D']:.3f}" if truth else ''))
            else:
                d = plan[n_saved] if n_saved < len(plan) else live.ask(
                    disp, 'true forward distance D [m] (Enter: none, classifier only)')
                if d == 'esc':
                    continue
                d = float('nan') if d is None else d
                got = 0
                for _ in range(args.burst * 3):
                    fr = live.frame()
                    ob2, _ = detect_objects(fr, bg, sp)
                    if not ob2:
                        continue
                    o = ob2[0]
                    geo = object_geometry(o, calib.K, g)
                    feats = object_features(o, geo, size)
                    db().execute('INSERT INTO samples(ts,session,object,lighting,d_true,w_true,h_true,features,'
                                 'geo,brightness,crop,frame) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                                 (time.time(), session, args.object, args.lighting, d, args.width, args.height,
                                  jdump(feats), jdump(dict(geo, v_b=o['v_b'], u_b=o['u_b'], bbox=o['bbox'])),
                                  live.brightness, png_bytes(object_crop(fr, o)), jpg_bytes(fr, 90)))
                    got += 1
                    if got >= args.burst:
                        break
                db().commit()
                n_saved += 1
                print(f"saved {got} frames of '{args.object}' at D = {d:.3f} m ({args.lighting})")
                if plan and n_saved >= len(plan):
                    print('plan complete')
                    break
    live.close()


def cmd_ground(args):
    calib = load_calib(args.calib)
    if args.setup:
        ground_setup(args, calib)
    else:
        ground_loop(args, calib, 'measure')


def cmd_collect(args):
    objs = cfg_get('objects', {})
    size = objs.get(args.object, {})
    args.width = args.width if args.width is not None else size.get('W')
    args.height = args.height if args.height is not None else size.get('H')
    if args.width is not None and args.height is not None:
        objs[args.object] = dict(W=args.width, H=args.height)
        cfg_set('objects', objs)
    else:
        print(f"note: no true size for '{args.object}' (--width/--height); its frames serve the classifier only")
    ground_loop(args, load_calib(args.calib), 'collect')
    n = db().execute('SELECT COUNT(*), COUNT(DISTINCT object) FROM samples').fetchone()
    print(f'dataset: {n[0]} frames of {n[1]} objects (target: 500 frames, >= 10 objects; handoff 4)')


# ----------------------------------------------------------------------------
# learning (handoff 4, 5)
# ----------------------------------------------------------------------------
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        _TORCH = torch
    return _TORCH


def mlp(sizes, flatten=False):
    nn = _torch().nn
    layers = [nn.Flatten()] if flatten else []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def make_corrector(n_in=None, n_out=2):
    """features -> 32 -> 32 -> (residual means, log-variances); 12 -> 2 for the ground corrector."""
    return mlp([n_in or len(FEATURES), 32, 32, 2 * n_out])


def make_cls_mlp(in_shape, n_cls):
    return mlp([int(np.prod(in_shape)), 64, 32, n_cls], flatten=True)


def make_cls_cnn(in_shape, n_cls):
    nn = _torch().nn
    c, h, w = in_shape
    return nn.Sequential(nn.Conv2d(c, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(16 * (h // 4) * (w // 4), n_cls))


def state_bytes(obj):
    bio = io.BytesIO()
    _torch().save(obj, bio)
    return bio.getvalue()


def gauss_nll(out, y):
    k = out.shape[1] // 2
    mu, lv = out[:, :k], out[:, k:].clamp(-8, 6)
    return (0.5 * (lv + (y - mu) ** 2 * _torch().exp(-lv))).mean()


def train_corrector_once(Xtr, Ytr, Xva, Yva, epochs, lr, seed, fixed_epochs=None):
    torch = _torch()
    torch.manual_seed(seed)
    m = make_corrector(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    Xv, Yv = (torch.tensor(Xva), torch.tensor(Yva)) if Xva is not None else (None, None)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state = float('inf'), 0, None
    for ep in range(fixed_epochs or epochs):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            gauss_nll(m(Xt[idx]), Yt[idx]).backward()
            opt.step()
        if Xv is not None and (ep % 5 == 4 or ep == epochs - 1):
            m.eval()
            with torch.no_grad():
                vl = float(gauss_nll(m(Xv), Yv))
            if vl < best:
                best, best_ep, best_state = vl, ep + 1, {k: v.clone() for k, v in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, best_ep or (fixed_epochs or epochs)


def load_samples(need_truth=True):
    q = 'SELECT id,object,lighting,session,d_true,w_true,h_true,features,geo FROM samples'
    rows = db().execute(q).fetchall()
    out = []
    for r in rows:
        if need_truth and (r[4] is None or not np.isfinite(r[4]) or r[5] is None):
            continue
        out.append(dict(id=r[0], object=r[1], lighting=r[2], session=r[3], d=r[4], w=r[5], h=r[6],
                        x=json.loads(r[7]), geo=json.loads(r[8])))
    return out


ROOM_FEATURES = ['L', 'W', 'H', 'log_views', 'iou', 'box', 'round', 'dome', 'distance', 'elevation',
                 'cam_height', 'cut', 'rho_L', 'rho_W', 'rho_H', 'has_info', 'log_sigma_own', 'log_sigma_other']


def room_features(o):
    """Corrector inputs for one measured object (a room_record entry)."""
    cam, c = np.asarray(o['cam'], float), np.asarray(o['centre'], float)
    horiz = float(np.hypot(*(cam[:2] - c)))
    return [o['L'], o['W'], o['H'], math.log(o.get('n_views', 1)), o['iou'], o['shape'] == 'box',
            o['shape'] == 'round', o['shape'] == 'dome', math.hypot(horiz, cam[2]), math.atan2(cam[2], horiz),
            cam[2], float(o['cut'])] + ([o['info'][d][0] for d in 'LWH'] + [1.0] if o.get('info') else [0.0] * 4) + fusion_features(o)


def fusion_features(o):
    """What the fusion knew: the outline noise [px] (log) of this camera's own view and the mean of the other cameras' views
    (the same when there is only one view). The corrector learns how far to trust a size from how the cameras agree."""
    sg = [float(v) for v in (o.get('sigmas') or [1.0])]
    return [math.log(sg[0]), math.log(float(np.mean(sg[1:])) if len(sg) > 1 else sg[0])]


def log_key(truth, est):
    """Identity of a logged measurement: the object and every measured number. Pressing space
    again before the measurement updates logs the same numbers twice."""
    return (truth.get('object'), tuple(round(o[d] * 1e4) for o in est['objects'] for d in 'LWH'))


def log_camera(calib_id):
    """Which camera a room log came from, by its calibration: the ESP32-CAM streams 800 px wide, the phone's browser stream
    has source 'webphone', anything else is the phone over USB."""
    r = db().execute("SELECT width, meta FROM calibrations WHERE id = ?", (calib_id,)).fetchone()
    if r is None:
        return 'phone-usb'
    return 'esp32' if r[0] == 800 else ('phone-wifi' if json.loads(r[1] or '{}').get('source') == 'webphone' else 'phone-usb')


def log_tags():
    """{log id: (camera, setup, object, use, note)}: the labels of the recent logs (table log_tags). A log without a label
    counts as a normal log in the room and gets its camera from its calibration."""
    db().execute("CREATE TABLE IF NOT EXISTS log_tags(id INTEGER PRIMARY KEY, camera TEXT, setup TEXT, object TEXT, use INTEGER, note TEXT)")
    return {r[0]: r[1:] for r in db().execute("SELECT id, camera, setup, object, use, note FROM log_tags")}


def log_label(mid, calib_id, tags):
    """(camera, setup, object, use, note) of one log: its row in log_tags, else a normal log in the room whose camera follows
    from its calibration."""
    cam, setup, obj, use, note = tags.get(mid, (None, 'room', None, 1, ''))
    if cam is None and db().execute("SELECT 1 FROM results WHERE kind='room_raw2' AND (ref=? OR ref=?)", (mid, str(mid))).fetchone():
        cam = 'rig'  # both cameras were logged: the phone's estimate already holds the ESP32-CAM's views
    return cam or log_camera(calib_id), setup, obj, use, note


def room_samples(min_views=1, setup='room', camera=None):
    """Logged room measurements with a named object: per log, the measured object whose size is
    closest to the truth (in case something else was in view). Labelled logs that are marked unusable are skipped, and
    setup ('room', 'outside room' or None for both) and camera filter the rest. camera None is the phone's logs: the ESP32-CAM
    alone ('esp32') and the rig ('rig', both cameras' views fused in one estimate) are separate datasets, asked for by name."""
    out, seen, tags = [], set(), log_tags()
    for mid, est, truth, cid in db().execute("SELECT id, est, truth, calib_id FROM measurements WHERE kind='room' AND truth IS NOT NULL"):
        cam, st, _, use, _ = log_label(mid, cid, tags)
        if not use or (setup and st != setup) or (camera and cam != camera) or (camera is None and cam in ('rig', 'esp32')):
            continue  # three datasets: phone (the default), ESP alone (--camera esp32), rig = both cameras fused (--camera rig)
        est, truth = json.loads(est), json.loads(truth)
        if log_key(truth, est) in seen:  # a repeat adds nothing
            continue
        seen.add(log_key(truth, est))
        objs = [o for o in est['objects'] if 'cam' in o]
        if not truth.get('object') or not objs:
            continue
        t = np.array([truth['L'], truth['W'], truth['H']])
        o = min(objs, key=lambda o: np.abs(np.array([o['L'], o['W'], o['H']]) - t).sum())
        if o.get('n_views', 1) < min_views or o['shape'] not in ROOM_SHAPES:  # e.g. the removed harmonic
            continue
        out.append(dict(id=mid, camera=cam, object=truth['object'], x=room_features(o),
                        y=t - np.array([o['L'], o['W'], o['H']])))
    return out


def room_std(X):
    """Feature scales; a feature that never varied in training is left unscaled (not blown up)."""
    s = X.std(0)
    return np.where(s > 1e-6, s, 1.0).astype(np.float32)


def cmd_train_corrector_room(args):
    """Residual corrector for `room` (handoff 4): truth minus geometry for L, W, H, from logged
    measurements of known objects; folds hold out whole objects. Used live only for the sizes it
    improves on objects it never saw."""
    torch = _torch()
    S = room_samples(args.min_views, None if args.setup == 'all' else args.setup, args.camera)
    objects = sorted({s['object'] for s in S})
    print(f'room corrector data: {len(S)} logs with {args.min_views}+ views of {len(objects)} objects {objects}')
    by = collections.Counter(s['camera'] for s in S)
    print('  by camera: ' + ', '.join(f'{k} {v}' for k, v in sorted(by.items())) + f'   (setup: {args.setup}, camera: {args.camera or "phone"})')
    if len(objects) < 3:
        raise SystemExit('need logs of at least 3 known objects: `room --object NAME`, press space from many angles')
    X = np.array([s['x'] for s in S], np.float32)
    Y = np.array([s['y'] for s in S], np.float32)
    obj = np.array([s['object'] for s in S])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(objects)
    k = min(args.folds, len(objects))
    oof_mu, oof_sd, eps = np.zeros_like(Y), np.zeros_like(Y), []
    for fi in range(k):
        te = np.isin(obj, perm[fi::k])
        idx = np.flatnonzero(~te)
        va = np.zeros(len(S), bool)
        va[rng.choice(idx, max(1, len(idx) // 7), replace=False)] = True
        tr = ~te & ~va
        xm, xs, ys = X[tr].mean(0), room_std(X[tr]), Y[tr].std(0) + 1e-6
        m, ep = train_corrector_once(np.clip((X[tr] - xm) / xs, -5, 5), Y[tr] / ys,
                                     np.clip((X[va] - xm) / xs, -5, 5), Y[va] / ys,
                                     args.epochs, args.lr, args.seed + fi)
        eps.append(ep)
        with torch.no_grad():
            out = m(torch.tensor(np.clip((X[te] - xm) / xs, -5, 5))).numpy()
        oof_mu[te], oof_sd[te] = out[:, :3] * ys, np.exp(0.5 * np.clip(out[:, 3:], -8, 6)) * ys
        print(f'  fold {fi + 1}/{k}: held out {list(perm[fi::k])}, best epoch {ep}')
    geo, cor = np.abs(Y).mean(0), np.abs(Y - oof_mu).mean(0)
    cover = (np.abs((Y - oof_mu) / oof_sd) < 1).mean(0)
    accepted = {d: bool(cor[i] < 0.9 * geo[i]) for i, d in enumerate('LWH')}  # 10 % better, not noise
    print(f"{'':<4}{'geometry [mm]':>15}{'corrected [mm]':>16}{'within 1 sigma':>16}")
    for i, d in enumerate('LWH'):
        print(f"{d:<4}{geo[i] * 1e3:>15.1f}{cor[i] * 1e3:>16.1f}{cover[i]:>16.2f}   "
              f"{'ACCEPTED' if accepted[d] else 'rejected'}")
    for o in objects:  # per object, held out
        mo = obj == o
        print(f"  {o:<10} n={mo.sum():<4} geo {' '.join(f'{v * 1e3:5.1f}' for v in np.abs(Y[mo]).mean(0))}"
              f"   corrected {' '.join(f'{v * 1e3:5.1f}' for v in np.abs(Y[mo] - oof_mu[mo]).mean(0))} mm")
    xm, xs, ys = X.mean(0), room_std(X), Y.std(0) + 1e-6
    ep = int(np.median(eps))
    final, _ = train_corrector_once(np.clip((X - xm) / xs, -5, 5), Y / ys, None, None, ep, args.lr, args.seed,
                                    fixed_epochs=ep)
    rid = put_result('room_corrector', dict(camera=args.camera if args.camera in ('esp32', 'rig') else None, features=ROOM_FEATURES, n=len(S), objects=objects, epochs=ep,
                                            mae_geo=geo, mae_cor=cor, coverage_1sigma=cover, accepted=accepted,
                                            norm=dict(x_mean=xm, x_std=xs, y_std=ys, x_min=X.min(0), x_max=X.max(0))),
                     blob=state_bytes(final.state_dict()))
    print(f'stored room corrector as results #{rid}; `room` applies it to the accepted sizes')


def rig_pick(res, T):
    """The measured object whose size is closest to the truth T (L, W, H), or None."""
    objs = [o for o in (res or {}).get('objects', []) if not o.get('cut')] or (res or {}).get('objects', [])
    return min(objs, key=lambda o: np.abs(np.array([o['L'], o['W'], o['H']]) - T).sum()) if objs else None


def rig_refit(img1, img2, K1, d1, K2, d2, shape, guide=True, corrector=False, rounds=2):
    """The two stored frames of a rig log refitted offline: (phone alone, ESP alone, both together), each a Room result.
    `guide` False: the ESP gets no prediction from the phone (its own detection only)."""
    ns = lambda: argparse.Namespace(shape=shape, object_class=None, truth=None, seg='room', corrector=corrector)

    def room(img, K, d):
        r = Room(ns(), (img.shape[1], img.shape[0]), store=False)
        r.K, r.dist = np.array(K, float), np.array(d, float)
        if not corrector:
            r.corrector = False
        return r
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        alone1 = room(img1, K1, d1).process(img1, None, ())
        alone2 = room(img2, K2, d2).process(img2, None, ())
        A, B = room(img1, K1, d1), room(img2, K2, d2)
        ra = rb = None
        for _ in range(rounds):
            ra = A.process(img1, None, rig_views(B, rb))
            rb = B.process(img2, None, rig_views(A, ra), rig_priors(A, ra) if guide else ())
    return alone1, alone2, ra, rb


def cmd_rig_eval(args):
    """Is the rig better than either camera? Every log with both frames and a true size is refitted offline from its stored
    pictures: the phone alone, the ESP32-CAM alone, both together without and with the phone's guidance (and with the
    latest corrector, --corrected). Mean absolute size error against the truth, in mm. The corrector has seen these logs
    when it was trained: its honest held-out error is the one `train-corrector` prints."""
    rows = []
    for mid, est, truth in db().execute("SELECT id, est, truth FROM measurements WHERE kind='room' AND truth IS NOT NULL ORDER BY id"):
        if mid < args.min_id:
            continue
        r1 = db().execute("SELECT blob FROM results WHERE kind='room_raw' AND (ref=? OR ref=?)", (mid, str(mid))).fetchone()
        r2 = db().execute("SELECT data, blob FROM results WHERE kind='room_raw2' AND (ref=? OR ref=?)", (mid, str(mid))).fetchone()
        t = json.loads(truth)
        if r1 and r2 and t.get('object'):
            rows.append((mid, json.loads(est), t, r1[0], json.loads(r2[0]), r2[1]))
    if not rows:
        raise SystemExit('no rig logs with a true size yet: `room --url phone --name --object NAME`, press space with the ESP32-CAM streaming')
    if args.last:
        rows = rows[-args.last:]
    names = ['phone alone', 'ESP alone', 'both, no guidance', 'both, phone guides ESP'] + (['both + corrector'] if args.corrected else [])
    errs = {n: [] for n in names}
    per_obj = collections.defaultdict(lambda: {n: [] for n in names})
    print(f'{len(rows)} rig logs, refitting from the stored frames ...')
    for k, (mid, est, t, b1, d2, b2) in enumerate(rows):
        T = np.array([t['L'], t['W'], t['H']])
        o0 = rig_pick(dict(objects=[o for o in est['objects'] if 'cam' in o]), T)
        k2 = d2.get('K') or (d2.get('est') or {}).get('K')
        if k2 is not None:
            K2, dd2 = k2, d2.get('dist') or d2['est']['dist']
        else:  # logged before the ESP's K was saved: the calibration the ESP's Room started from
            fb = fallback_calib('esp32')
            K2, dd2 = (fb.K.tolist(), fb.dist.tolist()) if fb is not None else (None, None)
        if o0 is None or not est.get('K') or K2 is None:
            continue
        img1, img2 = decode_img(b1), decode_img(b2)
        if mid <= args.flip_until:
            img2 = cv2.flip(img2, 1)
        shape = o0['shape'] if o0['shape'] in ROOM_SHAPES else 'round'
        K1, dd1 = est['K'], est['dist']
        a1, a2, f1, _ = rig_refit(img1, img2, K1, dd1, K2, dd2, shape, guide=False)
        _, _, g1, _ = rig_refit(img1, img2, K1, dd1, K2, dd2, shape, guide=True)
        got = [rig_pick(a1, T), rig_pick(a2, T), rig_pick(f1, T), rig_pick(g1, T)]
        if args.corrected:
            _, _, c1, _ = rig_refit(img1, img2, K1, dd1, K2, dd2, shape, guide=True, corrector=True)
            got.append(rig_pick(c1, T))
        for n, o in zip(names, got):
            e = np.abs(np.array([o['L'], o['W'], o['H']]) - T) if o is not None else None
            errs[n].append(e)
            per_obj[t['object']][n].append(e)
        print(f"  {k + 1}/{len(rows)} log {mid} {t['object']}: " + '  '.join(
            f"{n.split(',')[0] if ',' in n else n} " + ('-' if o is None else f"{np.mean(np.abs(np.array([o['L'], o['W'], o['H']]) - T)) * 1e3:.1f}")
            for n, o in zip(names, got)), flush=True)

    def mae(lst):
        ok = [e for e in lst if e is not None]
        return (np.mean(ok, 0) * 1e3 if ok else np.full(3, np.nan)), len(ok), len(lst)
    print(f"\n{'':<26}{'found':>8}{'L':>8}{'W':>8}{'H':>8}{'mean':>8}   [mm absolute error]")
    means = {}
    for n in names:
        m, ok, tot = mae(errs[n])
        means[n] = float(np.mean(m))
        print(f'{n:<26}{ok:>4}/{tot:<3}{m[0]:>8.1f}{m[1]:>8.1f}{m[2]:>8.1f}{np.mean(m):>8.1f}')
    for o, d in sorted(per_obj.items()):
        print(f'  {o:<18}n={len(d[names[0]]):<3}' + '  '.join(f'{np.mean(mae(d[n])[0]):5.1f}' for n in names))
    best_single = min(means['phone alone'], means['ESP alone'])
    for n in names[2:]:
        print(f"{n}: {means[n]:.1f} mm against the best single camera {best_single:.1f} mm -> "
              + ('BETTER' if means[n] < best_single else 'not better yet'))


def load_room_corrector(camera=None):
    """Latest room corrector of one dataset (None: the phone's; 'esp32'; 'rig') as a function (record -> (residuals, sigmas))
    and its accepted sizes. A corrector trained on another dataset never replaces it."""
    row = next(((d, b) for d, b in db().execute("SELECT data, blob FROM results WHERE kind='room_corrector' ORDER BY id DESC")
                if json.loads(d).get('camera') == camera), None)
    if not row:
        return None
    torch = _torch()
    d = json.loads(row[0])
    m = make_corrector(len(d['features']), 3)
    m.load_state_dict(torch.load(io.BytesIO(row[1])))
    m.eval()
    xm, xs, ys = (np.array(d['norm'][k], np.float32) for k in ('x_mean', 'x_std', 'y_std'))
    lo, hi = (np.array(d['norm'][k], np.float32) if k in d['norm'] else None for k in ('x_min', 'x_max'))

    def apply(o):
        x = np.array(room_features(o)[:len(d['features'])], np.float32)  # older correctors: fewer inputs
        if lo is None or ((x < lo - 0.1 * (hi - lo)) | (x > hi + 0.1 * (hi - lo))).any():
            return np.zeros(3), np.full(3, np.nan)  # unlike anything it was trained on: no correction
        with torch.no_grad():
            z = np.clip((x[None] - xm) / xs, -5, 5)  # no extrapolating far
            out = m(torch.tensor(z)).numpy()[0]
        return out[:3] * ys, np.exp(0.5 * np.clip(out[3:], -8, 6)) * ys
    # the acceptance rule as it stands now (10 % better on held-out objects), also for older correctors
    return apply, {k: bool(c < 0.9 * g) for k, c, g in zip('LWH', d['mae_cor'], d['mae_geo'])}


SIZER_SHAPES = ['round', 'box', 'dome', 'cylinder', 'sphere', 'revolve']
SIZER_PER_SHAPE = ['logL', 'logW', 'logH', 'iou', 'rhoL', 'rhoW', 'rhoH', 'logsdL', 'logsdW', 'logsdH']
SIZER_FEATURES = [f'{s}_{k}' for s in SIZER_SHAPES for k in SIZER_PER_SHAPE] + \
    ['log_views', 'distance', 'elevation', 'cam_height']


def sizer_features(views, K, dist, up, warm=None):
    """Every shape fitted blindly to the same outlines (views: (hull, pose), the first is the
    current one): log sizes, fit overlap, retentions and log standard errors per shape, plus the
    view geometry. The sizer network decides from these which numbers to trust.
    Returns (feature vector, {shape: params} for warm starts next time) or (None, {})."""
    hull, pose = views[0]
    warm = warm or {}
    base = fit_shape(hull, pose, K, dist, up, 'round', [warm['round']] if 'round' in warm else (), views[1:])
    if base is None:
        return None, {}
    x, ps = [], {}
    for sh in SIZER_SHAPES:
        f = base if sh == 'round' else fit_shape(hull, pose, K, dist, up, sh, [warm.get(sh, base['p'])], views[1:])
        if f is None:
            f = base
        ps[sh] = f['p']
        try:
            info = size_information(f['p'], sh, up, views, K, dist, sigma_px=None)
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            info = {d: (0.0, 1.0) for d in 'LWH'}
        sz = [max(f[d], 5e-4) for d in 'LWH']
        x += [math.log(v) for v in sz] + [f['iou']] + [info[d][0] for d in 'LWH'] + \
             [math.log(min(max(info[d][1], 1e-5), 1.0)) for d in 'LWH']
    cam = -pose['R'].T @ pose['t'] if 'R' in pose else -cv2.Rodrigues(pose['rvec'])[0].T @ np.ravel(pose['tvec'])
    horiz = float(np.hypot(*(cam[:2] - base['p'][:2])))
    x += [math.log(len(views)), math.hypot(horiz, cam[2]), math.atan2(abs(cam[2]), horiz), abs(cam[2])]
    return np.array(x, np.float32), ps


def _pose_from(v):
    rvec, tvec = np.array(v['rvec'], float).reshape(3, 1), np.array(v['tvec'], float).reshape(3, 1)
    R = cv2.Rodrigues(rvec)[0]
    out = dict(rvec=rvec, tvec=tvec, R=R, t=tvec.ravel())
    if 'K' in v:  # a view from the second camera
        out.update(K=np.array(v['K'], float), dist=np.array(v['dist'], float))
    return out


def cmd_sizer_features(args):
    """Six-shape features for every logged measurement of a known object that has none yet, cached
    in sizer_feats: each log is processed once, so new photos only cost their own time. Logs that
    stored their outlines are refitted from them (all their views); older logs from their saved
    frame (one view). Stop any time (Ctrl+C): it continues where it stopped."""
    rows = db().execute("SELECT m.id, m.est, m.truth, m.calib_id FROM measurements m LEFT JOIN sizer_feats s "
                        "ON s.measurement = m.id WHERE m.kind='room' AND m.truth IS NOT NULL "
                        "AND s.measurement IS NULL ORDER BY m.id DESC").fetchall()
    rows = [r for r in rows if json.loads(r[2]).get('object')]
    keys = {}  # repeats: the same object and numbers as another log
    for r in sorted(rows, key=lambda r: r[0]):
        keys.setdefault(log_key(json.loads(r[2]), json.loads(r[1])), r[0])
    done_keys = set()
    for mid, est, tr in db().execute("SELECT m.id, m.est, m.truth FROM measurements m JOIN sizer_feats s "
                                     "ON s.measurement = m.id"):
        done_keys.add(log_key(json.loads(tr), json.loads(est)))
    n0 = len(rows)
    rows = [r for r in rows if keys[log_key(json.loads(r[2]), json.loads(r[1]))] == r[0]
            and log_key(json.loads(r[2]), json.loads(r[1])) not in done_keys]
    print(f'{n0 - len(rows)} repeats of other logs skipped')
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} logs without six-shape features (newest first)')
    t0, done = time.time(), 0
    for mid, est, truth, calib_id in rows:
        try:
            x, source = sizer_features_for_log(mid, est, truth, calib_id)
        except Exception as e:  # one bad log must not stop a long run
            x, source = None, f'failed: {type(e).__name__}'
        db().execute('INSERT OR REPLACE INTO sizer_feats(measurement, ts, source, feats) VALUES(?,?,?,?)',
                     (mid, time.time(), source or 'failed', None if x is None else jdump(x)))
        db().commit()
        done += 1
        if done % 10 == 0 or done == len(rows):
            el = time.time() - t0
            print(f'  {done}/{len(rows)}  {el / done:.1f} s per log, about {el / done * (len(rows) - done) / 60:.0f} min left',
                  flush=True)


def sizer_features_for_log(mid, est, truth, calib_id):
    """Six-shape features of one logged measurement: (vector or None, source)."""
    est, t = json.loads(est), json.loads(truth)
    T = np.array([t['L'], t['W'], t['H']])
    objs = est['objects']
    o = min(objs, key=lambda o: np.abs(np.array([o['L'], o['W'], o['H']]) - T).sum()) if objs else None
    x, source = None, ''
    if o is not None and o.get('views') and est.get('K'):
        views = [(np.array(v['hull'], np.float32), _pose_from(v)) for v in o['views']]
        x, _ = sizer_features(views, np.array(est['K']), np.array(est['dist']), o.get('up', 1.0))
        source = f'{len(views)} stored views'
    else:  # older log: redo the frame (one view)
        row = db().execute("SELECT blob FROM results WHERE kind='room_raw' AND (ref=? OR ref=?)",
                           (mid, str(mid))).fetchone()
        if row:
            img = decode_img(row[0])
            cal = load_calib(calib_id, required=False) if calib_id else None  # the run's own calibration
            if cal is not None and (cal.width, cal.height) != (img.shape[1], img.shape[0]):
                cal = None
            room = Room(argparse.Namespace(shape='round', object_class=None, truth=None, seg='room'),
                        (img.shape[1], img.shape[0]), cal, store=False)
            room.corrector = False
            with contextlib.redirect_stdout(io.StringIO()):
                res = room.process(img, None)
            if res['objects']:
                ob = min(res['objects'], key=lambda q: np.abs(np.array([q['L'], q['W'], q['H']]) - T).sum())
                x, _ = sizer_features(ob['views'][:1], room.K, room.dist, ob['up'])
                source = 'frame, 1 view'
    return x, source


def sizer_samples():
    """Cached features with their truth and the logged (chosen-shape) geometry."""
    out, seen = [], set()
    for mid, feats, est, truth in db().execute(
            "SELECT s.measurement, s.feats, m.est, m.truth FROM sizer_feats s JOIN measurements m "
            "ON m.id = s.measurement WHERE s.feats IS NOT NULL"):
        t, est = json.loads(truth), json.loads(est)
        if not t.get('object') or not est['objects'] or log_key(t, est) in seen:
            continue
        seen.add(log_key(t, est))
        T = np.array([t['L'], t['W'], t['H']])
        o = min(est['objects'], key=lambda o: np.abs(np.array([o['L'], o['W'], o['H']]) - T).sum())
        if o['shape'] not in ROOM_SHAPES:  # logged with a removed shape: its geometry is no baseline
            continue
        out.append(dict(id=mid, object=t['object'], x=np.array(json.loads(feats), np.float32), truth=T,
                        geo=np.array([o['L'], o['W'], o['H']])))
    return out


def cmd_train_sizer(args):
    """The sizer: an MLP from the six-shape features to the true L, W, H (log sizes, Gaussian
    NLL, so it also states its uncertainty). Folds hold out whole objects; a size is used live only
    if the sizer beats the chosen-shape geometry by 10 % on objects it never saw. Seconds to minutes
    on cached features; --warm continues from the last sizer instead of starting fresh."""
    torch = _torch()
    S = sizer_samples()
    objects = sorted({s['object'] for s in S})
    print(f'sizer data: {len(S)} logs of {len(objects)} objects (run `sizer-features` to add new logs)')
    if len(objects) < 3:
        raise SystemExit('need six-shape features of at least 3 known objects: run `sizer-features` first')
    X = np.nan_to_num(np.array([s['x'] for s in S], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.log(np.maximum(np.array([s['truth'] for s in S]), 5e-4)).astype(np.float32)
    G = np.array([s['geo'] for s in S])
    obj = np.array([s['object'] for s in S])
    prev = db().execute("SELECT data, blob FROM results WHERE kind='sizer' ORDER BY id DESC LIMIT 1").fetchone()
    warm = None
    if args.warm and prev and len(json.loads(prev[0])['features']) == X.shape[1]:
        warm = torch.load(io.BytesIO(prev[1]))
        nm = json.loads(prev[0])['norm']
        xm, xs, ym, ys = (np.array(nm[k], np.float32) for k in ('x_mean', 'x_std', 'y_mean', 'y_std'))
        print(f'continuing from sizer results #{db().execute("SELECT id FROM results WHERE kind=? ORDER BY id DESC", ("sizer",)).fetchone()[0]}')
    else:
        xm, xs, ym, ys = X.mean(0), room_std(X), Y.mean(0), Y.std(0) + 1e-6

    # checkpoint: survives a power cut; the same command resumes if the data and settings match
    ck_path = os.path.join(DATA, 'checkpoints', 'train_sizer.pt')
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    fp = hashlib.sha256(jdump(dict(ids=sorted(int(s['id']) for s in S), epochs=args.epochs, lr=args.lr,
                                   folds=args.folds, seed=args.seed, warm=bool(warm is not None))).encode()).hexdigest()
    state = None
    if os.path.exists(ck_path) and not args.fresh:
        try:
            state = torch.load(ck_path, weights_only=False)
        except Exception:  # cut off while writing: the previous file was replaced atomically, so this is rare
            state = None
        if state is not None and state.get('fp') != fp:
            print('checkpoint is for other data or settings: starting fresh')
            state = None
        elif state is not None:
            print(f"resuming from the checkpoint: {state['fold']} fold(s) done"
                  + (f", fold {state['fold'] + 1} at epoch {state['cur']['ep']}" if state.get('cur') else ''))
    if state is None:
        state = dict(fp=fp, fold=0, pred=np.zeros_like(Y), sd=np.zeros_like(Y), eps=[], cur=None)

    def save():
        tmp = ck_path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, ck_path)  # atomic: a cut during the write leaves the old checkpoint

    def train(tr, va, seed, epochs, fixed=None, tag=0):
        torch.manual_seed(seed)
        m = make_corrector(X.shape[1], 3)
        if warm is not None:
            m.load_state_dict(warm)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=1e-4)
        Xt = torch.tensor(np.clip((X[tr] - xm) / xs, -5, 5))
        Yt = torch.tensor((Y[tr] - ym) / ys)
        Xv = torch.tensor(np.clip((X[va] - xm) / xs, -5, 5)) if va is not None else None
        Yv = torch.tensor((Y[va] - ym) / ys) if va is not None else None
        best, best_state, best_ep = float('inf'), None, epochs
        g = torch.Generator().manual_seed(seed)
        start = 0
        cur = state.get('cur')
        if cur is not None and cur['tag'] == tag:  # resume inside this run
            m.load_state_dict(cur['model'])
            opt.load_state_dict(cur['opt'])
            g.set_state(cur['gen'])
            best, best_state, best_ep, start = cur['best'], cur['best_state'], cur['best_ep'], cur['ep']
        for ep in range(start, fixed or epochs):
            if ep % 10 == 0 and ep > start:
                state['cur'] = dict(tag=tag, ep=ep, model=m.state_dict(), opt=opt.state_dict(), gen=g.get_state(),
                                    best=best, best_state=best_state, best_ep=best_ep)
                save()
            m.train()
            for idx in torch.randperm(len(Xt), generator=g).split(64):
                opt.zero_grad()
                gauss_nll(m(Xt[idx]), Yt[idx]).backward()
                opt.step()
            if Xv is not None and ep % 5 == 4:
                m.eval()
                with torch.no_grad():
                    vl = float(gauss_nll(m(Xv), Yv))
                if vl < best:
                    best, best_ep, best_state = vl, ep + 1, {k: v.clone() for k, v in m.state_dict().items()}
        if best_state is not None:
            m.load_state_dict(best_state)
        m.eval()
        return m, best_ep

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(objects)
    k = min(args.folds, len(objects))
    splits = []  # drawn up front, so a resumed run uses the same folds
    for fi in range(k):
        te = np.isin(obj, perm[fi::k])
        idx = np.flatnonzero(~te)
        va = np.zeros(len(S), bool)
        va[rng.choice(idx, max(1, len(idx) // 7), replace=False)] = True
        splits.append((te, va))
    pred, sd, eps = state['pred'], state['sd'], state['eps']
    for fi in range(state['fold'], k):
        te, va = splits[fi]
        m, ep = train(~te & ~va, va, args.seed + fi, args.epochs, tag=fi)
        eps.append(ep)
        with torch.no_grad():
            out = m(torch.tensor(np.clip((X[te] - xm) / xs, -5, 5))).numpy()
        pred[te] = out[:, :3] * ys + ym
        sd[te] = np.exp(0.5 * np.clip(out[:, 3:], -8, 6)) * ys
        state.update(fold=fi + 1, cur=None)
        save()
        print(f'  fold {fi + 1}/{k}: held out {[str(o) for o in perm[fi::k]]}, best epoch {ep}')
    T = np.exp(Y)
    err_geo, err_siz = np.abs(G - T).mean(0), np.abs(np.exp(pred) - T).mean(0)
    cover = (np.abs((Y - pred) / sd) < 1).mean(0)
    accepted = {d: bool(err_siz[i] < 0.9 * err_geo[i]) for i, d in enumerate('LWH')}
    print(f"{'':<4}{'chosen shape [mm]':>19}{'sizer [mm]':>12}{'within 1 sigma':>16}")
    for i, d in enumerate('LWH'):
        print(f"{d:<4}{err_geo[i] * 1e3:>19.1f}{err_siz[i] * 1e3:>12.1f}{cover[i]:>16.2f}   "
              f"{'ACCEPTED' if accepted[d] else 'rejected'}")
    for o in objects:
        mo = obj == o
        print(f"  {o:<17} n={mo.sum():<5} chosen {' '.join(f'{v * 1e3:5.1f}' for v in np.abs(G[mo] - T[mo]).mean(0))}"
              f"   sizer {' '.join(f'{v * 1e3:5.1f}' for v in np.abs(np.exp(pred[mo]) - T[mo]).mean(0))} mm")
    ep = int(np.median(eps))
    final, _ = train(np.ones(len(S), bool), None, args.seed, ep, fixed=ep, tag=k)
    rid = put_result('sizer', dict(features=SIZER_FEATURES, n=len(S), objects=objects, epochs=ep,
                                   mae_geo=err_geo, mae_sizer=err_siz, coverage_1sigma=cover, accepted=accepted,
                                   norm=dict(x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, x_min=X.min(0), x_max=X.max(0))),
                     blob=state_bytes(final.state_dict()))
    if os.path.exists(ck_path):
        os.remove(ck_path)  # finished: nothing to resume
    print(f'stored sizer as results #{rid}; `room --sizer` uses it for the accepted sizes')


def load_sizer():
    """Latest sizer as a function (features -> (sizes, sigmas) [m]) and its accepted sizes."""
    row = db().execute("SELECT data, blob FROM results WHERE kind='sizer' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    torch = _torch()
    d = json.loads(row[0])
    m = make_corrector(len(d['features']), 3)
    m.load_state_dict(torch.load(io.BytesIO(row[1])))
    m.eval()
    xm, xs, ym, ys, lo, hi = (np.array(d['norm'][k], np.float32)
                              for k in ('x_mean', 'x_std', 'y_mean', 'y_std', 'x_min', 'x_max'))

    def apply(x):
        x = np.nan_to_num(np.asarray(x, np.float32))
        if ((x < lo - 0.1 * (hi - lo)) | (x > hi + 0.1 * (hi - lo))).any():
            return None  # unlike anything it was trained on
        with torch.no_grad():
            out = m(torch.tensor(np.clip((x[None] - xm) / xs, -5, 5))).numpy()[0]
        mu = out[:3] * ys + ym
        return np.exp(mu), np.exp(mu) * np.exp(0.5 * np.clip(out[3:], -8, 6)) * ys
    return apply, d['accepted']


OUTLINE_SIZE = 128
OUTLINE_CHANNELS = ['red', 'green', 'blue', 'shade-free', 'expected colour', 'square border', 'hidden squares',
                    'current outline']
OUTLINE_CACHE = os.path.join(DATA, 'outline_cache')


def outline_inputs(crop, flat, occ, pat, border, cur, size=OUTLINE_SIZE):
    """The outline network's input for one object: 8 channels, size x size. The room's prediction of
    every square (expected colour, borders) sits next to what the camera saw, so the network can
    learn this room's shadows and reflections, which carry over to objects it has never seen."""
    def rs(a, interp=cv2.INTER_NEAREST):
        return cv2.resize(np.asarray(a, np.float32), (size, size), interpolation=interp)
    rgb = cv2.resize(crop[..., ::-1], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    expect = np.choose(np.clip(pat, 0, 2).astype(int), [0.5, 1.0, 0.0])  # 0 none, 1 white, 2 black
    return np.concatenate([rgb.transpose(2, 0, 1), rs(flat, cv2.INTER_AREA)[None] / 255, rs(expect)[None],
                           rs(border)[None], rs(occ)[None], rs(cur)[None]]).astype(np.float32)


def make_outline_net():
    """A small U-Net (about 100k parameters): 8 input channels -> object logit per pixel."""
    torch = _torch()
    nn = torch.nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3, self.mid = block(8, 16), block(16, 32), block(32, 48), block(48, 64)
            self.u3, self.d3 = nn.ConvTranspose2d(64, 48, 2, 2), block(96, 48)
            self.u2, self.d2 = nn.ConvTranspose2d(48, 32, 2, 2), block(64, 32)
            self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), block(32, 16)
            self.out = nn.Conv2d(16, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            a = self.e1(x)
            b = self.e2(self.pool(a))
            c = self.e3(self.pool(b))
            m = self.mid(self.pool(c))
            c = self.d3(torch.cat([self.u3(m), c], 1))
            b = self.d2(torch.cat([self.u2(c), b], 1))
            a = self.d1(torch.cat([self.u1(b), a], 1))
            return self.out(a)
    return UNet()


def fit_pose_fixed(hull, pose, K, dist, up, shape, dims, p0):
    """Place a shape of the TRUE size where it best matches the outline (position and yaw only).
    Its projection is the outline the object really has: the label for the outline network."""
    from scipy.optimize import minimize
    hull = np.asarray(hull, np.float32).reshape(-1, 2)
    A = cv2.contourArea(hull)
    base = np.array(p0[:6], float)
    L, W, H = dims
    base[3:6] = np.log([L, W if shape not in ROUND_TIES else L, H if shape != 'sphere' else L])

    def cost(q):
        p = base.copy()
        p[:3] = q
        uv = cv2.projectPoints(shape_points(p, shape, up), pose['rvec'], pose['tvec'], K, dist)[0].reshape(-1, 2)
        c = cv2.convexHull(uv.astype(np.float32))
        inter, _ = cv2.intersectConvexConvex(c, hull)
        return 1 - inter / (cv2.contourArea(c) + A - inter)

    best = None
    for dyaw in (0.0, math.pi / 2):
        q0 = np.r_[p0[0], p0[1], p0[2] + dyaw]
        r = minimize(cost, q0, method='Nelder-Mead',
                     options=dict(initial_simplex=np.vstack([q0, q0 + np.diag([0.01, 0.01, 0.3])]), maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    p = base.copy()
    p[:3] = best.x
    return p, 1 - best.fun


def cmd_room_labels(args):
    """Training data for the outline network, one pass per photo (resumable): every distinct named
    log is redone from its frame with its run's calibration, the shape of the TRUE size is placed
    where it best matches the detected outline, and its projection becomes the label. The inputs and
    label are cached in data/outline_cache. Frames where the true-size shape and the detected
    outline disagree badly (overlap below 0.6) or the object is cut off are skipped as untrustworthy."""
    db().execute('CREATE TABLE IF NOT EXISTS outline_labels(measurement INTEGER PRIMARY KEY, ts REAL, '
                 'source TEXT, data TEXT)')
    os.makedirs(OUTLINE_CACHE, exist_ok=True)
    rows, seen = [], set()
    for mid, est, truth, calib_id in db().execute(
            "SELECT m.id, m.est, m.truth, m.calib_id FROM measurements m LEFT JOIN outline_labels o "
            "ON o.measurement = m.id WHERE m.kind='room' AND m.truth IS NOT NULL AND o.measurement IS NULL "
            "ORDER BY m.id DESC"):
        t, e = json.loads(truth), json.loads(est)
        if not t.get('object') or not e['objects'] or log_key(t, e) in seen:
            continue
        seen.add(log_key(t, e))
        rows.append((mid, e, t, calib_id))
    if args.limit:
        rows = rows[:args.limit]
    print(f'{len(rows)} distinct logs to label (newest first)')
    t0 = time.time()
    for n, (mid, e, t, calib_id) in enumerate(rows, 1):
        try:
            source, data = outline_label_for_log(mid, e, t, calib_id)
        except Exception as ex:  # one bad log must not stop a long run
            source, data = f'failed: {type(ex).__name__}', None
        db().execute('INSERT OR REPLACE INTO outline_labels(measurement, ts, source, data) VALUES(?,?,?,?)',
                     (mid, time.time(), source, None if data is None else jdump(data)))
        db().commit()
        if n % 10 == 0 or n == len(rows):
            el = time.time() - t0
            ok = db().execute("SELECT count(*) FROM outline_labels WHERE data IS NOT NULL").fetchone()[0]
            print(f'  {n}/{len(rows)}  {el / n:.1f} s per log, about {el / n * (len(rows) - n) / 60:.0f} min left, '
                  f'{ok} usable labels so far', flush=True)


def outline_label_for_log(mid, e, t, calib_id):
    T = np.array([t['L'], t['W'], t['H']])
    o_log = min(e['objects'], key=lambda o: np.abs(np.array([o['L'], o['W'], o['H']]) - T).sum())
    shape = o_log['shape'] if o_log['shape'] in ROOM_SHAPES else 'box'
    if shape == 'revolve':  # the truth is a bounding size: a cylinder of it
        shape = 'cylinder'
    row = db().execute("SELECT blob FROM results WHERE kind='room_raw' AND (ref=? OR ref=?)", (mid, str(mid))).fetchone()
    if not row:
        return 'no frame', None
    img = decode_img(row[0])
    cal = load_calib(calib_id, required=False) if calib_id else None
    if cal is not None and (cal.width, cal.height) != (img.shape[1], img.shape[0]):
        cal = None
    room = Room(argparse.Namespace(shape=shape, object_class=None, truth=None, seg='room'),
                (img.shape[1], img.shape[0]), cal, store=False)
    room.corrector, room.want_inputs = False, True
    with contextlib.redirect_stdout(io.StringIO()):
        res = room.process(img, None)
    objs = [o for o in res['objects'] if 'inputs' in o]
    if not objs:
        return 'not found', None
    o = min(objs, key=lambda q: np.abs(np.array([q['L'], q['W'], q['H']]) - T).sum())
    if o['cut']:
        return 'cut off', None
    pose = o['views'][0][1]
    p, iou = fit_pose_fixed(o['hull'], pose, room.K, room.dist, o['up'], shape, T, o['p'])
    if iou < 0.6:
        return f'disagrees (overlap {iou:.2f})', None
    uv = cv2.projectPoints(shape_points(p, shape, o['up']), pose['rvec'], pose['tvec'], room.K, room.dist)[0].reshape(-1, 2)
    x, y, w, h = o['rect']
    lab = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lab, [np.round(cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2) - [x, y]).astype(np.int32)], 1)
    Y = cv2.resize(lab, (OUTLINE_SIZE, OUTLINE_SIZE), interpolation=cv2.INTER_NEAREST)
    np.savez_compressed(os.path.join(OUTLINE_CACHE, f'{mid}.npz'), X=o['inputs'].astype(np.float16), Y=Y)
    return 'ok', dict(object=t['object'], shape=shape, iou=float(iou), rect=list(map(int, o['rect'])))


def outline_iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def cmd_train_outline(args):
    """The outline network: folds hold out whole objects; each held-out photo's predicted outline is
    compared with its label (overlap), next to the built-in outline's overlap. Accepted only if it
    is better on objects it never saw. Checkpoints every epoch (data/checkpoints/train_outline.pt):
    after a power cut the same command resumes; --fresh starts over."""
    torch = _torch()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    db().execute('CREATE TABLE IF NOT EXISTS outline_labels(measurement INTEGER PRIMARY KEY, ts REAL, '
                 'source TEXT, data TEXT)')
    rows = [(mid, json.loads(d)) for mid, d in db().execute(
        "SELECT measurement, data FROM outline_labels WHERE data IS NOT NULL ORDER BY measurement")]
    rows = [(mid, d) for mid, d in rows if os.path.exists(os.path.join(OUTLINE_CACHE, f'{mid}.npz'))]
    objects = sorted({d['object'] for _, d in rows})
    print(f'outline data: {len(rows)} labelled photos of {len(objects)} objects, training on {dev}')
    if len(objects) < 3:
        raise SystemExit('need labels of at least 3 objects: run `room-labels` first')
    X = np.stack([np.load(os.path.join(OUTLINE_CACHE, f'{mid}.npz'))['X'] for mid, _ in rows])  # float16
    Y = np.stack([np.load(os.path.join(OUTLINE_CACHE, f'{mid}.npz'))['Y'] for mid, _ in rows]).astype(np.float32)
    obj = np.array([d['object'] for _, d in rows])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(objects)
    k = min(args.folds, len(objects))
    ck_path = os.path.join(DATA, 'checkpoints', 'train_outline.pt')
    os.makedirs(os.path.dirname(ck_path), exist_ok=True)
    fp = hashlib.sha256(jdump(dict(ids=[int(m) for m, _ in rows], epochs=args.epochs, lr=args.lr, folds=args.folds,
                                   seed=args.seed, batch=args.batch)).encode()).hexdigest()
    state = None
    if os.path.exists(ck_path) and not args.fresh:
        try:
            state = torch.load(ck_path, weights_only=False)
        except Exception:
            state = None
        if state is not None and state.get('fp') != fp:
            print('checkpoint is for other data or settings: starting fresh')
            state = None
        elif state is not None:
            print(f"resuming: {state['fold']} fold(s) done, run {state['fold'] + 1} at epoch {state['epoch']}")
    if state is None:
        state = dict(fp=fp, fold=0, epoch=0, model=None, opt=None, iou_net=np.full(len(rows), np.nan))

    def save():
        tmp = ck_path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, ck_path)

    def augment(xb, gen):
        # flips (all channels together) and fake shadows / brightness on the colour channels only
        if gen.random() < 0.5:
            xb = xb.flip(3)
        if gen.random() < 0.5:
            xb = xb.flip(2)
        n = xb.shape[0]
        gain = torch.tensor(gen.uniform(0.6, 1.3, (n, 1, 1, 1)), dtype=xb.dtype, device=xb.device)
        xb[:, :3] = (xb[:, :3] * gain).clamp(0, 1)
        for i in range(n):  # one random shadow band per crop
            if gen.random() < 0.5:
                a, b = sorted(gen.integers(0, OUTLINE_SIZE, 2))
                xb[i, :3, :, a:b + 1] *= float(gen.uniform(0.4, 0.8))
        return xb

    def run(train_idx, tag, epochs):
        torch.manual_seed(args.seed + tag)
        net = make_outline_net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        start = 0
        if state['model'] is not None and state['fold'] == tag:
            net.load_state_dict(state['model'])
            opt.load_state_dict(state['opt'])
            start = state['epoch']
        gen = np.random.default_rng(args.seed * 1000 + tag * 100 + start)
        bce = torch.nn.BCEWithLogitsLoss()
        for ep in range(start, epochs):
            net.train()
            order = gen.permutation(train_idx)
            tot = 0.0
            for s in range(0, len(order), args.batch):
                idx = order[s:s + args.batch]
                xb = augment(torch.tensor(X[idx], dtype=torch.float32, device=dev), gen)
                yb = torch.tensor(Y[idx], device=dev)[:, None]
                logit = net(xb)
                pr = torch.sigmoid(logit)
                dice = 1 - (2 * (pr * yb).sum((1, 2, 3)) + 1) / (pr.sum((1, 2, 3)) + yb.sum((1, 2, 3)) + 1)
                loss = bce(logit, yb) + dice.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            state.update(epoch=ep + 1, model=net.state_dict(), opt=opt.state_dict())
            save()
            print(f'    epoch {ep + 1}/{epochs}  loss {tot / len(order):.3f}', flush=True)
        net.eval()
        return net

    def predict(net, idx):
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                xb = torch.tensor(X[idx[s:s + 64]], dtype=torch.float32, device=dev)
                out.append(torch.sigmoid(net(xb))[:, 0].cpu().numpy())
        return np.concatenate(out)

    for fi in range(state['fold'], k):
        te = np.flatnonzero(np.isin(obj, perm[fi::k]))
        print(f'  fold {fi + 1}/{k}: held out {[str(o) for o in perm[fi::k]]}')
        net = run(np.flatnonzero(~np.isin(obj, perm[fi::k])), fi, args.epochs)
        P = predict(net, te)
        for j, i in enumerate(te):
            state['iou_net'][i] = outline_iou(P[j], Y[i])
        state.update(fold=fi + 1, epoch=0, model=None, opt=None)
        save()
    iou_net = state['iou_net']
    iou_base = np.array([outline_iou(X[i, 7], Y[i]) for i in range(len(rows))])
    print(f"{'object':<18}{'photos':>7}{'built-in outline':>18}{'learned outline':>17}   (overlap with the true outline)")
    for o in objects:
        m = obj == o
        print(f'{o:<18}{m.sum():>7}{iou_base[m].mean():>18.3f}{np.nanmean(iou_net[m]):>17.3f}')
    accepted = bool(np.nanmean(iou_net) >= iou_base.mean() + 0.02)
    print(f"{'all':<18}{len(rows):>7}{iou_base.mean():>18.3f}{np.nanmean(iou_net):>17.3f}   "
          f"{'ACCEPTED' if accepted else 'rejected (needs +0.02 on objects it never saw)'}")
    print('  final network on all photos')
    net = run(np.arange(len(rows)), k, args.epochs)
    rid = put_result('outline_net', dict(n=len(rows), objects=objects, epochs=args.epochs, accepted=accepted,
                                         iou_base=float(iou_base.mean()), iou_net=float(np.nanmean(iou_net)),
                                         channels=OUTLINE_CHANNELS, size=OUTLINE_SIZE),
                     blob=state_bytes(net.state_dict()))
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print(f'stored outline network as results #{rid}' + ('; use it with `room --learned-outline`' if accepted else ''))


def load_outline_net():
    """Latest outline network as (inputs -> probability map, accepted) or None."""
    row = db().execute("SELECT data, blob FROM results WHERE kind='outline_net' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    torch = _torch()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = json.loads(row[0])
    net = make_outline_net()
    net.load_state_dict(torch.load(io.BytesIO(row[1])))
    net.to(dev).eval()

    def apply(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(X[None], device=dev)))[0, 0].cpu().numpy()
    return apply, d['accepted']


def cmd_train_corrector(args):
    if getattr(args, 'room', False):
        return cmd_train_corrector_room(args)
    torch = _torch()
    S = [s for s in load_samples() if np.all(np.isfinite(s['x']))]
    objects = sorted({s['object'] for s in S})
    print(f'corrector data: {len(S)} labelled frames, {len(objects)} objects '
          f'(handoff target: 500 frames, >= 10 objects)')
    if len(objects) < 3:
        raise SystemExit('need labelled frames (D, W known) from at least 3 objects; run `collect`')
    X = np.array([s['x'] for s in S], np.float32)
    geoD = np.array([s['geo']['D'] for s in S])
    geoW = np.array([s['geo']['W'] for s in S])
    Y = np.stack([np.array([s['d'] for s in S]) - geoD, np.array([s['w'] for s in S]) - geoW], 1).astype(np.float32)
    obj = np.array([s['object'] for s in S])
    light = np.array([s['lighting'] for s in S])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(objects)
    k = min(args.folds, len(objects))
    folds = [perm[i::k] for i in range(k)]
    oof_mu, oof_sd, best_eps = np.zeros_like(Y), np.zeros_like(Y), []

    def norm_stats(idx):
        xm, xs = X[idx].mean(0), X[idx].std(0) + 1e-6
        ys = Y[idx].std(0) + 1e-6
        return xm, xs, ys

    for fi, test_objs in enumerate(folds):
        te = np.isin(obj, test_objs)
        tr_objs = [o for o in objects if o not in test_objs]
        va_objs = [tr_objs[fi % len(tr_objs)]] if len(tr_objs) >= 3 else []
        va = np.isin(obj, va_objs)
        tr = ~te & ~va
        if not va.any():  # too few objects: random 15 % of training frames for early stopping
            idx = np.flatnonzero(~te)
            va = np.zeros(len(S), bool)
            va[rng.choice(idx, max(1, len(idx) // 7), replace=False)] = True
            tr = ~te & ~va
        xm, xs, ys = norm_stats(tr)
        m, ep = train_corrector_once((X[tr] - xm) / xs, Y[tr] / ys, (X[va] - xm) / xs, Y[va] / ys,
                                     args.epochs, args.lr, args.seed + fi)
        best_eps.append(ep)
        with torch.no_grad():
            out = m(torch.tensor((X[te] - xm) / xs)).numpy()
        oof_mu[te] = out[:, :2] * ys
        oof_sd[te] = np.exp(0.5 * np.clip(out[:, 2:], -8, 6)) * ys
        print(f'  fold {fi + 1}/{k}: held-out objects {[str(o) for o in test_objs]}, best epoch {ep}')
    err_geo = np.abs(Y)  # geometry error = |truth - geo|
    err_cor = np.abs(Y - oof_mu)
    metrics = {}
    for lt in ['all'] + sorted(set(light)):
        m_ = np.ones(len(S), bool) if lt == 'all' else light == lt
        metrics[lt] = dict(n=int(m_.sum()), mae_geo_D=err_geo[m_, 0].mean(), mae_cor_D=err_cor[m_, 0].mean(),
                           mae_geo_W=err_geo[m_, 1].mean(), mae_cor_W=err_cor[m_, 1].mean())
    z = (Y - oof_mu) / oof_sd
    calib_cov = (np.abs(z) < 1).mean(0)  # fraction inside +-1 sigma (0.68 if variances are honest)
    acc = dict(D=bool(metrics['all']['mae_cor_D'] < metrics['all']['mae_geo_D']),
               W=bool(metrics['all']['mae_cor_W'] < metrics['all']['mae_geo_W']))
    print(f"{'lighting':<12}{'n':>6}{'D geo [mm]':>12}{'D corr [mm]':>13}{'W geo [mm]':>12}{'W corr [mm]':>13}")
    for lt, mt in metrics.items():
        print(f"{lt:<12}{mt['n']:>6}{mt['mae_geo_D'] * 1e3:>12.1f}{mt['mae_cor_D'] * 1e3:>13.1f}"
              f"{mt['mae_geo_W'] * 1e3:>12.1f}{mt['mae_cor_W'] * 1e3:>13.1f}")
    print(f'held-out frames within +-1 predicted sigma: D {calib_cov[0]:.2f}, W {calib_cov[1]:.2f} (0.68 ideal)')
    print('acceptance (corrected beats geometry on held-out objects): '
          f"D {'ACCEPTED' if acc['D'] else 'rejected'}, W {'ACCEPTED' if acc['W'] else 'rejected'}")
    # final model on all objects for deployment, epochs from the folds
    xm, xs, ys = norm_stats(np.ones(len(S), bool))
    ep = int(np.median(best_eps))
    final, _ = train_corrector_once((X - xm) / xs, Y / ys, None, None, ep, args.lr, args.seed, fixed_epochs=ep)
    data = dict(features=FEATURES, n=len(S), objects=objects, folds=[list(f) for f in folds], epochs=ep,
                metrics=metrics, accepted=acc, coverage_1sigma=calib_cov,
                norm=dict(x_mean=xm, x_std=xs, y_std=ys),
                oof=dict(ids=[s['id'] for s in S], lighting=light, y=Y, mu=oof_mu, sd=oof_sd))
    rid = put_result('corrector', data, blob=state_bytes(final.state_dict()))
    print(f'stored corrector as results #{rid}')
    print(fig_f8())


def crops_array(blobs, rgb, size=32):
    out = []
    for b in blobs:
        im = decode_img(b)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        if rgb:
            out.append(im[..., ::-1].transpose(2, 0, 1))
        else:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[None])
    return np.array(out, np.float32) / 255.0


def augment(x, rng):
    """Brightness, contrast, small rotation, simulated shadow polygon (handoff 5)."""
    x = x.copy()
    n, c, h, w = x.shape
    for i in range(n):
        im = x[i]
        mean = im.mean()
        im = (im - mean) * rng.uniform(0.7, 1.3) + mean
        im = im * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12), rng.uniform(0.92, 1.08))
        im = np.stack([cv2.warpAffine(ch, M, (w, h), borderMode=cv2.BORDER_REFLECT) for ch in im])
        if rng.random() < 0.5:
            poly = rng.uniform(-0.2, 1.2, (rng.integers(3, 6), 2)) * [w, h]
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            im = np.where(m[None] > 0, im * rng.uniform(0.4, 0.8), im)
        x[i] = np.clip(im, 0, 1)
    return x


def train_classifier_model(make, X, y, Xv, yv, n_cls, args, seed):
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = make(X.shape[1:], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        m.train()
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // 64)):
            xb = torch.tensor(augment(X[idx], rng) if args.augment else X[idx])
            opt.zero_grad()
            lossf(m(xb), torch.tensor(y[idx])).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(torch.tensor(Xv)).argmax(1).numpy() == yv).mean())
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    return m, best


def choose_test_sessions(sessions, explicit):
    """Held-out sessions: a different day if there is one, else a different lighting."""
    if explicit:
        return set(explicit.split(',')), 'given'
    days = sorted({s.split('_')[0] for s in sessions})
    if len(days) > 1:
        return {s for s in sessions if s.startswith(days[-1])}, f'held-out day {days[-1]}'
    if len(sessions) > 1:
        last = sorted(sessions)[-1]
        return {last}, f'held-out session {last} (different lighting)'
    return set(), 'single session: random 25 % of frames (optimistic, same session)'


def cmd_train_classifier(args):
    torch = _torch()
    rows = db().execute('SELECT id,object,session,crop FROM samples WHERE crop IS NOT NULL').fetchall()
    if not rows:
        raise SystemExit('no object crops stored: run `collect`')
    sessions = sorted({r[2] for r in rows})
    test_s, how = choose_test_sessions(sessions, args.test_sessions)
    ids = np.array([r[0] for r in rows])
    labels_all = np.array([r[1] for r in rows])
    sess = np.array([r[2] for r in rows])
    rng = np.random.default_rng(args.seed)
    te = np.isin(sess, list(test_s)) if test_s else rng.random(len(rows)) < 0.25
    classes = sorted(set(labels_all[~te]))
    if len(classes) < 2:
        raise SystemExit(f'need at least 2 object classes in the training split (have {classes})')
    te &= np.isin(labels_all, classes)
    tr = ~te
    X = crops_array([r[3] for r in rows], args.rgb)
    y = np.array([classes.index(l) if l in classes else -1 for l in labels_all])
    tr_idx = np.flatnonzero(tr)
    va_idx = rng.choice(tr_idx, max(1, len(tr_idx) // 10), replace=False)
    fit_idx = np.setdiff1d(tr_idx, va_idx)
    classes = [str(c) for c in classes]
    print(f'classifier data: {len(rows)} crops, {len(classes)} classes {classes}; split: {how}; '
          f'train {len(fit_idx)}, val {len(va_idx)}, test {te.sum()}')
    if te.sum() == 0:
        raise SystemExit('empty test split; pass --test-sessions')
    res, states = dict(classes=classes, split=how, test_sessions=sorted(test_s), rgb=args.rgb,
                       in_shape=list(X.shape[1:]), test_ids=ids[te], models={}), {}
    for name, make in (('mlp', make_cls_mlp), ('cnn', make_cls_cnn)):
        m, vacc = train_classifier_model(make, X[fit_idx], y[fit_idx], X[va_idx], y[va_idx], len(classes),
                                         args, args.seed)
        with torch.no_grad():
            pred = m(torch.tensor(X[te])).argmax(1).numpy()
        conf = np.zeros((len(classes), len(classes)), int)
        for t, p in zip(y[te], pred):
            conf[t, p] += 1
        acc = float((pred == y[te]).mean())
        n_par = sum(p.numel() for p in m.parameters())
        res['models'][name] = dict(test_acc=acc, val_acc=vacc, confusion=conf, params=n_par)
        states[name] = m.state_dict()
        print(f'  {name}: {n_par} parameters, val acc {vacc:.3f}, held-out test acc {acc:.3f}')
    rid = put_result('classifier', res, blob=state_bytes(states))
    print(f'stored classifiers as results #{rid}')
    print(fig_f9())


# ----------------------------------------------------------------------------
# export: post-training int8 quantization (handoff 6)
# ----------------------------------------------------------------------------
def int8_quantize(model, calib_x):
    """Post-training int8 quantization of a Sequential(Linear/Conv, ReLU, MaxPool, Flatten).

    Weights: symmetric per-output-channel int8. Activations: asymmetric per-tensor int8 at the
    model input and after every Linear/Conv(+ReLU) block, ranges from calibration data.
    Biases: int32 at scale s_in * s_w. Returns (fake-quant model that reproduces integer
    inference, dict of int8 tensors and scales for a C port).
    """
    torch = _torch()
    nn = torch.nn

    class FQ(nn.Module):
        def __init__(self, lo, hi):
            super().__init__()
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            self.s = max(hi - lo, 1e-8) / 255.0
            self.zp = int(round(-128 - lo / self.s))

        def forward(self, x):
            return (torch.clamp(torch.round(x / self.s) + self.zp, -128, 127) - self.zp) * self.s

    layers = list(model)
    x = torch.tensor(calib_x)
    q_layers = [FQ(float(x.min()), float(x.max()))]
    s_in = q_layers[0].s
    packed = dict(input=dict(scale=s_in, zero_point=q_layers[0].zp), layers=[])
    with torch.no_grad():
        for i, L in enumerate(layers):
            y = L(x)
            if isinstance(L, (nn.Linear, nn.Conv2d)):
                W = L.weight.detach().clone()
                s_w = W.reshape(W.shape[0], -1).abs().max(1).values.clamp_min(1e-12) / 127.0
                shape = (-1,) + (1,) * (W.dim() - 1)
                Wq = torch.clamp(torch.round(W / s_w.reshape(shape)), -127, 127)
                bq = torch.round(L.bias.detach() / (s_in * s_w))
                L2 = type(L)(*([L.in_features, L.out_features] if isinstance(L, nn.Linear) else
                               [L.in_channels, L.out_channels, L.kernel_size, L.stride, L.padding]))
                L2.weight.copy_(Wq * s_w.reshape(shape))
                L2.bias.copy_(bq * s_in * s_w)
                q_layers.append(L2)
                packed['layers'].append(dict(index=i, type=type(L).__name__, w_int8=Wq.to(torch.int8).numpy(),
                                             w_scale=s_w.numpy(), b_int32=bq.to(torch.int32).numpy()))
                nxt = layers[i + 1] if i + 1 < len(layers) else None
                if not isinstance(nxt, nn.ReLU):  # requantize the linear output here
                    fq = FQ(float(y.min()), float(y.max()))
                    q_layers.append(fq)
                    s_in = fq.s
                    packed['layers'][-1].update(out_scale=fq.s, out_zero_point=fq.zp)
            elif isinstance(L, nn.ReLU):
                q_layers.append(nn.ReLU())
                fq = FQ(0.0, float(y.max()))
                q_layers.append(fq)
                s_in = fq.s
                packed['layers'][-1].update(out_scale=fq.s, out_zero_point=fq.zp)
            else:  # MaxPool / Flatten keep the scale
                q_layers.append(L)
            x = y
    return nn.Sequential(*q_layers).eval(), packed


def model_footprint(model, in_shape):
    """Parameter count, weight size (float32 / int8), and peak activation memory at batch 1."""
    torch = _torch()
    nn = torch.nn
    n_par = sum(p.numel() for p in model.parameters())
    n_w = sum(m.weight.numel() for m in model if isinstance(m, (nn.Linear, nn.Conv2d)))
    n_b = n_par - n_w
    n_ch = sum(m.weight.shape[0] for m in model if isinstance(m, (nn.Linear, nn.Conv2d)))
    x = torch.zeros((1,) + tuple(in_shape))
    peak = 0
    with torch.no_grad():
        for L in model:
            y = L(x)
            if not isinstance(L, (nn.ReLU, nn.Flatten)):  # ReLU runs in place, Flatten is a view
                peak = max(peak, x.numel() + y.numel())
            x = y
    return dict(params=n_par, kb_float=4 * n_par / 1024, kb_int8=(n_w + 4 * n_b + 4 * n_ch) / 1024,
                act_peak_kb_float=4 * peak / 1024, act_peak_kb_int8=peak / 1024, act_peak_elems=peak)


def onnx_export(model, in_shape, path, calib_x):
    torch = _torch()
    try:
        import onnx  # noqa: F401
    except ImportError:
        return 'onnx not installed (pip install onnx onnxruntime): ONNX files skipped'
    torch.onnx.export(model, (torch.zeros((1,) + tuple(in_shape)),), path, dynamo=False,
                      input_names=['x'], output_names=['y'], dynamic_axes={'x': {0: 'n'}, 'y': {0: 'n'}},
                      opset_version=13)
    msg = f'wrote {os.path.relpath(path, ROOT)}'
    try:
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                              quantize_static)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{'x': calib_x[i:i + 1].astype(np.float32)} for i in range(len(calib_x))])

            def get_next(self):
                return next(self.it, None)

        import logging
        logging.getLogger().setLevel(logging.ERROR)  # ORT's "consider pre-processing" advice
        qpath = path.replace('.onnx', '_int8.onnx')
        quantize_static(path, qpath, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        msg += f', {os.path.relpath(qpath, ROOT)} (QDQ int8, STM32Cube.AI imports it)'
    except ImportError:
        msg += ' (onnxruntime not installed: no QDQ int8 file)'
    return msg


def ort_run(path, x):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return s.run(None, {'x': x.astype(np.float32)})[0]


def cmd_export(args):
    torch = _torch()
    os.makedirs(EXPORT, exist_ok=True)
    report = {}
    rc = latest_result('corrector')
    if rc:
        d = rc['data']
        m = make_corrector()
        m.load_state_dict(torch.load(io.BytesIO(rc['blob']), weights_only=True))
        m.eval()
        ids = d['oof']['ids']
        S = {s['id']: s for s in load_samples()}
        S = [S[i] for i in ids if i in S]
        X = (np.array([s['x'] for s in S], np.float32) - d['norm']['x_mean']) / d['norm']['x_std']
        X = X.astype(np.float32)
        Y = np.array(d['oof']['y'], np.float32)[:len(S)]
        ys = np.array(d['norm']['y_std'])
        mq, packed = int8_quantize(m, X)
        with torch.no_grad():
            pf = m(torch.tensor(X)).numpy()[:, :2] * ys
            pq = mq(torch.tensor(X)).numpy()[:, :2] * ys
        fp = model_footprint(m, (len(FEATURES),))
        rep = dict(fp, metric='MAE of residual D,W on all collected frames [m] (in-sample)',
                   float=np.abs(Y - pf).mean(0), int8=np.abs(Y - pq).mean(0),
                   max_abs_diff=np.abs(pf - pq).max(0))
        path = os.path.join(EXPORT, 'corrector.onnx')
        rep['onnx'] = onnx_export(m, (len(FEATURES),), path, X[:200])
        if os.path.exists(path.replace('.onnx', '_int8.onnx')):
            try:
                po = ort_run(path.replace('.onnx', '_int8.onnx'), X)[:, :2] * ys
                rep['ort_int8'] = np.abs(Y - po).mean(0)
            except Exception as e:
                rep['ort_int8_error'] = str(e)
        np.savez(os.path.join(EXPORT, 'corrector_int8.npz'), x_mean=d['norm']['x_mean'], x_std=d['norm']['x_std'],
                 y_std=ys, **{f"l{L['index']}_{k}": v for L in packed['layers'] for k, v in L.items()
                               if k not in ('index', 'type')})
        report['corrector'] = rep
        print(f"corrector: {fp['params']} params, {fp['kb_float']:.1f} KB float / {fp['kb_int8']:.1f} KB int8, "
              f"peak activations {fp['act_peak_kb_int8'] * 1024:.0f} B int8; residual MAE D,W float "
              f"{rep['float'][0] * 1e3:.1f},{rep['float'][1] * 1e3:.1f} mm, int8 "
              f"{rep['int8'][0] * 1e3:.1f},{rep['int8'][1] * 1e3:.1f} mm")
        print('  ' + rep['onnx'])
    else:
        print('no corrector trained: skipped')
    rk = latest_result('classifier')
    if rk:
        d = rk['data']
        states = torch.load(io.BytesIO(rk['blob']), weights_only=True)
        rows = db().execute('SELECT id,object,crop FROM samples WHERE crop IS NOT NULL').fetchall()
        test_ids = set(d['test_ids'])
        tr_rows = [r for r in rows if r[0] not in test_ids and r[1] in d['classes']]
        te_rows = [r for r in rows if r[0] in test_ids and r[1] in d['classes']]
        Xte = crops_array([r[2] for r in te_rows], d['rgb'])
        yte = np.array([d['classes'].index(r[1]) for r in te_rows])
        rng = np.random.default_rng(0)
        cal = [tr_rows[i] for i in rng.permutation(len(tr_rows))[:300]]
        Xcal = crops_array([r[2] for r in cal], d['rgb'])
        shape = tuple(d['in_shape'])
        for name, make in (('mlp', make_cls_mlp), ('cnn', make_cls_cnn)):
            m = make(shape, len(d['classes']))
            m.load_state_dict(states[name])
            m.eval()
            mq, packed = int8_quantize(m, Xcal)
            with torch.no_grad():
                af = float((m(torch.tensor(Xte)).argmax(1).numpy() == yte).mean())
                aq = float((mq(torch.tensor(Xte)).argmax(1).numpy() == yte).mean())
            fp = model_footprint(m, shape)
            rep = dict(fp, metric='held-out accuracy', float=af, int8=aq)
            path = os.path.join(EXPORT, f'classifier_{name}.onnx')
            rep['onnx'] = onnx_export(m, shape, path, Xcal)
            if os.path.exists(path.replace('.onnx', '_int8.onnx')):
                try:
                    rep['ort_int8'] = float((ort_run(path.replace('.onnx', '_int8.onnx'), Xte).argmax(1) == yte).mean())
                except Exception as e:
                    rep['ort_int8_error'] = str(e)
            np.savez(os.path.join(EXPORT, f'classifier_{name}_int8.npz'), classes=np.array(d['classes']),
                     **{f"l{L['index']}_{k}": v for L in packed['layers'] for k, v in L.items()
                        if k not in ('index', 'type')})
            report[f'classifier_{name}'] = rep
            print(f"classifier {name}: {fp['params']} params, {fp['kb_float']:.1f} KB float / "
                  f"{fp['kb_int8']:.1f} KB int8, peak activations {fp['act_peak_kb_int8']:.1f} KB int8; "
                  f"accuracy float {af:.3f}, int8 {aq:.3f}" +
                  (f", onnxruntime int8 {rep['ort_int8']:.3f}" if 'ort_int8' in rep else ''))
            print('  ' + rep['onnx'])
    else:
        print('no classifier trained: skipped')
    if report:
        put_result('export', report)
        print(fig_f10())


# ----------------------------------------------------------------------------
# report: figures F1-F10 and tables, from the database only (handoff 8)
# ----------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.bbox': 'tight'})
    return plt


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    _plt().close(fig)
    return f'{name}: written'


def fig_f1():
    plt = _plt()
    g = latest_result('ground_cal')
    h, th = (g['data']['h'], g['data']['theta']) if g else (0.40, math.radians(15))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ph = latest_result('setup_photo')
    if ph:
        a0.imshow(decode_img(ph['blob'])[..., ::-1])
    else:
        a0.text(0.5, 0.5, 'setup photo slot\n\npython camdist.py report --setup-photo PATH',
                ha='center', va='center', color='0.4')
        a0.set_facecolor('0.95')
    a0.set_xticks([]), a0.set_yticks([])
    a0.set_title('setup')
    D, Hobj = float(np.clip(1.3 * h / math.tan(max(th, 0.05)), 0.6, 2.1)), 0.35  # below the optical axis
    a1.plot([-0.2, 2.6], [0, 0], color='0.3', lw=2)
    a1.plot([0, 0], [0, h], color='0.6', ls=':')
    a1.add_patch(plt.Rectangle((D, 0), 0.25, Hobj, fc='#c9d6e3', ec='#34495e'))
    ax_len = 2.4
    a1.annotate('', xy=(ax_len * math.cos(th), h - ax_len * math.sin(th)), xytext=(0, h),
                arrowprops=dict(arrowstyle='->', color='#1f6f8b', lw=1.2))
    a1.plot([0, D], [h, 0], color='#c0392b', lw=1.2)
    a1.plot([0, D], [h, Hobj], color='#c0392b', lw=1, ls='--')
    a1.plot([0, 2.5], [h, h], color='0.7', lw=0.8, ls='--')
    a1.add_patch(plt.Polygon([[0, h - 0.05], [-0.1, h - 0.1], [-0.1, h + 0.1], [0, h + 0.05]], fc='0.2'))
    alpha = math.atan2(h, D) - th
    a1.text(0.55, h - 0.03, r'$\theta$', color='#1f6f8b')
    a1.text(0.9, h - 0.2, r'$\alpha$', color='#c0392b')
    a1.text(-0.12, h / 2, '$h$', ha='right')
    a1.annotate('', xy=(D, -0.06), xytext=(0, -0.06), arrowprops=dict(arrowstyle='<->'))
    a1.text(D / 2, -0.12, r'$D = h / \tan(\theta+\alpha)$', ha='center')
    a1.text(D + 0.3, Hobj / 2, 'object\n$W$, $H$', va='center')
    a1.text(D, 0.03, '$v_b$', ha='right', color='#c0392b')
    a1.text(0.1, h + 0.07, 'pinhole camera ($f_x, f_y, c_x, c_y, k_1, k_2$)', fontsize=8)
    a1.set_aspect('equal')
    a1.set_xlim(-0.3, 2.7)
    a1.set_ylim(-0.2, h + 0.25)
    a1.axis('off')
    a1.set_title(f'ground-plane geometry (h = {h:.2f} m, theta = {math.degrees(th):.1f} deg, '
                 f'alpha = {math.degrees(alpha):.1f} deg)')
    return _save(fig, 'F1_setup_geometry.png')


def fig_f2():
    calib = load_calib('board', required=False)
    r = latest_result('board_residuals', ref=calib.id) if calib else None
    if r is None:
        return 'F2: skipped (no board calibration)'
    plt = _plt()
    P = np.array(r['data']['points'])
    W, H = r['data']['size']
    mag = np.linalg.norm(P[:, 2:], axis=1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    sc = a0.quiver(P[:, 0], P[:, 1], P[:, 2], -P[:, 3], mag, cmap='viridis', angles='xy',
                   scale_units='xy', scale=1 / 60, width=0.002)
    fig.colorbar(sc, ax=a0, label='|residual| [px]')
    a0.set_xlim(0, W), a0.set_ylim(H, 0), a0.set_aspect('equal')
    a0.set_title(f'reprojection residuals (x60), rms {r["data"]["rms"]:.3f} px')
    gx, gy = 8, 6
    S, N = np.zeros((gy, gx)), np.zeros((gy, gx))
    for (u, v), m in zip(P[:, :2], mag):
        j, i = min(int(v * gy / H), gy - 1), min(int(u * gx / W), gx - 1)
        S[j, i] += m ** 2
        N[j, i] += 1
    with np.errstate(invalid='ignore'):
        im = a1.imshow(np.sqrt(S / N), extent=(0, W, H, 0), cmap='magma')
    fig.colorbar(im, ax=a1, label='rms residual [px]')
    a1.set_title('rms by image region (blank = no corners: add views there)')
    return _save(fig, 'F2_board_reprojection.png')


def fig_f3():
    r = latest_result('ruler_fit')
    if r is None:
        return 'F3: skipped (no ruler calibration)'
    plt = _plt()
    d = r['data']
    z = np.array(d['z'])
    l = np.linalg.norm(np.array(d['P2']) - np.array(d['P1']), axis=1)
    L = float(np.array(d['L'])[0])
    main, f0 = d['fits'][d['main']], d['fits']['f']
    p, se = main['params'], main['se']
    zz = np.linspace(z.min() * 0.9, z.max() * 1.1, 200)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    a0.errorbar(1 / z, l, yerr=np.array(d['l_std']), fmt='o', ms=4, color='k', label='measured')
    a0.plot(1 / zz, p['f'] * L / (zz + p['delta']), color='#c0392b',
            label=f"fit $l = fL/(Z+\\delta)$: f = {p['f']:.1f} $\\pm$ {se['f']:.1f} px, "
                  f"$\\delta$ = {p['delta'] * 1e3:.1f} $\\pm$ {se['delta'] * 1e3:.1f} mm")
    a0.plot(1 / zz, f0['params']['f'] * L / zz, color='0.5', ls='--',
            label=f"$\\delta$ = 0: f = {f0['params']['f']:.1f} px")
    a0.set_ylabel('l [px]')
    a0.legend(fontsize=8)
    a0.set_title(f'ruler calibration, L = {L * 100:.1f} cm, {len(z)} placements')
    a1.axhline(0, color='0.6', lw=0.8)
    a1.plot(1 / z, -np.array(main['resid']), 'o', ms=4, color='#c0392b')
    a1.plot(1 / z, -np.array(f0['resid']), 'x', ms=4, color='0.5')
    a1.set_xlabel('1 / Z_meas [1/m]')
    a1.set_ylabel('residual [px]')
    return _save(fig, 'F3_ruler_calibration.png')


def fig_f4():
    rows = db().execute("SELECT data FROM results WHERE kind='fisher' ORDER BY id DESC").fetchall()
    d = next((json.loads(r[0]) for r in rows if json.loads(r[0])['real']), None)
    if d is None or 'se_boot' not in d['table'][0]:
        return 'F4: skipped (no Fisher report on real ruler data with bootstrap)'
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    pred = np.array([t['se_pred'] for t in d['table']])
    obs = np.array([t['se_boot'] for t in d['table']])
    names = ['{' + ','.join(t['nuisance']) + '}' for t in d['table']]
    for x, y, n, c in zip(pred, obs, names, ['#1f6f8b', '#c0392b', '#7d6608', '#6c3483']):
        ax.plot(x, y, 'o', ms=8, color=c, label=f'{n}: rho = {d["table"][names.index(n)]["rho"]:.3f}')
    lim = [min(pred.min(), np.nanmin(obs)) / 1.5, max(pred.max(), np.nanmax(obs)) * 1.5]
    ax.plot(lim, lim, color='0.5', ls='--', lw=0.8, label='observed = predicted')
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel(r'predicted SE of f = $I_{\rm marg}^{-1/2}$ [px]')
    ax.set_ylabel('bootstrap SE of f [px]')
    ax.set_title('estimability of f: Fisher prediction vs bootstrap')
    ax.legend(fontsize=7.5)
    return _save(fig, 'F4_fisher_pred_vs_boot.png')


def fig_f5():
    plt = _plt()
    R = np.logspace(0.01, 3, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(R, [rho_uniform_w(r) for r in R], color='#1f6f8b', label='uniform in 1/Z (limit 1/16)')
    ax.plot(R, [rho_uniform_z(r) for r in R], color='#7d6608', ls='-.', label='uniform in Z (limit 1/4)')
    ax.axhline(1 / 16, color='#c0392b', ls='--', lw=0.8)
    ax.axhline(1 / 4, color='#7d6608', ls=':', lw=0.8)
    ax.plot(4, rho_uniform_w(4), 's', color='k', ms=5, label=f'Z in [0.5, 2] m: {rho_uniform_w(4):.3f}')
    d = latest_result('fisher')
    if d:
        dd = d['data']
        ax.plot(dd['spread'], dd['table'][1]['rho'], '*', ms=12, color='#c0392b',
                label=f"this design ({'real' if dd['real'] else 'plan'}): {dd['table'][1]['rho']:.3f}")
    ax.set_xscale('log')
    ax.set_xlabel('distance spread Zmax / Zmin')
    ax.set_ylabel(r'retention $\rho$ of f with $\delta$ unknown')
    ax.set_ylim(0, 0.3)
    ax.legend(fontsize=8)
    ax.set_title('information on f that survives estimating the offset')
    return _save(fig, 'F5_retention_vs_spread.png')


def fig_f6():
    rows = db().execute("SELECT est, truth FROM measurements WHERE kind='ruler' AND truth IS NOT NULL").fetchall()
    if not rows:
        return 'F6: skipped (no ruler measurements with truth; run `measure --truth`)'
    plt = _plt()
    E = [json.loads(r[0]) for r in rows]
    T = [json.loads(r[1]) for r in rows]
    z = np.array([t['Z'] for t in T])
    err = np.array([e['Z'] for e in E]) - z
    f, L = np.median([e['f'] for e in E]), np.median([e['L'] for e in E])
    s_l = float(np.median([e['l_std'] for e in E])) or 1.0
    zz = np.linspace(0, z.max() * 1.1, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, c in ((1, '#1f6f8b'), (2, '#a9cce3')):
        ax.fill_between(zz, -k * zz ** 2 / (f * L) * s_l * 1e3, k * zz ** 2 / (f * L) * s_l * 1e3, color=c,
                        alpha=0.35 if k == 1 else 0.25, lw=0,
                        label=f'{k} sigma, $\\delta Z = Z^2 \\delta l / (fL)$, $\\delta l$ = {s_l:.2f} px' if k == 1
                        else f'{k} sigma')
    ax.plot(z, err * 1e3, 'o', color='k', ms=4, label='measured')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel('true distance Z [m]')
    ax.set_ylabel('error [mm]')
    ax.set_title(f'ruler distance error, L = {L * 100:.1f} cm, f = {f:.0f} px')
    ax.legend(fontsize=8)
    return _save(fig, 'F6_distance_error_vs_Z.png')


def fig_f7():
    g = latest_result('ground_cal')
    pts = []
    for s in load_samples():
        pts.append((s['geo']['v_b'], s['geo']['D'] - s['d'], 'collect'))
    for e, t in db().execute("SELECT est, truth FROM measurements WHERE kind='ground' AND truth IS NOT NULL"):
        e, t = json.loads(e), json.loads(t)
        pts.append((e['v_b'], e['D'] - t['D'], 'ground'))
    if g is None or not pts:
        return 'F7: skipped (needs ground calibration and frames with true D)'
    plt = _plt()
    gd = g['data']
    rows = gd['rows']
    v = np.array([r['v'] for r in rows])
    sD = np.array([r['sigma_D'] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(v, -sD * 1e3, sD * 1e3, color='#a9cce3', alpha=0.6, lw=0,
                    label='predicted 1 sigma from h, theta, pixel noise')
    P = np.array([(p[0], p[1]) for p in pts])
    src = np.array([p[2] for p in pts])
    for s, c in (('collect', '#1f6f8b'), ('ground', '#c0392b')):
        m = src == s
        if m.any():
            ax.plot(P[m, 0], P[m, 1] * 1e3, 'o', ms=3, color=c, alpha=0.7, label=f'{s} frames')
    ax.axvline(gd['v_horizon'], color='0.4', ls='--', lw=0.8, label='horizon row')
    ax.axhline(0, color='0.5', lw=0.8)
    lim = np.nanpercentile(np.abs(P[:, 1]) * 1e3, 98) * 1.5 if len(P) else 100
    ax.set_ylim(-max(lim, sD.min() * 3e3), max(lim, sD.min() * 3e3))
    ax.set_xlabel('bottom row $v_b$ [px]')
    ax.set_ylabel('D error (geometry - truth) [mm]')
    ax.set_title('ground-plane distance error across image rows')
    ax.legend(fontsize=8)
    return _save(fig, 'F7_ground_error_vs_row.png')


def fig_f8():
    r = latest_result('corrector')
    if r is None:
        return 'F8: skipped (no corrector)'
    plt = _plt()
    M = r['data']['metrics']
    names = list(M)
    x = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, q in zip(axs, ('D', 'W')):
        ax.bar(x - 0.2, [M[n][f'mae_geo_{q}'] * 1e3 for n in names], 0.4, color='0.6', label='geometry')
        ax.bar(x + 0.2, [M[n][f'mae_cor_{q}'] * 1e3 for n in names], 0.4, color='#1f6f8b',
               label='geometry + MLP')
        ax.set_xticks(x, [f'{n}\n(n={M[n]["n"]})' for n in names])
        ax.set_ylabel(f'MAE of {q} on held-out objects [mm]')
        ax.set_title(f"{q}: corrector {'accepted' if r['data']['accepted'][q] else 'rejected'}")
        ax.legend(fontsize=8)
    return _save(fig, 'F8_corrector_by_lighting.png')


def fig_f9():
    r = latest_result('classifier')
    if r is None:
        return 'F9: skipped (no classifier)'
    plt = _plt()
    d = r['data']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, name in zip(axs, ('mlp', 'cnn')):
        C = np.array(d['models'][name]['confusion'], float)
        Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
        ax.imshow(Cn, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(C)):
            for j in range(len(C)):
                if C[i, j]:
                    ax.text(j, i, int(C[i, j]), ha='center', va='center', fontsize=7,
                            color='w' if Cn[i, j] > 0.5 else 'k')
        ax.set_xticks(range(len(C)), d['classes'], rotation=45, ha='right')
        ax.set_yticks(range(len(C)), d['classes'])
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        m = d['models'][name]
        ax.set_title(f"{name.upper()} ({m['params']} params): accuracy {m['test_acc']:.3f}")
    fig.suptitle(f"held-out split: {d['split']}")
    return _save(fig, 'F9_classifier_confusion.png')


def fig_f10():
    r = latest_result('export')
    if r is None:
        return 'F10: skipped (run export)'
    plt = _plt()
    d = r['data']
    names = list(d)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    cls = [n for n in names if n.startswith('classifier')]
    x = np.arange(len(cls))
    if cls:
        a0.bar(x - 0.2, [d[n]['float'] for n in cls], 0.4, color='0.6', label='float32')
        a0.bar(x + 0.2, [d[n]['int8'] for n in cls], 0.4, color='#1f6f8b', label='int8')
        a0.set_xticks(x, [n.replace('classifier_', '').upper() for n in cls])
        a0.set_ylim(0, 1.05)
        a0.set_ylabel('held-out accuracy')
    if 'corrector' in d:
        c = d['corrector']
        a0.set_title(f"classifier accuracy; corrector MAE D float {c['float'][0] * 1e3:.1f} mm "
                     f"/ int8 {c['int8'][0] * 1e3:.1f} mm", fontsize=8)
    a0.legend(fontsize=8)
    x = np.arange(len(names))
    a1.bar(x - 0.27, [d[n]['kb_float'] for n in names], 0.27, color='0.6', label='weights float32')
    a1.bar(x, [d[n]['kb_int8'] for n in names], 0.27, color='#1f6f8b', label='weights int8')
    a1.bar(x + 0.27, [d[n]['act_peak_kb_int8'] for n in names], 0.27, color='#c0392b',
           label='peak activations int8')
    a1.set_xticks(x, [f"{n.replace('classifier_', 'cls ')}\n{d[n]['params']} params" for n in names])
    a1.set_ylabel('KB')
    a1.set_yscale('log')
    a1.legend(fontsize=8)
    a1.set_title('memory footprint (flash: weights, RAM: activations)')
    return _save(fig, 'F10_quantization.png')


def write_tables():
    out = ['# camdist tables', f'generated {time.strftime("%Y-%m-%d %H:%M")} from data/camdist.sqlite', '']
    sc = db().execute('SELECT results, passed FROM selfcheck_runs ORDER BY id DESC LIMIT 1').fetchone()
    if sc:
        out += ['## Self-checks', '', '| id | result | detail |', '|---|---|---|']
        out += [f"| {r['id']} | {'pass' if r['passed'] else 'FAIL'} | {r['detail']} |" for r in json.loads(sc[0])]
        out.append('')
    for method in ('board', 'phone', 'ruler', 'nominal'):
        c = load_calib(method, required=False)
        if c:
            out += [f'## Calibration ({method}, #{c.id}, {c.width}x{c.height})', '',
                    '| fx | fy | cx | cy | k1 | k2 | p1 | p2 | rms [px] |', '|---|---|---|---|---|---|---|---|---|',
                    f'| {c.fx:.2f} | {c.fy:.2f} | {c.cx:.2f} | {c.cy:.2f} | {c.dist[0]:+.4f} | {c.dist[1]:+.4f} | '
                    f'{c.dist[2]:+.5f} | {c.dist[3]:+.5f} | {c.rms:.3f} |', '']
            if method == 'ruler':
                se = c.meta.get('se', {})
                out += [f"delta = {c.meta['delta'] * 1e3:.1f} +- {se.get('delta', float('nan')) * 1e3:.1f} mm, "
                        f"SE(f) = {se.get('f', float('nan')):.2f} px, {c.meta.get('n')} placements", '']
    fr = latest_result('fisher')
    if fr:
        d = fr['data']
        out += [f"## Fisher report ({d['desc']})", '', '| nuisance | I_raw | I_marg | rho | SE pred | SE boot |',
                '|---|---|---|---|---|---|']
        for t in d['table']:
            out.append(f"| {{{','.join(t['nuisance'])}}} | {t['I_raw']:.4g} | {t['I_marg']:.4g} | {t['rho']:.4f} | "
                       f"{t['se_pred']:.3f} | {t.get('se_boot', float('nan')):.3f} |")
        out.append('')
    g = latest_result('ground_cal')
    if g:
        d = g['data']
        out += ['## Ground plane', '', f"h = {d['h']:.3f} m, theta = {math.degrees(d['theta']):.3f} +- "
                f"{math.degrees(d['se_theta']):.3f} deg, horizon row {d['v_horizon']:.0f}, residual rms "
                f"{d['rms'] * 1e3:.1f} mm", '', '| row v | D [m] | sigma_D [mm] |', '|---|---|---|']
        out += [f"| {r['v']:.0f} | {r['D']:.2f} | {r['sigma_D'] * 1e3:.1f} |" for r in d['rows'][::8]]
        out.append('')
    c = latest_result('corrector')
    if c:
        d = c['data']
        out += [f"## Corrector ({d['n']} frames, {len(d['objects'])} objects, split by object)", '',
                '| lighting | n | D geo [mm] | D corr [mm] | W geo [mm] | W corr [mm] |', '|---|---|---|---|---|---|']
        out += [f"| {k} | {m['n']} | {m['mae_geo_D'] * 1e3:.1f} | {m['mae_cor_D'] * 1e3:.1f} | "
                f"{m['mae_geo_W'] * 1e3:.1f} | {m['mae_cor_W'] * 1e3:.1f} |" for k, m in d['metrics'].items()]
        out += ['', f"accepted: {d['accepted']}", '']
    k = latest_result('classifier')
    if k:
        d = k['data']
        out += [f"## Classifier ({len(d['classes'])} classes, split: {d['split']})", '',
                '| model | params | val acc | test acc |', '|---|---|---|---|']
        out += [f"| {n} | {m['params']} | {m['val_acc']:.3f} | {m['test_acc']:.3f} |" for n, m in d['models'].items()]
        out.append('')
    e = latest_result('export')
    if e:
        out += ['## Export (int8)', '', '| model | params | KB float | KB int8 | peak act. KB int8 | float | int8 |',
                '|---|---|---|---|---|---|---|']
        for n, m in e['data'].items():
            fl = m['float'] if np.ndim(m['float']) == 0 else m['float'][0]
            q8 = m['int8'] if np.ndim(m['int8']) == 0 else m['int8'][0]
            out.append(f"| {n} | {m['params']} | {m['kb_float']:.1f} | {m['kb_int8']:.1f} | "
                       f"{m['act_peak_kb_int8']:.2f} | {fl:.4f} | {q8:.4f} |")
        out.append('')
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, 'tables.md'), 'w') as fh:
        fh.write('\n'.join(out))
    return 'tables.md: written'


def cmd_report(args):
    if args.setup_photo:
        img = cv2.imread(args.setup_photo)
        if img is None:
            raise SystemExit(f'cannot read {args.setup_photo}')
        s = 1600 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        put_result('setup_photo', dict(source=os.path.basename(args.setup_photo)), blob=jpg_bytes(img, 90))
    for fn in (fig_f1, fig_f2, fig_f3, fig_f4, fig_f5, fig_f6, fig_f7, fig_f8, fig_f9, fig_f10, write_tables):
        try:
            print(fn())
        except Exception as e:
            print(f'{fn.__name__}: FAILED {type(e).__name__}: {e}')


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def live_args(p, calib=True):
        p.add_argument('--url', help='http://IP of the ESP32-CAM, `phone` (a phone browser over Wi-Fi), a webcam index, or a video file '
                                     '(remembered after first use)')
        p.add_argument('--backend', choices=['requests', 'capture', 'cv2'], default='requests',
                       help='requests: own MJPEG stream parser (default); capture: poll single frames '
                            '(slow on the stock sketch, ~2 s each); cv2: OpenCV stream reader')
        p.add_argument('--no-lock', action='store_true', help='do not lock ESP32 settings')
        p.add_argument('--aec', type=int, help='aec_value 0..1200 (default: last used, else 300)')
        p.add_argument('--agc', type=int, help='agc_gain 0..30 (default: last used, else 0)')
        p.add_argument('--framesize', type=int, help='numeric framesize, overrides reading SVGA from the board page')
        p.add_argument('--quality', type=int, help='JPEG quality 4-63, lower = better but bigger frames (default: '
                                                   'last used, else 10; 20 streams 2-3x faster, same geometry)')
        p.add_argument('--max-frames', type=int, help=argparse.SUPPRESS)
        if calib:
            p.add_argument('--calib', help="calibration to use: 'board', 'ruler' or an id (default: latest board)")

    def tape_args(p, length_required=True):
        p.add_argument('--length', type=float, required=length_required,
                       help='distance between the two tape centres [m]')
        p.add_argument('--color', help=f'tape colour preset {list(HSV_PRESETS)}; or click the tape in the window')
        p.add_argument('--min-area', type=int, default=25, help='minimum tape blob area [px]')

    def seg_args(p):
        sp = SegParams()
        p.add_argument('--tau-chroma', type=float, default=sp.tau_chroma)
        p.add_argument('--dark', type=float, default=sp.dark)
        p.add_argument('--bright', type=float, default=sp.bright)
        p.add_argument('--shadow', type=float, default=sp.shadow)
        p.add_argument('--min-obj-area', type=int, default=sp.min_area)

    sub.add_parser('selfcheck', help='synthetic tests S1-S7')

    p = sub.add_parser('stream', help='show live stream, lock camera settings')
    live_args(p, calib=False)
    p.add_argument('--scan', action='store_true', help='search the local /24 network for the board')
    p.add_argument('--auto', action='store_true', help='start with auto exposure/white balance (viewing only; key a toggles)')
    p.add_argument('--http', type=int, metavar='PORT', help='serve the view to a browser on this port instead of a window')
    p.add_argument('--host', default='127.0.0.1', help='address for --http (0.0.0.0 to reach it from the phone)')

    p = sub.add_parser('esp-capture', help='record bursts of still-scene frames from the ESP32-CAM (denoiser training data)')
    live_args(p, calib=False)
    p.add_argument('--burst', type=int, default=8, help='steady frames per scene (default 8: about 2 seconds still; more frames give a cleaner average)')
    p.add_argument('--steady', type=float, default=1.5, help='frames count as steady within this many pixels of one another (default 1.5)')
    p.add_argument('--script-bursts', type=int, help=argparse.SUPPRESS)

    p = sub.add_parser('denoise-train', help='train the grey denoiser (synthetic ESP-style pairs + the scenes recorded with esp-capture)')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--patch', type=int, default=128)
    p.add_argument('--ch', type=int, default=48, help='channels of the first level (the network grows with it)')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--synth', type=int, default=8000, help='synthetic training pairs (clean room pictures degraded like the ESP does)')
    p.add_argument('--sources', type=int, default=800, help='clean pictures to draw the synthetic pairs from')
    p.add_argument('--real-share', type=float, default=0.15, help='share of the training patches taken from the recorded scenes')
    p.add_argument('--val', type=int, default=2, help='recorded scenes held out to test it (the ones with the best targets)')
    p.add_argument('--save-every', type=int, default=250, help='steps between checkpoints (the run resumes from the last one)')
    p.add_argument('--fresh', action='store_true', help='ignore a saved checkpoint and start over')

    p = sub.add_parser('log-tags', help='the labels of the recent room logs: camera, setup, object, usable for training')
    p.add_argument('--last', type=int, default=80, help='how many of the newest logs')

    p = sub.add_parser('rig-eval', help='phone alone, ESP32-CAM alone and both together, refitted from the stored frames of the rig logs')
    p.add_argument('--last', type=int, help='only the newest N rig logs')
    p.add_argument('--min-id', type=int, default=0, help='only logs from this measurement id on')
    p.add_argument('--corrected', action='store_true', help='also with the latest corrector')
    p.add_argument('--flip-until', type=int, default=4590, help='the ESP frames of logs up to this measurement id were stored mirrored '
                   '(the app flips the ESP picture before forwarding it from then on): flipped back here')

    p = sub.add_parser('denoise-eval', help='does the denoiser help the measurement? corner jitter and plane fit on the recorded scenes')
    p.add_argument('--last', type=int, help='only the newest N scenes')

    p = sub.add_parser('calib-board', help='checkerboard calibration (reference)')
    live_args(p, calib=False)
    p.add_argument('--cols', type=int, default=9, help='inner corners per row')
    p.add_argument('--rows', type=int, default=6, help='inner corners per column')
    p.add_argument('--square', type=float, default=0.025, help='square size [m]')
    p.add_argument('--views', type=int, help='views to collect (default 12 with --screen, else 20)')
    p.add_argument('--auto', action='store_true', help='capture automatically when the board moved')
    p.add_argument('--interval', type=float, default=1.2, help='auto: minimum seconds between captures')
    p.add_argument('--steady', type=float, default=0.0,
                   help='auto: seconds the board must be held still (0: off; smeared views are dropped at '
                        'calibration anyway)')
    p.add_argument('--screen', nargs='?', const='auto', metavar='NAME',
                   help='show the checkerboard full-screen on a monitor (default: the largest; '
                        'names from xrandr, e.g. HDMI-A-1, eDP-1) and wave the camera at it')
    p.add_argument('--from-db', metavar='SESSION', help="recalibrate stored views ('all' or a session)")

    p = sub.add_parser('calib-ruler', help="ruler calibration (supervisor's method)")
    live_args(p, calib=False)
    tape_args(p, length_required=False)
    p.add_argument('--distances', help='planned distances, comma separated [m]; else typed per capture')
    p.add_argument('--fit-k1', action='store_true', help='also fit radial distortion k1 (needs edge placements)')
    p.add_argument('--use-board', action='store_true', help='undistort with the board calibration first')
    p.add_argument('--from-db', metavar='SESSION', help="refit stored observations ('latest' or a session)")

    p = sub.add_parser('calib-nominal', help='datasheet estimate of f when there is no calibration target')
    p.add_argument('--dfov', type=float, default=66.0, help='lens diagonal field of view [deg] (default 66, stock lens)')
    p.add_argument('--hfov', type=float, help='lens horizontal field of view [deg], instead of --dfov')
    p.add_argument('--size', type=int, nargs=2, default=[800, 600], metavar=('W', 'H'), help='resolution')

    p = sub.add_parser('calib-phone', help='Android phone factory intrinsics via adb (no target needed)')
    p.add_argument('--size', type=int, nargs=2, default=[1600, 1200], metavar=('W', 'H'),
                   help='resolution of the phone video stream')

    p = sub.add_parser('fisher', help='estimability report for the ruler design')
    p.add_argument('--session', help='ruler session (default: latest)')
    p.add_argument('--plan', metavar='ZMIN:ZMAX:N', help='planned design instead of stored data')
    p.add_argument('--spacing', choices=['z', 'w'], default='w', help='plan spacing: uniform in Z or in 1/Z')
    p.add_argument('--positions', default='0', help='plan: ruler centre positions as fractions of the half width')
    p.add_argument('--length', type=float, default=0.30, help='plan: ruler length [m]')
    p.add_argument('--delta', type=float, default=0.0, help='plan: nominal offset [m]')
    p.add_argument('--f', type=float, help='plan: nominal focal length [px]')
    p.add_argument('--sigma', type=float, help='pixel noise [px] (default: fit residual, or 1 for a plan)')
    p.add_argument('--bootstrap', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('measure', help='live: distance to a ruler of known length')
    live_args(p)
    tape_args(p, length_required=False)
    p.add_argument('--width', type=float, help='instead of a taped ruler: one object of this colour and width [m]')
    p.add_argument('--shape', choices=['cylinder', 'flat'], default='cylinder',
                   help='with --width: round object (can, bottle) or a flat face square to the camera')
    p.add_argument('--min-obj-area', type=int, default=200, help='with --width: minimum blob area [px]')
    p.add_argument('--detect', action='store_true',
                   help='with --width: find objects with a pretrained detector instead of by colour')
    p.add_argument('--object-class', help="with --detect: default target class, e.g. 'cup', 'bottle', 'vase'")
    p.add_argument('--min-score', type=float, default=0.5, help='with --detect: detection confidence threshold')
    p.add_argument('--truth', action='store_true', help='ask for the true distance at each log')
    p.add_argument('--lighting', default='', help='lighting label stored with each measurement')

    p = sub.add_parser('scene', help='live: approximate distance and size of every detected object')
    live_args(p)
    p.add_argument('--ref-width', type=float, help='width of one reference object in view [m]; click it')
    p.add_argument('--ref-shape', choices=['flat', 'cylinder'], default='flat')
    p.add_argument('--ref-class', help="pick the reference by class instead of clicking, e.g. 'cell phone'")
    p.add_argument('--no-refine', action='store_true', help='use the raw detector boxes (no tight outline)')
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--lighting', default='')

    p = sub.add_parser('room', help='live: object size in the checkerboard room (floor and walls, 10 mm squares)')
    live_args(p)
    p.add_argument('--image', help='measure one photo instead of the live stream')
    p.add_argument('--out', help='with --image: write the annotated image here instead of showing it')
    p.add_argument('--report', metavar='JPG', help='with --image: wireframe-mesh analysis report (JPEG, plus .json and '
                                                    '.obj next to it); live: key m writes one into reports/')
    p.add_argument('--cad', action='store_true', help='also write a solid CAD model (STEP + FreeCAD file) with every report; '
                                                       'live: key k writes the selected object (or all) into reports/')
    p.add_argument('--name', action='store_true', help='name each object (open-vocabulary CLIP) in the report')
    p.add_argument('--denoise', action='store_true', help='clean every frame with the trained denoiser (denoise-train) before measuring: for the ESP32-CAM')
    p.add_argument('--max-side', type=int, default=1000,
                   help='measure on frames shrunk to this many px (faster; the window shows the same size)')
    p.add_argument('--shape', choices=['auto'] + list(ROOM_SHAPES), default='auto',
                   help='shape fitted to each object: auto (default; chosen from the object\'s CLIP name), or round '
                        '(rounded box), box, dome, cylinder, sphere, revolve; key s cycles')
    p.add_argument('--seg', choices=['room', 'detect', 'salient', 'none'], default='salient',
                   help='find objects where the room pattern is hidden (default), with the COCO detector, '
                        'as salient objects (BiRefNet: anything, anywhere on the plane), or not at all')
    p.add_argument('--table', action='store_true',
                   help='boards scattered on a table instead of a room: salient objects anywhere on the table, '
                        'f from the boards (3+ coplanar boards seen at an angle) unless --calib is given')
    p.add_argument('--refine-f', action='store_true', help='with --table and --calib: refine f from the boards, '
                                                           'keeping the calibration\'s distortion')
    p.add_argument('--learned-outline', action='store_true',
                   help='outlines from the trained outline network (train-outline) instead of the built-in ones')
    p.add_argument('--sizer', action='store_true',
                   help='sizes from the sizer (all shapes fitted, a network picks; slower per frame)')
    p.add_argument('--corrector', action='store_true',
                   help='apply the trained room corrector (only to objects like those it was trained on)')
    p.add_argument('--stack', type=int, help='average up to this many frames while the camera is still '
                                             '(less sensor noise; default 6 for the ESP32, 1 otherwise)')
    p.add_argument('--show-occ', action='store_true', help='start with the occupied cells shown (key o)')
    p.add_argument('--object-class', help="with --seg detect: only this COCO class, e.g. 'cup'")
    p.add_argument('--min-score', type=float, default=0.5, help='with --seg detect: confidence threshold')
    p.add_argument('--truth', type=float, nargs=3, metavar=('L', 'W', 'H'),
                   help='true size of the object [m], shown as the error and stored with each log')
    p.add_argument('--object', help=f'name of the object in the room, stored with each log (training data); '
                                    f'known sizes: {", ".join(ROOM_OBJECTS)}; a new name needs --truth once')
    p.add_argument('--lighting', default='', help='lighting label stored with each log')

    p = sub.add_parser('ground', help='live: distance and size of objects on the floor')
    live_args(p)
    seg_args(p)
    p.add_argument('--setup', action='store_true', help='calibrate the camera height h and tilt theta')
    p.add_argument('--height', type=float, help='setup: camera height [m]; measure: true object height [m]')
    p.add_argument('--sigma-h', type=float, default=0.005, help='setup: uncertainty of the measured h [m]')
    p.add_argument('--near-only', action='store_true', help='setup: use only the near tape of each placement')
    p.add_argument('--from-db', metavar='SESSION', help="setup: refit stored floor points ('latest' or session)")
    tape_args(p, length_required=False)
    p.add_argument('--truth', action='store_true', help='ask for the true distance at each log')
    p.add_argument('--width', type=float, help='true object width for logged measurements [m]')
    p.add_argument('--lighting', default='', help='lighting label')

    p = sub.add_parser('collect', help='record labelled frames for training')
    live_args(p)
    seg_args(p)
    p.add_argument('--object', required=True, help='object name (class label)')
    p.add_argument('--lighting', required=True, help='lighting condition, e.g. lamp, window, dark')
    p.add_argument('--width', type=float, help='true width [m] (remembered per object)')
    p.add_argument('--height', type=float, help='true height [m] (remembered per object)')
    p.add_argument('--distances', help='planned true distances D, comma separated [m]')
    p.add_argument('--burst', type=int, default=5, help='frames stored per placement')

    p = sub.add_parser('train-corrector', help='MLP 1: geometric residual corrector')
    p.add_argument('--room', action='store_true', help='train on `room` logs of known objects (L, W, H)')
    p.add_argument('--min-views', type=int, default=3, help='--room: use logs fitted from at least this many views')
    p.add_argument('--setup', choices=['room', 'outside room', 'all'], default='room', help='--room: which logs, by where the object stood (labels from log-tags)')
    p.add_argument('--camera', choices=['esp32', 'phone-usb', 'phone-wifi', 'rig'], help='--room: which dataset: the phone (default), esp32 (ESP32-CAM alone) or rig (phone and ESP32-CAM fused)')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--folds', type=int, default=5, help='object-grouped folds')
    p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('room-labels', help='true outlines for the outline network, one pass per photo (resumable)')
    p.add_argument('--limit', type=int, help='process at most this many logs this time')

    p = sub.add_parser('train-outline', help='train the outline network (GPU; checkpoints, resumes after a cut)')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--folds', type=int, default=5, help='object-grouped folds')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fresh', action='store_true', help='ignore a checkpoint left by an interrupted run')

    p = sub.add_parser('sizer-features', help='six-shape features for logs that have none yet (resumable)')
    p.add_argument('--limit', type=int, help='process at most this many logs this time')

    p = sub.add_parser('train-sizer', help='train the sizer on cached six-shape features (fast)')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--folds', type=int, default=5, help='object-grouped folds')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warm', action='store_true', help='continue from the last sizer instead of starting fresh')
    p.add_argument('--fresh', action='store_true', help='ignore a checkpoint left by an interrupted run')

    p = sub.add_parser('train-classifier', help='network 2: object identification')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--rgb', action='store_true', help='32x32x3 input instead of grayscale')
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--test-sessions', help='comma-separated held-out sessions (default: last day or lighting)')
    p.add_argument('--seed', type=int, default=0)

    sub.add_parser('export', help='int8 quantization for STM32')

    p = sub.add_parser('report', help='all figures and tables from saved data')
    p.add_argument('--setup-photo', help='store a photo of the setup for F1')

    args = ap.parse_args(argv)
    if args.cmd != 'selfcheck':
        require_selfcheck()
    {'selfcheck': cmd_selfcheck, 'stream': cmd_stream, 'esp-capture': cmd_esp_capture, 'denoise-eval': cmd_denoise_eval, 'rig-eval': cmd_rig_eval, 'log-tags': cmd_log_tags, 'denoise-train': cmd_denoise_train, 'calib-board': cmd_calib_board,
     'calib-ruler': cmd_calib_ruler, 'calib-nominal': cmd_calib_nominal, 'calib-phone': cmd_calib_phone, 'fisher': cmd_fisher, 'measure': cmd_measure, 'ground': cmd_ground,
     'collect': cmd_collect, 'scene': cmd_scene, 'room': cmd_room, 'train-corrector': cmd_train_corrector, 'sizer-features': cmd_sizer_features,
     'train-sizer': cmd_train_sizer, 'room-labels': cmd_room_labels, 'train-outline': cmd_train_outline, 'train-classifier': cmd_train_classifier,
     'export': cmd_export, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
