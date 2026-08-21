#!/usr/bin/env python3
"""
Stereo live view + frame capture over HTTP.

Run on Pi, open http://<PI_IP>:8080 in laptop browser.
Shows live side-by-side stereo feed. Click Capture to save a frame pair.
Frames are saved as calib_0001_left.png / calib_0001_right.png for calibration.

Streaming notes:
- One capture thread per camera (parallel, each at full sensor rate).
- MJPEG stream skips frames when nothing changed (bandwidth saver).
- FPS capped to STREAM_FPS_CAP to avoid flooding 2.4GHz WiFi.
"""

import os
import re
import time
import json
import base64
import signal
import socket
import threading
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np
from picamera2 import Picamera2

HOST = "0.0.0.0"
PORT = 8080
DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "calibration"
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_BIN = BASE_DIR / "build" / "stereo_pipeline"
OUTPUT_DIR = BASE_DIR / "output"
JPEG_QUALITY = 65          # lower = smaller frames = smoother on WiFi
STREAM_FPS_CAP = 24        # max frames/sec pushed to the browser


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StereoCamera:
    def __init__(self, left_num=0, right_num=1, stream_w=768, stream_h=432,
                 full_w=1536, full_h=864):
        self.left_num = left_num
        self.right_num = right_num
        self.stream_w = stream_w
        self.stream_h = stream_h
        self.full_w = full_w
        self.full_h = full_h
        self.left_cam = None
        self.right_cam = None
        self.lock = threading.Lock()
        self.left_frame = None
        self.right_frame = None
        self.left_ts = 0.0
        self.right_ts = 0.0
        self.running = False
        self._threads = []

    def _open_one(self, num, label):
        print(f"  Opening camera {num} ({label}) at {self.stream_w}x{self.stream_h}...",
              flush=True)
        cam = Picamera2(num)
        config = cam.create_video_configuration(
            main={"size": (self.stream_w, self.stream_h), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        return cam

    def open(self):
        self.left_cam = self._open_one(self.left_num, "LEFT")
        time.sleep(0.3)
        self.right_cam = self._open_one(self.right_num, "RIGHT")
        time.sleep(0.3)

        self.running = True
        self._start_threads()
        print(f"  Streaming at {self.stream_w}x{self.stream_h}.", flush=True)

    def _start_threads(self):
        self._threads = [
            threading.Thread(target=self._capture_loop, args=("left",), daemon=True),
            threading.Thread(target=self._capture_loop, args=("right",), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _stop_threads(self):
        self.running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        with self.lock:
            self.left_frame = None
            self.right_frame = None

    def _capture_loop(self, side):
        cam = getattr(self, f"{side}_cam")
        while self.running:
            try:
                frame = cam.capture_array("main")
                ts = time.time()
                with self.lock:
                    setattr(self, f"{side}_frame", frame)
                    setattr(self, f"{side}_ts", ts)
            except Exception as e:
                print(f"  {side} capture error: {e}", flush=True)
                time.sleep(0.1)

    def grab_full_pair(self):
        """Stop streaming, grab full-res pair, restart streaming."""
        self._stop_threads()
        left = None
        right = None
        gap_ms = 0.0
        try:
            # Reconfigure both cameras to full res first, so the two
            # captures below run back-to-back with an honest gap.
            for cam in (self.left_cam, self.right_cam):
                cam.stop()
                cam.configure(cam.create_video_configuration(
                    main={"size": (self.full_w, self.full_h), "format": "RGB888"}
                ))
                cam.start()
            time.sleep(0.5)  # exposure/AWB settle at new mode

            left = self.left_cam.capture_array("main")
            t_left = time.time()
            right = self.right_cam.capture_array("main")
            t_right = time.time()
            gap_ms = (t_right - t_left) * 1000.0
        except Exception as e:
            print(f"  Full-res grab error: {e}", flush=True)
        finally:
            try:
                for cam in (self.left_cam, self.right_cam):
                    cam.stop()
                    cam.configure(cam.create_video_configuration(
                        main={"size": (self.stream_w, self.stream_h), "format": "RGB888"}
                    ))
                    cam.start()
            except Exception as e:
                print(f"  Stream restore error: {e}", flush=True)
            time.sleep(0.3)
            with self.lock:
                self.left_frame = None
                self.right_frame = None
            self.running = True
            self._start_threads()

        if left is None or right is None:
            return None
        return left, right, gap_ms

    def get_stacked_frame(self):
        with self.lock:
            if self.left_frame is None or self.right_frame is None:
                return None
            if self.left_frame.shape != self.right_frame.shape:
                return None
            left = self.left_frame.copy()
            right = self.right_frame.copy()
            gap_ms = abs(self.right_ts - self.left_ts) * 1000.0
        stack = np.hstack([left, right])
        return stack, gap_ms

    def stamp(self):
        with self.lock:
            return (self.left_ts, self.right_ts)

    def get_pair(self):
        with self.lock:
            if self.left_frame is None or self.right_frame is None:
                return None
            return (
                self.left_frame.copy(),
                self.right_frame.copy(),
                abs(self.right_ts - self.left_ts) * 1000.0,
            )

    def close(self):
        self._stop_threads()
        for attr in ("left_cam", "right_cam"):
            cam = getattr(self, attr, None)
            if cam:
                try:
                    cam.stop()
                    cam.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        print("  Cameras closed.", flush=True)


class StreamHandler(BaseHTTPRequestHandler):
    stereo = None
    outdir = None
    frame_count = 0
    fps_start = 0
    current_fps = 0
    processing = False

    def setup(self):
        super().setup()
        # Send frames immediately instead of waiting for Nagle batching
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self._html())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_stamp = None
            frame_interval = 1.0 / STREAM_FPS_CAP
            next_due = time.time()
            try:
                while True:
                    result = self.stereo.get_stacked_frame()
                    if result is None:
                        time.sleep(0.05)
                        continue
                    frame, _ = result
                    stamp = self.stereo.stamp()
                    if stamp != last_stamp:
                        _, jpeg = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                        )
                        blob = jpeg.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(blob)}\r\n\r\n".encode()
                        )
                        self.wfile.write(blob)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        self.frame_count += 1
                        last_stamp = stamp
                        now = time.time()
                        if self.fps_start == 0:
                            self.fps_start = now
                        if now - self.fps_start >= 1.0:
                            self.current_fps = self.frame_count / (now - self.fps_start)
                            self.frame_count = 0
                            self.fps_start = now
                    delay = next_due - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    next_due = time.time() + frame_interval
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/status":
            count = self._count_captures()
            result = self.stereo.get_stacked_frame()
            gap = result[1] if result else 0
            data = {"captures": count, "gap_ms": round(gap, 1),
                    "fps": round(self.current_fps, 1)}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/capture":
            result = self.stereo.grab_full_pair()
            if result is None:
                self.send_error(503, "Full-res grab failed")
                return
            left, right, gap_ms = result

            count = self._count_captures() + 1
            self.outdir.mkdir(parents=True, exist_ok=True)
            left_path = self.outdir / f"calib_{count:04d}_left.png"
            right_path = self.outdir / f"calib_{count:04d}_right.png"

            cv2.imwrite(
                str(left_path), cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                str(right_path), cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {
                "ok": True,
                "count": count,
                "gap_ms": round(gap_ms, 1),
                "left": str(left_path),
                "right": str(right_path),
            }
            self.wfile.write(json.dumps(data).encode())
        elif self.path == "/process" or self.path.startswith("/process?"):
            if StreamHandler.processing:
                self._json_response({"ok": False, "error": "Already processing"}, code=409)
                return
            qs = parse_qs(urlparse(self.path).query)
            try:
                downscale = int(qs.get("downscale", ["1"])[0])
            except ValueError:
                downscale = 1
            StreamHandler.processing = True
            try:
                result = self.stereo.grab_full_pair()
                if result is None:
                    self._json_response({"ok": False, "error": "Full-res grab failed"})
                    return
                left, right, gap_ms = result

                count = self._count_captures() + 1
                self.outdir.mkdir(parents=True, exist_ok=True)
                left_path = self.outdir / f"calib_{count:04d}_left.png"
                right_path = self.outdir / f"calib_{count:04d}_right.png"
                cv2.imwrite(str(left_path), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(right_path), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

                data = self._run_pipeline(left_path, right_path, downscale)
                data["count"] = count
                data["gap_ms"] = round(gap_ms, 1)
                data["left_png"] = self._jpg_data_url(left)
                data["right_png"] = self._jpg_data_url(right)
                self._json_response(data)
            finally:
                StreamHandler.processing = False
        else:
            self.send_error(404)

    def _json_response(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    @staticmethod
    def _jpg_data_url(img_rgb):
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def _run_pipeline(self, left_path, right_path, downscale=1):
        cmd = [str(PIPELINE_BIN), "--left", str(left_path),
               "--right", str(right_path), "--save-all"]
        if downscale > 1:
            cmd += ["--downscale", str(downscale)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Pipeline timed out (300s)"}

        out = proc.stdout or ""
        if proc.returncode != 0:
            return {"ok": False,
                    "error": (proc.stderr or "pipeline failed")[-800:]}

        stats = {}
        m = re.search(r"Valid pixels:\s*\d+ / \d+ \(([\d.]+)%\)", out)
        if m:
            stats["valid_pct"] = float(m.group(1))
        m = re.search(r"Z min: ([\d.]+) mm \| max: ([\d.]+) mm \| mean: ([\d.]+) mm", out)
        if m:
            stats["z_min"] = float(m.group(1))
            stats["z_max"] = float(m.group(2))
            stats["z_mean"] = float(m.group(3))
        m = re.search(r"Texture mask: \d+ px \(([\d.]+)%\)", out)
        if m:
            stats["masked_pct"] = float(m.group(1))
        m = re.search(r"Total:\s*([\d.]+) ms", out)
        if m:
            stats["total_ms"] = float(m.group(1))

        def b64(name):
            p = OUTPUT_DIR / name
            if p.exists():
                return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            return None

        return {"ok": True, "stats": stats,
                "depth_png": b64("depth.png"),
                "disp_png": b64("disparity.png")}

    def _count_captures(self):
        if not self.outdir.exists():
            return 0
        return len(list(self.outdir.glob("calib_*_left.png")))

    def _html(self):
        return b"""<!DOCTYPE html>
<html>
<head>
<title>Stereo Calibration Capture</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: monospace; text-align: center; }
  h1 { padding: 10px; font-size: 18px; }
  #feed { max-width: 100%; height: auto; border: 2px solid #333; }
  #status { padding: 8px; font-size: 14px; color: #aaa; }
  #count { font-size: 28px; color: #4f4; font-weight: bold; }
  button {
    margin: 10px; padding: 12px 30px; font-size: 16px; font-family: monospace;
    background: #2a2; color: #fff; border: none; border-radius: 6px; cursor: pointer;
  }
  button:active { background: #181; }
  button:disabled { background: #555; cursor: wait; }
  #flash { color: #ff0; font-size: 14px; min-height: 20px; }
  #procbtn { background: #e80; }
  #procbtn:active { background: #c60; }
  #stats { white-space: pre-wrap; color: #4f4; font-size: 13px; padding: 6px; min-height: 18px; }
  #results img { max-width: 48%; border: 2px solid #333; margin: 4px; }
  #results h3 { font-size: 13px; color: #aaa; display: inline; margin: 10px; }
</style>
</head>
<body>
  <h1>Stereo Live View + Calibration Capture</h1>
  <img id="feed" src="/stream" />
  <div id="status">
    Gap: <span id="gap">--</span> ms |
    Captures: <span id="count">0</span> |
    FPS: <span id="fps">--</span>
  </div>
  <button id="capbtn" onclick="capture()">CAPTURE</button>
  <button id="procbtn" onclick="processPair()">PROCESS + SHOW DEPTH</button>
  <label style="font-size:14px;"><input type="checkbox" id="fastmode"/> Fast mode (half-res, ~2-3x faster)</label>
  <div id="flash"></div>
  <div id="stats"></div>
  <div id="results">
    <h3>Left capture</h3><h3>Right capture</h3><br/>
    <img id="leftimg" alt=""/>
    <img id="rightimg" alt=""/><br/>
    <h3>Disparity</h3><h3>Depth (near=red, far=blue, invalid=black)</h3><br/>
    <img id="dispimg" alt=""/>
    <img id="depthimg" alt=""/>
  </div>
<script>
async function capture() {
  const btn = document.getElementById("capbtn");
  btn.disabled = true;
  btn.textContent = "SAVING...";
  try {
    const r = await fetch("/capture", {method: "POST"});
    const d = await r.json();
    document.getElementById("count").textContent = d.count;
    document.getElementById("flash").textContent =
      "Saved #" + d.count + " (gap " + d.gap_ms + "ms)";
    setTimeout(() => document.getElementById("flash").textContent = "", 1500);
  } catch(e) {
    document.getElementById("flash").textContent = "Error: " + e;
  }
  btn.disabled = false;
  btn.textContent = "CAPTURE";
}
async function processPair() {
  const btn = document.getElementById("procbtn");
  const cap = document.getElementById("capbtn");
  btn.disabled = true;
  cap.disabled = true;
  const t0 = Date.now();
  btn.textContent = "PROCESSING 0s...";
  const tick = setInterval(() => {
    btn.textContent = "PROCESSING " + ((Date.now() - t0) / 1000 | 0) + "s...";
  }, 500);
  try {
    const fast = document.getElementById("fastmode").checked ? 2 : 1;
    const r = await fetch("/process?downscale=" + fast, {method: "POST"});
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "pipeline failed");
    document.getElementById("count").textContent = d.count;
    let s = "Capture #" + d.count + " (gap " + d.gap_ms + "ms)";
    const st = d.stats || {};
    if (st.valid_pct !== undefined) s += "\\nValid depth: " + st.valid_pct + "%";
    if (st.masked_pct !== undefined) s += " | texture-masked: " + st.masked_pct + "%";
    if (st.z_min !== undefined)
      s += "\\nZ: " + st.z_min + " - " + st.z_max + " mm (mean " + st.z_mean + " mm)";
    if (st.total_ms !== undefined) s += "\\nPipeline time: " + (st.total_ms / 1000).toFixed(1) + "s";
    document.getElementById("stats").textContent = s;
    if (d.disp_png) document.getElementById("dispimg").src = d.disp_png;
    if (d.depth_png) document.getElementById("depthimg").src = d.depth_png;
    if (d.left_png) document.getElementById("leftimg").src = d.left_png;
    if (d.right_png) document.getElementById("rightimg").src = d.right_png;
  } catch(e) {
    document.getElementById("flash").textContent = "Error: " + e;
  }
  clearInterval(tick);
  btn.disabled = false;
  cap.disabled = false;
  btn.textContent = "PROCESS + SHOW DEPTH";
}
setInterval(async () => {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    document.getElementById("count").textContent = d.captures;
    document.getElementById("gap").textContent = d.gap_ms;
    document.getElementById("fps").textContent = d.fps;
  } catch(e) {}
}, 1000);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Stereo live view + capture")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--right", type=int, default=1)
    parser.add_argument("--stream-w", type=int, default=768,
                        help="Stream width (default: 768)")
    parser.add_argument("--stream-h", type=int, default=432,
                        help="Stream height (default: 432)")
    parser.add_argument("--full-w", type=int, default=1536,
                        help="Full capture width (default: 1536)")
    parser.add_argument("--full-h", type=int, default=864,
                        help="Full capture height (default: 864)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing = len(list(outdir.glob("calib_*_left.png")))
    if existing > 0:
        print(f"  Found {existing} existing captures in {outdir}")

    print("Opening cameras...")
    stereo = StereoCamera(args.left, args.right,
                          args.stream_w, args.stream_h,
                          args.full_w, args.full_h)
    stereo.open()

    StreamHandler.stereo = stereo
    StreamHandler.outdir = outdir

    server = ThreadedHTTPServer((HOST, args.port), StreamHandler)
    print(f"\nLive view: http://0.0.0.0:{args.port}")
    print(f"Captures save to: {outdir}")
    print("Press Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        print("\nStopping...")
        stereo.close()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    finally:
        stereo.close()


if __name__ == "__main__":
    main()
