"""
Sauron Edge — Monitoring Dashboard (port 5000)

Runs as a daemon Flask thread alongside the main processing loop.
The dashboard is read-only — it never touches hardware or the main loop.

Routes:
  GET /              → HTML dashboard (video + sensors + telemetry + health)
  GET /video_feed    → MJPEG stream (annotated frames from SharedState)
  GET /api/status    → JSON: full component health + latest sensor readings
  GET /api/telemetry → JSON: last 5 telemetry payloads
  GET /api/errors    → JSON: recent WARNING+ log entries

All routes tolerate SharedState returning None / empty values — the dashboard
is always up even if the camera is still initialising.
"""
from __future__ import annotations

import logging
import time
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np
from flask import Flask, Response, jsonify

if TYPE_CHECKING:
    from sauron_edge.shared_state import SharedState

logger = logging.getLogger(__name__)

_DASHBOARD_PORT = 5000
_JPEG_QUALITY = 70

# Blank grey frame served when no camera data is available yet
_PLACEHOLDER_FRAME: np.ndarray = np.full((480, 640, 3), 40, dtype=np.uint8)
cv2.putText(
    _PLACEHOLDER_FRAME,
    "Waiting for camera...",
    (160, 240),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 220, 0),
    2,
)


def _make_app(state: "SharedState") -> Flask:
    """Build and return the Flask application, closing over `state`."""

    # Silence Flask's access log — already very noisy alongside component logs
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    app = Flask(__name__, static_folder=None)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]

    # ------------------------------------------------------------------
    # MJPEG generator
    # ------------------------------------------------------------------

    def _generate_frames():
        """Yield annotated JPEG frames forever for the MJPEG endpoint."""
        logger.info("Dashboard: MJPEG stream client connected")
        frames_sent = 0
        while True:
            frame = state.get_frame()
            if frame is None:
                frame = _PLACEHOLDER_FRAME

            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                logger.debug("Dashboard: JPEG encode failed — sending placeholder")
                time.sleep(0.05)
                continue

            frames_sent += 1
            if frames_sent % 100 == 0:
                logger.debug("Dashboard: MJPEG frames sent = %d", frames_sent)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(0.04)  # ~25 fps cap to the browser

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/video_feed")
    def video_feed():
        return Response(
            _generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        h = state.get_health()
        ncoords = state.get_latest_ncoords()

        # Derive an overall health verdict
        issues = []
        if not h.camera_alive:
            issues.append("Camera capture thread is not running")
        if not h.sensor_thread_alive:
            issues.append("Sensor reader thread is not running")
        if not h.gps_locked:
            issues.append("GPS has no satellite fix")
        if not h.imu_calibrated:
            issues.append("IMU is not yet calibrated")
        if not h.publisher_connected:
            issues.append("Greengrass IPC publisher is not connected")
        if h.total_publish_failures > 0:
            issues.append(
                f"{h.total_publish_failures} publish failure(s) since startup"
            )

        last_pub_ago = None
        if h.last_publish_ts is not None:
            last_pub_ago = round(time.time() - h.last_publish_ts, 1)

        return jsonify(
            {
                "overall": "degraded" if issues else "ok",
                "issues": issues,
                "health": {
                    "camera_alive": h.camera_alive,
                    "sensor_thread_alive": h.sensor_thread_alive,
                    "gps_locked": h.gps_locked,
                    "imu_calibrated": h.imu_calibrated,
                    "publisher_connected": h.publisher_connected,
                    "uptime_s": round(h.uptime_s, 1),
                    "frames_processed": h.frames_processed,
                    "total_detections": h.total_detections,
                    "total_publishes": h.total_publishes,
                    "total_publish_failures": h.total_publish_failures,
                    "last_publish_ago_s": last_pub_ago,
                },
                "current_detections": len(ncoords),
                "current_ncoords": ncoords,
            }
        )

    @app.route("/api/telemetry")
    def api_telemetry():
        packets = state.get_recent_telemetry()
        return jsonify(
            {
                "count": len(packets),
                "packets": list(reversed(packets)),  # newest first
            }
        )

    @app.route("/api/errors")
    def api_errors():
        entries = state.get_errors()
        return jsonify(
            {
                "count": len(entries),
                "entries": [
                    {
                        "ts": e.timestamp,
                        "ts_str": time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.localtime(e.timestamp)
                        ),
                        "level": e.level,
                        "logger": e.logger_name,
                        "message": e.message,
                    }
                    for e in reversed(entries)  # newest first
                ],
            }
        )

    @app.route("/")
    def index():
        return Response(_DASHBOARD_HTML, mimetype="text/html")

    return app


# ------------------------------------------------------------------
# Dashboard HTML
# ------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sauron Edge — Monitor</title>
<style>
  :root {
    --green: #00ff88;
    --green-dim: #00cc66;
    --orange: #ff9900;
    --red: #ff3333;
    --bg: #0a0a0a;
    --panel: #111111;
    --border: #1e2e1e;
    --text: #cccccc;
    --mono: 'Courier New', Courier, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    padding: 12px;
  }
  h1 {
    color: var(--green);
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 12px;
  }
  .grid {
    display: grid;
    grid-template-columns: 640px 1fr;
    gap: 12px;
  }
  .right-col { display: flex; flex-direction: column; gap: 12px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
  }
  .panel h2 {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    color: var(--green-dim);
    text-transform: uppercase;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
  }
  img#feed {
    display: block;
    width: 640px;
    height: 480px;
    object-fit: cover;
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .kv { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 2px 0; }
  .kv .label { color: #888; }
  .kv .value { color: var(--green); font-weight: bold; }
  .kv .value.warn { color: var(--orange); }
  .kv .value.err  { color: var(--red); }
  .pill {
    display: inline-block;
    border-radius: 3px;
    padding: 1px 7px;
    font-size: 0.75rem;
    font-weight: bold;
  }
  .pill.ok   { background: #002200; color: var(--green); border: 1px solid #004400; }
  .pill.warn { background: #221100; color: var(--orange); border: 1px solid #442200; }
  .pill.err  { background: #220000; color: var(--red);    border: 1px solid #440000; }
  #overall-badge { font-size: 1rem; margin-bottom: 8px; }
  .issue-list { list-style: none; font-size: 0.78rem; color: var(--orange); }
  .issue-list li::before { content: "⚠ "; }
  .telem-packet {
    font-size: 0.72rem;
    border: 1px solid #1a2a1a;
    border-radius: 3px;
    margin-bottom: 6px;
    padding: 6px 8px;
    background: #0d140d;
  }
  .telem-packet .ts { color: #556655; margin-bottom: 3px; }
  .telem-packet .coords { color: var(--green-dim); word-break: break-all; }
  .err-entry {
    font-size: 0.72rem;
    padding: 3px 0;
    border-bottom: 1px solid #1a1a1a;
    word-break: break-word;
  }
  .err-entry .level-WARNING { color: var(--orange); }
  .err-entry .level-ERROR   { color: var(--red); }
  .err-entry .level-CRITICAL { color: var(--red); font-weight: bold; }
  .err-entry .ets { color: #444; font-size: 0.68rem; }
  .ncoords-dot { display: inline-block; width: 6px; height: 6px;
                 background: var(--green); border-radius: 50%; margin-right: 4px; }
  #detections-list { font-size: 0.78rem; color: var(--green-dim); }
  .fps-note { color: #444; font-size: 0.68rem; margin-top: 4px; }
</style>
</head>
<body>
<h1>&#9679; SAURON EDGE &mdash; MONITORING DASHBOARD</h1>

<div class="grid">
  <!-- Left: live video -->
  <div>
    <img id="feed" src="/video_feed" alt="Live feed">
    <p class="fps-note">Live feed &mdash; annotated with MOG2 detections.</p>
  </div>

  <!-- Right: panels -->
  <div class="right-col">

    <!-- Overall health -->
    <div class="panel" id="health-panel">
      <h2>Service Health</h2>
      <div id="overall-badge"><span class="pill ok" id="overall-pill">OK</span></div>
      <ul class="issue-list" id="issues-list"></ul>
      <div class="kv"><span class="label">Uptime</span><span class="value" id="uptime">—</span></div>
      <div class="kv"><span class="label">Frames processed</span><span class="value" id="frames">—</span></div>
      <div class="kv"><span class="label">Total detections</span><span class="value" id="total-det">—</span></div>
      <div class="kv"><span class="label">Publishes OK</span><span class="value" id="pub-ok">—</span></div>
      <div class="kv"><span class="label">Publish failures</span><span class="value" id="pub-fail">—</span></div>
      <div class="kv"><span class="label">Last publish</span><span class="value" id="last-pub">—</span></div>
    </div>

    <!-- Component status -->
    <div class="panel">
      <h2>Components</h2>
      <div class="kv"><span class="label">Camera thread</span><span class="pill ok" id="c-camera">—</span></div>
      <div class="kv" style="margin-top:4px"><span class="label">Sensor thread</span><span class="pill ok" id="c-sensor">—</span></div>
      <div class="kv" style="margin-top:4px"><span class="label">GPS fix</span><span class="pill ok" id="c-gps">—</span></div>
      <div class="kv" style="margin-top:4px"><span class="label">IMU calibrated</span><span class="pill ok" id="c-imu">—</span></div>
      <div class="kv" style="margin-top:4px"><span class="label">IPC publisher</span><span class="pill ok" id="c-pub">—</span></div>
    </div>

    <!-- Current detections -->
    <div class="panel">
      <h2>Live Detections (<span id="det-count">0</span>)</h2>
      <div id="detections-list">No detections in current frame.</div>
    </div>

  </div>
</div>

<!-- Second row: telemetry + error log -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">

  <div class="panel">
    <h2>Recent Telemetry (last 5 sent)</h2>
    <div id="telemetry-list"><em style="color:#444">No telemetry yet.</em></div>
  </div>

  <div class="panel">
    <h2>Warning / Error Log</h2>
    <div id="error-log"><em style="color:#444">No issues logged.</em></div>
  </div>

</div>

<script>
  const $ = id => document.getElementById(id);

  function pill(el, ok, labels) {
    el.className = 'pill ' + (ok ? 'ok' : 'err');
    el.textContent = ok ? (labels ? labels[0] : 'OK') : (labels ? labels[1] : 'FAIL');
  }

  function fmtUptime(s) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return (h > 0 ? h + 'h ' : '') + (m > 0 ? m + 'm ' : '') + sec + 's';
  }

  async function fetchStatus() {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      const h = d.health;

      // Overall
      const op = $('overall-pill');
      if (d.overall === 'ok') {
        op.className = 'pill ok'; op.textContent = 'ALL SYSTEMS OK';
      } else {
        op.className = 'pill err'; op.textContent = 'DEGRADED';
      }

      const il = $('issues-list');
      il.innerHTML = d.issues.map(i => `<li>${i}</li>`).join('');

      // Counters
      $('uptime').textContent    = fmtUptime(h.uptime_s);
      $('frames').textContent    = h.frames_processed.toLocaleString();
      $('total-det').textContent = h.total_detections.toLocaleString();
      $('pub-ok').textContent    = h.total_publishes.toLocaleString();

      const pf = $('pub-fail');
      pf.textContent = h.total_publish_failures;
      pf.className   = 'value' + (h.total_publish_failures > 0 ? ' warn' : '');

      $('last-pub').textContent = h.last_publish_ago_s !== null
        ? h.last_publish_ago_s + 's ago'
        : 'never';

      // Component pills
      pill($('c-camera'), h.camera_alive,      ['ALIVE',      'DEAD']);
      pill($('c-sensor'), h.sensor_thread_alive,['ALIVE',     'DEAD']);
      pill($('c-gps'),    h.gps_locked,         ['LOCKED',    'NO FIX']);
      pill($('c-imu'),    h.imu_calibrated,      ['CALIBRATED','UNCAL']);
      pill($('c-pub'),    h.publisher_connected, ['CONNECTED', 'DISCONNECTED']);

      // Current detections
      $('det-count').textContent = d.current_detections;
      const dl = $('detections-list');
      if (d.current_ncoords.length === 0) {
        dl.innerHTML = '<span style="color:#444">No detections in current frame.</span>';
      } else {
        dl.innerHTML = d.current_ncoords.map((nc, i) =>
          `<div><span class="ncoords-dot"></span>#${i+1} &nbsp; x=${nc.xnorm.toFixed(3)} &nbsp; y=${nc.ynorm.toFixed(3)}</div>`
        ).join('');
      }

    } catch(e) { console.error('status fetch error', e); }
  }

  async function fetchTelemetry() {
    try {
      const r = await fetch('/api/telemetry');
      const d = await r.json();
      const el = $('telemetry-list');
      if (d.count === 0) {
        el.innerHTML = '<em style="color:#444">No telemetry yet.</em>';
        return;
      }
      el.innerHTML = d.packets.map((p, idx) => {
        const ts = new Date(p.timestamp * 1000).toISOString().replace('T',' ').slice(0,23);
        const nc = p.ncoords.length;
        const gps = p.cameras.gps;
        const imu = p.cameras.imu;
        return `<div class="telem-packet">
          <div class="ts">#${idx+1} &nbsp; ${ts}</div>
          <div class="coords">
            GPS: ${gps.lat.toFixed(6)}, ${gps.lon.toFixed(6)} |
            IMU: hdg=${imu.heading.toFixed(1)}&deg; pitch=${imu.pitch.toFixed(1)}&deg; roll=${imu.roll.toFixed(1)}&deg; |
            FOV: ${p.cameras.fov}&deg; |
            Detections: ${nc}
            ${nc > 0 ? '<br>' + p.ncoords.map(c => `(${c.xnorm.toFixed(3)}, ${c.ynorm.toFixed(3)})`).join(' ') : ''}
          </div>
        </div>`;
      }).join('');
    } catch(e) { console.error('telemetry fetch error', e); }
  }

  async function fetchErrors() {
    try {
      const r = await fetch('/api/errors');
      const d = await r.json();
      const el = $('error-log');
      if (d.count === 0) {
        el.innerHTML = '<em style="color:#444">No issues logged.</em>';
        return;
      }
      el.innerHTML = d.entries.slice(0, 20).map(e =>
        `<div class="err-entry">
          <span class="ets">${e.ts_str}</span>
          <span class="level-${e.level}"> [${e.level}]</span>
          ${e.message}
        </div>`
      ).join('');
    } catch(e) { console.error('errors fetch error', e); }
  }

  // Initial load
  fetchStatus(); fetchTelemetry(); fetchErrors();

  // Poll
  setInterval(fetchStatus,    1500);
  setInterval(fetchTelemetry, 2000);
  setInterval(fetchErrors,    3000);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# Thread launcher
# ------------------------------------------------------------------


def start_dashboard(state: "SharedState", port: int = _DASHBOARD_PORT) -> threading.Thread:
    """
    Launch the Flask dashboard in a background daemon thread.

    Returns the thread object (already started).  The thread is a daemon so
    it exits automatically when the main process does.
    """
    app = _make_app(state)

    def _run():
        logger.info("Dashboard: Flask server starting on http://0.0.0.0:%d", port)
        try:
            # threaded=True so concurrent MJPEG + API requests don't block each other
            app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False, debug=False)
        except Exception as exc:
            logger.error("Dashboard: Flask server crashed — %s", exc, exc_info=True)

    t = threading.Thread(target=_run, name="dashboard", daemon=True)
    t.start()
    logger.info("Dashboard: daemon thread launched (port=%d)", port)
    return t
