"""
Web viewer — serves live annotated MJPEG streams in the browser.

Access at: http://<host>:8080
"""
import cv2
import threading
import logging
from flask import Flask, Response, render_template_string

logger = logging.getLogger(__name__)

# --- HTML page ---------------------------------------------------------------

_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Camera Monitor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif; padding: 16px; }
    h1 { font-size: 1.2rem; margin-bottom: 14px; color: #aaa; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
      gap: 12px;
    }
    .cam-card { background: #222; border-radius: 8px; overflow: hidden; }
    .cam-label {
      padding: 6px 12px; font-size: 0.85rem;
      background: #1a1a1a; color: #9cf; letter-spacing: .05em;
    }
    .cam-card img { width: 100%; display: block; }
    .no-signal { padding: 40px; text-align: center; color: #555; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Camera Monitor</h1>
  <div class="grid">
    {% for cam_id in camera_ids %}
    <div class="cam-card">
      <div class="cam-label">{{ cam_id }}</div>
      <img src="/stream/{{ cam_id }}"
           onerror="this.style.display='none';this.nextElementSibling.style.display='block'"
           alt="{{ cam_id }}">
      <div class="no-signal" style="display:none">No signal</div>
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""

# --- Viewer ------------------------------------------------------------------

class WebViewer:
    """
    Wraps a Flask app that serves per-camera MJPEG streams.

    Usage:
        viewer = WebViewer(camera_ids=['cam_01', 'cam_02'], port=8080)
        viewer.start()
        # push frames:
        viewer.push_frame('cam_01', annotated_frame)
    """

    def __init__(self, camera_ids: list[str], port: int = 8080, jpeg_quality: int = 80):
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._frames: dict[str, bytes] = {}
        self._locks: dict[str, threading.Lock] = {cid: threading.Lock() for cid in camera_ids}
        self._camera_ids = camera_ids
        self._app = self._build_app()

    def push_frame(self, camera_id: str, frame):
        """Call this with every new annotated frame (numpy BGR array)."""
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        data = buf.tobytes()
        with self._locks[camera_id]:
            self._frames[camera_id] = data

    def start(self):
        t = threading.Thread(target=self._serve, daemon=True, name="web-viewer")
        t.start()
        logger.info(f"Web viewer running at http://0.0.0.0:{self.port}")

    # -------------------------------------------------------------------------

    def _build_app(self):
        app = Flask(__name__)
        camera_ids = self._camera_ids

        @app.route("/")
        def index():
            return render_template_string(_PAGE, camera_ids=camera_ids)

        @app.route("/stream/<camera_id>")
        def stream(camera_id):
            if camera_id not in self._locks:
                return "Unknown camera", 404
            return Response(
                self._mjpeg_generator(camera_id),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        return app

    def _mjpeg_generator(self, camera_id: str):
        import time
        while True:
            with self._locks[camera_id]:
                data = self._frames.get(camera_id)
            if data:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                )
            else:
                time.sleep(0.05)

    def _serve(self):
        import logging as _log
        _log.getLogger("werkzeug").setLevel(_log.WARNING)
        self._app.run(host="0.0.0.0", port=self.port, threaded=True)
