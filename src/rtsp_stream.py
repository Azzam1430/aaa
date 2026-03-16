import cv2
import threading
import logging
import time
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class RTSPStream:
    """Handles RTSP stream capture in a background thread with a frame queue."""

    def __init__(self, url: str, camera_id: str, queue_size: int = 10, reconnect_delay: float = 5.0):
        self.url = url
        self.camera_id = camera_id
        self.queue = Queue(maxsize=queue_size)
        self.reconnect_delay = reconnect_delay

        self._cap = None
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"rtsp-{self.camera_id}", daemon=True)
        self._thread.start()
        logger.info(f"[{self.camera_id}] Stream started: {self.url}")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info(f"[{self.camera_id}] Stream stopped.")

    def read(self, timeout: float = 1.0):
        """Return the latest frame or None if unavailable."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def is_alive(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    def _open_capture(self) -> bool:
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Reduce internal buffer to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            logger.warning(f"[{self.camera_id}] Failed to open: {self.url}")
            return False
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[{self.camera_id}] Connected — {w}x{h} @ {fps:.1f} fps")
        return True

    def _capture_loop(self):
        while self._running:
            if not self._open_capture():
                time.sleep(self.reconnect_delay)
                continue

            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"[{self.camera_id}] Frame read failed — reconnecting...")
                    break

                # Drop oldest frame if queue is full to keep latency low
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                self.queue.put(frame)

            self._cap.release()
            if self._running:
                time.sleep(self.reconnect_delay)


class MultiStreamManager:
    """Manages multiple RTSP streams and provides unified frame access."""

    def __init__(self):
        self._streams: dict[str, RTSPStream] = {}

    def add_camera(self, camera_id: str, url: str, **kwargs):
        stream = RTSPStream(url=url, camera_id=camera_id, **kwargs)
        self._streams[camera_id] = stream

    def start_all(self):
        for stream in self._streams.values():
            stream.start()

    def stop_all(self):
        for stream in self._streams.values():
            stream.stop()

    def read_frame(self, camera_id: str, timeout: float = 1.0):
        stream = self._streams.get(camera_id)
        if stream is None:
            raise KeyError(f"Unknown camera: {camera_id}")
        return stream.read(timeout=timeout)

    def camera_ids(self) -> list[str]:
        return list(self._streams.keys())

    def status(self) -> dict[str, bool]:
        return {cid: s.is_alive() for cid, s in self._streams.items()}
