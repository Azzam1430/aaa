"""
Main application — runs YOLOv8 detection on multiple RTSP streams.

Usage:
    python src/app.py --config config/cameras.yaml             # web viewer at :8080
    python src/app.py --config config/cameras.yaml --port 9090 # custom port
    python src/app.py --config config/cameras.yaml --show      # OpenCV desktop windows
    python src/app.py --config config/cameras.yaml --save-frames
"""
import argparse
import cv2
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from rtsp_stream import MultiStreamManager
from detector import YOLOv8Detector
from web_viewer import WebViewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config: dict, show: bool = False, save_frames: bool = False, port: int = 8080):
    # --- Setup streams ---
    manager = MultiStreamManager()
    for cam in config["cameras"]:
        manager.add_camera(
            camera_id=cam["id"],
            url=cam["url"],
            queue_size=cam.get("queue_size", 10),
            reconnect_delay=cam.get("reconnect_delay", 5.0),
        )

    # --- Setup detector ---
    det_cfg = config.get("detector", {})
    detector = YOLOv8Detector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence", 0.5),
        iou=det_cfg.get("iou", 0.45),
        classes=det_cfg.get("classes"),
        device=det_cfg.get("device", "cpu"),
        use_tracking=det_cfg.get("tracking", True),
    )

    # --- Setup web viewer ---
    viewer = WebViewer(camera_ids=manager.camera_ids(), port=port)
    viewer.start()
    logger.info(f"Open your browser at  http://0.0.0.0:{port}")

    output_dir = Path(config.get("output", {}).get("frames_dir", "output/frames"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manager.start_all()
    logger.info(f"Monitoring cameras: {manager.camera_ids()}")

    try:
        while True:
            for cam_id in manager.camera_ids():
                frame = manager.read_frame(cam_id, timeout=0.1)
                if frame is None:
                    continue

                result = detector.detect(frame, camera_id=cam_id)
                display = result.annotated_frame if result.annotated_frame is not None else frame

                # Push to web viewer
                viewer.push_frame(cam_id, display)

                if result.detections:
                    classes_found = {d.class_name for d in result.detections}
                    logger.info(f"[{cam_id}] {len(result.detections)} detection(s): {classes_found}")

                if save_frames and result.detections:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = output_dir / f"{cam_id}_{ts}.jpg"
                    cv2.imwrite(str(path), display)

                if show:
                    cv2.imshow(f"Camera: {cam_id}", display)

            if show:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        manager.stop_all()
        if show:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + RTSP multi-camera detector")
    parser.add_argument("--config", default="config/cameras.yaml", help="Path to config YAML")
    parser.add_argument("--port", type=int, default=8080, help="Web viewer port (default 8080)")
    parser.add_argument("--show", action="store_true", help="Also open OpenCV desktop windows")
    parser.add_argument("--save-frames", action="store_true", help="Save frames that have detections")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, show=args.show, save_frames=args.save_frames, port=args.port)


if __name__ == "__main__":
    main()
