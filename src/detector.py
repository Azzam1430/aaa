import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None


@dataclass
class DetectionResult:
    camera_id: str
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None


class YOLOv8Detector:
    """Wraps Ultralytics YOLOv8 with optional object tracking."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou: float = 0.45,
        classes: list[int] | None = None,
        device: str = "cpu",
        use_tracking: bool = True,
    ):
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.device = device
        self.use_tracking = use_tracking

        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Model loaded successfully.")

    def detect(self, frame: np.ndarray, camera_id: str = "cam") -> DetectionResult:
        result = DetectionResult(camera_id=camera_id, frame=frame)

        if self.use_tracking:
            outputs = self.model.track(
                frame,
                conf=self.confidence,
                iou=self.iou,
                classes=self.classes,
                device=self.device,
                persist=True,
                verbose=False,
            )
        else:
            outputs = self.model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou,
                classes=self.classes,
                device=self.device,
                verbose=False,
            )

        if outputs:
            raw = outputs[0]
            names = self.model.names

            boxes = raw.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if (self.use_tracking and box.id is not None) else None

                    result.detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=names.get(cls_id, str(cls_id)),
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            track_id=track_id,
                        )
                    )

            result.annotated_frame = raw.plot()

        return result


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on a frame (manual fallback)."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out
