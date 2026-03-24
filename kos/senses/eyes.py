"""KOS V6.0 — Eyes (Object Detection via YOLO)

Detects objects in images or live webcam feed using YOLOv8, then wires
the detected objects into the KOS knowledge graph with spatial
relationships (left_of, right_of, above, below, near, overlaps).

ultralytics and opencv-python are optional heavy dependencies.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
_YOLO = None
try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    pass

_cv2 = None
try:
    import cv2 as _cv2
except ImportError:
    pass


def _require_yolo():
    if _YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed.  Run:  pip install ultralytics"
        )


def _require_cv2():
    if _cv2 is None:
        raise RuntimeError(
            "OpenCV is not installed.  Run:  pip install opencv-python"
        )


# ---------------------------------------------------------------------------
# Spatial relationship helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Return (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_area(bbox: List[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-union of two bounding boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_spatial_relations(detections: List[Dict],
                              near_threshold: float = 150.0,
                              overlap_iou: float = 0.1) -> List[Dict]:
    """Compute pairwise spatial relationships between detected objects.

    Returns a list of dicts:
        {"subject": str, "relation": str, "object": str, "strength": float}
    """
    relations = []
    n = len(detections)
    for i in range(n):
        for j in range(i + 1, n):
            a = detections[i]
            b = detections[j]
            ba = a["bbox"]
            bb = b["bbox"]
            ca = _bbox_center(ba)
            cb = _bbox_center(bb)
            dist = _euclidean(ca, cb)
            la = a["label"]
            lb = b["label"]

            # Overlap
            overlap = _iou(ba, bb)
            if overlap >= overlap_iou:
                relations.append({
                    "subject": la, "relation": "overlaps",
                    "object": lb, "strength": round(overlap, 3),
                })

            # Nearness
            if dist < near_threshold:
                strength = round(1.0 - dist / near_threshold, 3)
                relations.append({
                    "subject": la, "relation": "near",
                    "object": lb, "strength": strength,
                })

            # Horizontal: left_of / right_of
            dx = cb[0] - ca[0]
            if abs(dx) > 30:
                if dx > 0:
                    relations.append({
                        "subject": la, "relation": "left_of",
                        "object": lb, "strength": 0.8,
                    })
                else:
                    relations.append({
                        "subject": la, "relation": "right_of",
                        "object": lb, "strength": 0.8,
                    })

            # Vertical: above / below
            dy = cb[1] - ca[1]
            if abs(dy) > 30:
                if dy > 0:
                    relations.append({
                        "subject": la, "relation": "above",
                        "object": lb, "strength": 0.8,
                    })
                else:
                    relations.append({
                        "subject": la, "relation": "below",
                        "object": lb, "strength": 0.8,
                    })

    return relations


# ---------------------------------------------------------------------------
# Eyes
# ---------------------------------------------------------------------------

class Eyes:
    """Object-detection sense organ for KOS.

    Parameters
    ----------
    model_path : str
        Path or name of a YOLO model weights file.
        Default is "yolov8n.pt" (nano — fast, CPU-friendly).
    confidence_threshold : float
        Minimum detection confidence to keep (0.0 - 1.0).
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None  # lazy-loaded
        self._last_labels: set = set()

    # -- Model management ---------------------------------------------------

    def _get_model(self):
        _require_yolo()
        if self._model is None:
            self._model = _YOLO(self.model_path)
        return self._model

    # -- Internal -----------------------------------------------------------

    def _parse_results(self, results) -> List[Dict]:
        """Convert YOLO results to a list of detection dicts."""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(boxes.cls[i])
                label = result.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                })
        return detections

    # -- Public API ---------------------------------------------------------

    def see_image(self, image_path: str) -> List[Dict]:
        """Run object detection on an image file.

        Returns
        -------
        list[dict]
            Each dict: {"label", "confidence", "bbox": [x1, y1, x2, y2]}
        """
        model = self._get_model()
        results = model(image_path, verbose=False)
        return self._parse_results(results)

    def see_webcam_frame(self, camera_index: int = 0) -> List[Dict]:
        """Capture one frame from the webcam and run detection.

        Returns
        -------
        list[dict]
            Detections for the captured frame.
        """
        _require_cv2()
        model = self._get_model()

        cap = _cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}"
            )
        try:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame from webcam")
            results = model(frame, verbose=False)
            return self._parse_results(results)
        finally:
            cap.release()

    def watch(self, callback: Callable[[List[Dict]], None],
              interval_sec: float = 1.0,
              max_frames: int = 100,
              camera_index: int = 0):
        """Continuous webcam monitoring.

        Calls *callback* whenever a new object label appears that was not
        present in the previous frame.

        Parameters
        ----------
        callback : callable
            Called with the full detection list whenever new objects appear.
        interval_sec : float
            Seconds between frame captures.
        max_frames : int
            Maximum number of frames to process before stopping.
        camera_index : int
            OpenCV camera device index.
        """
        _require_cv2()
        model = self._get_model()

        cap = _cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}"
            )

        print(f"[Eyes] Watching webcam (max {max_frames} frames).  "
              f"Ctrl+C to stop.")
        prev_labels: set = set()
        try:
            for frame_idx in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                detections = self._parse_results(results)
                current_labels = {d["label"] for d in detections}

                new_labels = current_labels - prev_labels
                if new_labels:
                    callback(detections)

                prev_labels = current_labels
                self._last_labels = current_labels
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n[Eyes] Watch stopped.")
        finally:
            cap.release()

    # -- KOS integration ----------------------------------------------------

    def process_for_kos(self, kernel, lexicon,
                        detections: List[Dict]) -> Dict:
        """Wire detected objects and their spatial relationships into the
        KOS knowledge graph.

        Parameters
        ----------
        kernel : kos.graph.KOSKernel
            The spreading-activation kernel.
        lexicon : kos.lexicon.KASMLexicon
            The semantic lexicon for UUID resolution.
        detections : list[dict]
            Output of see_image / see_webcam_frame.

        Returns
        -------
        dict
            {"nodes_created": int, "relations_created": int,
             "objects": list[str]}
        """
        nodes_created = 0
        relations_created = 0
        object_labels = []

        # 1. Create nodes for each unique detected label
        seen_labels = set()
        for det in detections:
            label = det["label"]
            if label in seen_labels:
                continue
            seen_labels.add(label)
            object_labels.append(label)

            uid = lexicon.get_or_create_id(label)
            kernel.add_node(uid)

            # Connect to a perceptual anchor
            vision_uid = lexicon.get_or_create_id("visual_object")
            kernel.add_node(vision_uid)
            kernel.add_connection(
                uid, vision_uid,
                weight=det["confidence"],
                source_text=f"YOLO detected: {label}",
            )
            nodes_created += 1

        # 2. Compute and wire spatial relationships
        relations = compute_spatial_relations(detections)
        for rel in relations:
            subj_uid = lexicon.get_or_create_id(rel["subject"])
            obj_uid = lexicon.get_or_create_id(rel["object"])
            rel_uid = lexicon.get_or_create_id(rel["relation"])

            kernel.add_node(rel_uid)
            kernel.add_connection(
                subj_uid, rel_uid,
                weight=rel["strength"],
                source_text=f"spatial: {rel['subject']} {rel['relation']} {rel['object']}",
            )
            kernel.add_connection(
                rel_uid, obj_uid,
                weight=rel["strength"],
                source_text=f"spatial: {rel['subject']} {rel['relation']} {rel['object']}",
            )
            relations_created += 1

        return {
            "nodes_created": nodes_created,
            "relations_created": relations_created,
            "objects": object_labels,
        }


# ---------------------------------------------------------------------------
# Module-level availability check
# ---------------------------------------------------------------------------

if _YOLO is None:
    print("[Eyes] WARNING: ultralytics not installed. "
          "Object detection will not work.  pip install ultralytics")
if _cv2 is None:
    print("[Eyes] WARNING: opencv-python not installed. "
          "Webcam capture will not work.  pip install opencv-python")
