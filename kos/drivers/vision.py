"""
KOS V2.0 — Vision Driver (YOLO/OCR Wrapper).

Parses object detection output (bounding boxes) into graph topology.
Spatial proximity of detected objects creates weighted edges.
"""


class VisionDriver:
    def __init__(self, kernel, lexicon):
        self.kernel = kernel
        self.lexicon = lexicon

    def ingest_yolo(self, bounding_boxes: list):
        """
        Parses local YOLOv8 output into Graph topology.
        bounding_boxes: list of (object_label_1, object_label_2) tuples
        representing spatially proximate detected objects.
        """
        for obj1, obj2 in bounding_boxes:
            id1 = self.lexicon.get_or_create_id(obj1)
            id2 = self.lexicon.get_or_create_id(obj2)
            self.kernel.add_connection(
                id1, id2, 0.5,
                f"[VISION] Spatial proximity detected.")
