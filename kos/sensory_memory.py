"""
KOS Sensory Memory — Ring buffer for visual and audio inputs.
"""

import time
from collections import deque


class SensoryMemory:
    """Timestamped ring buffers for YOLO detections and audio transcripts."""

    def __init__(self, max_visual=30, max_audio=20):
        self._visual = deque(maxlen=max_visual)
        self._audio = deque(maxlen=max_audio)

    def record_visual(self, detections: list):
        """Store timestamped YOLO detections."""
        self._visual.append({"time": time.time(), "detections": detections})

    def record_audio(self, transcript: str):
        """Store a timestamped transcript."""
        self._audio.append({"time": time.time(), "transcript": transcript})

    def get_recent_visual(self, seconds=30) -> list:
        """Return detections from the last N seconds."""
        cutoff = time.time() - seconds
        return [e for e in self._visual if e["time"] >= cutoff]

    def get_recent_audio(self, seconds=30) -> list:
        """Return transcripts from the last N seconds."""
        cutoff = time.time() - seconds
        return [e for e in self._audio if e["time"] >= cutoff]

    def search(self, query: str) -> list:
        """Search both buffers for keyword matches."""
        q = query.lower()
        results = []
        for entry in self._visual:
            for det in entry["detections"]:
                label = det.get("label", "")
                if q in label.lower():
                    results.append({"type": "visual", "time": entry["time"], "match": det})
        for entry in self._audio:
            if q in entry["transcript"].lower():
                results.append({"type": "audio", "time": entry["time"], "match": entry["transcript"]})
        return results

    def summary(self) -> str:
        """Human-readable summary of recent sensory input."""
        vis = self.get_recent_visual(30)
        aud = self.get_recent_audio(30)
        obj_count = sum(len(e["detections"]) for e in vis)
        utt_count = len(aud)
        return "I have seen %d objects and heard %d utterances in the last 30 seconds" % (
            obj_count, utt_count)
