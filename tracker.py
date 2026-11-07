"""
Temporal detection tracker for stabilizing object recognition across video frames.

Eliminates flickering by:
  - Tracking objects across frames using IoU matching
  - Smoothing bounding boxes with exponential moving average (EMA)
  - Requiring objects to appear in multiple frames before showing (confirmation)
  - Keeping objects visible for several frames after they disappear (persistence / hysteresis)
  - Smoothing confidence scores to prevent jitter
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


def iou(box_a: list, box_b: list) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackedObject:
    """A single tracked object across multiple video frames."""
    track_id: int
    class_name: str
    class_id: int
    # Smoothed (EMA) bounding box [x1, y1, x2, y2] — this is what gets displayed
    smoothed_bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    # Raw bbox from latest detection
    raw_bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    # Smoothed confidence
    smoothed_conf: float = 0.0
    raw_conf: float = 0.0
    # How many consecutive frames this object has been detected
    hits: int = 0
    # How many consecutive frames since last detection (0 = detected this frame)
    misses: int = 0
    # Frame index when first detected
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    # Whether this track is confirmed (visible to user)
    confirmed: bool = False


class DetectionTracker:
    """
    Frame-to-frame object tracker using IoU matching and EMA smoothing.

    Parameters:
        iou_threshold:      Minimum IoU to match a detection to an existing track.
        bbox_smoothing:     EMA factor for bounding box positions (0=no smoothing, 1=frozen).
                            Higher values = smoother but more laggy.
        conf_smoothing:     EMA factor for confidence scores.
        min_hits_to_confirm: Number of consecutive frames an object must be detected
                             before it becomes visible (prevents single-frame ghosts).
        max_misses:         Number of frames to keep showing an object after it disappears
                             from raw detections (prevents flicker on brief occlusions).
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        bbox_smoothing: float = 0.55,
        conf_smoothing: float = 0.45,
        min_hits_to_confirm: int = 3,
        max_misses: int = 8,
    ):
        self.iou_threshold = iou_threshold
        self.bbox_smoothing = bbox_smoothing
        self.conf_smoothing = conf_smoothing
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_misses = max_misses

        self.tracks: List[TrackedObject] = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Accept raw detections from a single frame and return stabilized detections.

        Args:
            raw_detections: List of dicts with keys: class, confidence, bbox, class_id, inference_ms

        Returns:
            Stabilized detection list with smooth bounding boxes and filtered flickers.
        """
        self.frame_count += 1

        # Mark all tracks as not-yet-matched this frame
        matched_track_indices = set()
        matched_det_indices = set()

        # Greedy IoU matching: match detections to existing tracks (same class, best IoU)
        # Build cost matrix
        match_pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(raw_detections):
                if det["class_id"] != track.class_id:
                    continue
                score = iou(track.smoothed_bbox, det["bbox"])
                if score >= self.iou_threshold:
                    match_pairs.append((score, ti, di))

        # Sort by IoU descending and greedily assign
        match_pairs.sort(key=lambda x: x[0], reverse=True)
        for _, ti, di in match_pairs:
            if ti in matched_track_indices or di in matched_det_indices:
                continue
            matched_track_indices.add(ti)
            matched_det_indices.add(di)

            track = self.tracks[ti]
            det = raw_detections[di]

            # Update raw values
            track.raw_bbox = det["bbox"]
            track.raw_conf = det["confidence"]

            # EMA-smooth the bounding box
            alpha = self.bbox_smoothing
            track.smoothed_bbox = [
                alpha * track.smoothed_bbox[i] + (1 - alpha) * det["bbox"][i]
                for i in range(4)
            ]

            # EMA-smooth confidence
            c_alpha = self.conf_smoothing
            track.smoothed_conf = c_alpha * track.smoothed_conf + (1 - c_alpha) * det["confidence"]

            track.hits += 1
            track.misses = 0
            track.last_seen_frame = self.frame_count

            # Promote to confirmed after enough consecutive hits
            if track.hits >= self.min_hits_to_confirm:
                track.confirmed = True

        # Handle unmatched tracks (object missing this frame)
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_indices:
                track.misses += 1
                # Decay confidence while missing
                track.smoothed_conf *= 0.85

        # Create new tracks for unmatched detections
        for di, det in enumerate(raw_detections):
            if di not in matched_det_indices:
                new_track = TrackedObject(
                    track_id=self.next_id,
                    class_name=det["class"],
                    class_id=det["class_id"],
                    smoothed_bbox=list(det["bbox"]),
                    raw_bbox=list(det["bbox"]),
                    smoothed_conf=det["confidence"],
                    raw_conf=det["confidence"],
                    hits=1,
                    misses=0,
                    first_seen_frame=self.frame_count,
                    last_seen_frame=self.frame_count,
                    confirmed=(self.min_hits_to_confirm <= 1),
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # Remove tracks that have been missing too long
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Build stabilized output — only confirmed tracks
        stabilized = []
        for track in self.tracks:
            if not track.confirmed:
                continue

            stabilized.append({
                "class": track.class_name,
                "confidence": round(track.smoothed_conf, 3),
                "bbox": [int(round(v)) for v in track.smoothed_bbox],
                "class_id": track.class_id,
                "track_id": track.track_id,
                "age": track.hits,
                "missing": track.misses > 0,
            })

        return stabilized

    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.frame_count = 0
        self.next_id = 1
