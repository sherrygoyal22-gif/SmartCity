"""Multi-frame confirmation for the live camera.

Why this exists
---------------
A single camera frame is noisy: blur, exposure changes and video compression
move the model's confidence by 10-20 points from one frame to the next. The
old rule ("every frame must be >= 60%") therefore rejected real garbage almost
every time, while a simple "lower the threshold" would let clean scenes through.

Instead a scene is accepted only when ALL of these agree:

1. enough frames contain a confident garbage box   (min_hits)
2. those boxes are big enough to be real garbage   (min_box_ratio)
3. the average confidence of those frames is solid (min_mean_confidence)
4. the detections sit in the same part of the view (min_overlap)

A random false alarm on a clean scene normally fails #1, #3 or #4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateVerdict:
    confirmed: bool
    reason: str
    hits: int
    frames: int
    best_confidence: float
    mean_hit_confidence: float | None


def frame_is_hit(inspection, hit_confidence: float, min_box_ratio: float) -> bool:
    """True when one camera frame shows a confident, reasonably large garbage box."""
    return bool(
        inspection.detection_count
        and inspection.highest_confidence is not None
        and inspection.highest_confidence >= hit_confidence
        and (inspection.largest_box_ratio or 0.0) >= min_box_ratio
    )


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def evaluate_camera_frames(
    inspections,
    *,
    hit_confidence: float,
    min_hits: int,
    min_mean_confidence: float,
    min_box_ratio: float,
    min_overlap: float,
) -> GateVerdict:
    frames = len(inspections)
    confidences = [
        (item.highest_confidence or 0.0) if item.detection_count else 0.0
        for item in inspections
    ]
    best = max(confidences, default=0.0)

    hit_items = [
        item for item in inspections
        if frame_is_hit(item, hit_confidence, min_box_ratio)
    ]
    hits = len(hit_items)

    def verdict(ok: bool, reason: str, mean_conf=None) -> GateVerdict:
        return GateVerdict(ok, reason, hits, frames, best, mean_conf)

    # 1 + 2: enough confident, large-enough frames
    if hits < min_hits:
        confident = sum(
            1 for item in inspections
            if item.detection_count
            and (item.highest_confidence or 0.0) >= hit_confidence
        )
        if confident >= min_hits:
            return verdict(
                False,
                "Something was spotted but it is too small in the view. "
                "Move closer so the garbage fills more of the screen, then try again.",
            )
        if best >= hit_confidence * 0.75:
            return verdict(
                False,
                f"Possible garbage ({best:.0f}% at best) but not clear enough. "
                "Move closer, improve the light and hold the camera steady.",
            )
        return verdict(
            False,
            "No garbage detected in the camera view. The area looks clean.",
        )

    mean_conf = sum(item.highest_confidence for item in hit_items) / hits

    # 3: average confidence of the confirming frames
    if mean_conf < min_mean_confidence:
        return verdict(
            False,
            f"Detection was not confident enough ({mean_conf:.0f}%). "
            "Get closer to the garbage and hold steady, then try again.",
            mean_conf,
        )

    # 4: the detections must stay in the same place across frames
    anchor = max(hit_items, key=lambda item: item.highest_confidence)
    anchor_box = getattr(anchor, "enclosing_box", None)
    if anchor_box is not None:
        consistent = sum(
            1 for item in hit_items
            if getattr(item, "enclosing_box", None) is None
            or _iou(anchor_box, item.enclosing_box) >= min_overlap
        )
        if consistent < min_hits:
            return verdict(
                False,
                "The detection kept moving between frames, so it was not confirmed. "
                "Hold the camera steady on the garbage and try again.",
                mean_conf,
            )

    return verdict(True, "Garbage confirmed.", mean_conf)
