"""Live-camera verdict: garbage, clean, or too blurry to judge.

Why this was rewritten
----------------------
The old gate had one rule for every scene (every frame must be a confident,
large box) and a third answer, "possible garbage, not clear enough". The
result: garbage that is far away or slightly blurry never passed (its box is
small and its confidence lower), while the "possible garbage" message showed a
percentage on clean scenes too, which looked like a wrong detection.

Now the camera always gives a firm answer, using evidence that behaves
differently for near and far objects:

* NEAR garbage (big box) must be confident: a large object the model is only
  half sure about is usually texture (cloth, grass, shadows), not litter.
* FAR / blurry garbage (small box) is allowed a lower confidence, because a
  small object is naturally scored lower, but the SAME box must show up in
  several frames at the same place.
* A very high confidence in two frames is accepted straight away.

Everything is tunable on Render with SMARTCITY_LIVE_* variables (see
``settings_from_env``) and every decision is written to the server log with
the numbers behind it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LiveSettings:
    # Lowest confidence (0-1) the model is asked to report for camera frames.
    floor: float = 0.30
    # Confidences below are in percent (0-100).
    far_conf: float = 38.0       # weakest box that can count as evidence
    far_mean: float = 44.0       # far/blurry: average over the matching frames
    far_best: float = 48.0       # far/blurry: at least one frame this good
    near_mean: float = 55.0      # near: average over the matching frames
    near_best: float = 60.0      # near: at least one frame this good
    strong_conf: float = 72.0    # confident enough to accept on its own
    near_area: float = 0.04      # boxes covering this much of the frame are "near"
    min_support: int = 2         # frames in which the same box must appear
    min_overlap: float = 0.20    # IoU for "same place" (also centre-inside test)
    # "Too blurry" answer: OFF by default (0). A plain wall or sheet also has very
    # little detail, and calling a clean wall "blurry" is a wrong answer. Set
    # SMARTCITY_LIVE_MIN_SHARPNESS (e.g. 6) if you want it.
    min_sharpness: float = 0.0
    max_frames: int = 3          # frames analysed per press
    # "Possible garbage, not clear enough": only when the model saw something at
    # least this confident (percent) AND the picture is blurry / dark / washed out.
    # Clean, sharp pictures never get this answer.
    possible_conf: float = 40.0
    poor_sharpness: float = 60.0   # whole-frame sharpness below this = blurry
    poor_blur_fraction: float = 0.35  # share of the frame with no detail
    poor_dark: float = 50.0        # average brightness below this = too dark
    poor_bright: float = 225.0     # average brightness above this = washed out
    # Wall-clock limit (seconds) for analysing one press.
    time_budget: float = 22.0

    def tightened(self, slider_percent: float) -> "LiveSettings":
        """The on-page confidence slider can make the camera stricter, never blind."""
        if slider_percent <= self.far_conf:
            return self
        return replace(
            self,
            far_conf=max(self.far_conf, slider_percent),
            far_mean=max(self.far_mean, slider_percent),
            far_best=max(self.far_best, slider_percent),
            near_mean=max(self.near_mean, slider_percent),
            near_best=max(self.near_best, slider_percent),
            strong_conf=max(self.strong_conf, slider_percent),
        )


def settings_from_env() -> LiveSettings:
    d = LiveSettings()

    def number(name, default, cast=float):
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    return LiveSettings(
        floor=number("SMARTCITY_LIVE_FLOOR", d.floor),
        far_conf=number("SMARTCITY_LIVE_FAR_CONF", d.far_conf),
        far_mean=number("SMARTCITY_LIVE_FAR_MEAN", d.far_mean),
        far_best=number("SMARTCITY_LIVE_FAR_BEST", d.far_best),
        near_mean=number("SMARTCITY_LIVE_NEAR_MEAN", d.near_mean),
        near_best=number("SMARTCITY_LIVE_NEAR_BEST", d.near_best),
        strong_conf=number("SMARTCITY_LIVE_STRONG_CONF", d.strong_conf),
        near_area=number("SMARTCITY_LIVE_NEAR_AREA", d.near_area),
        min_support=number("SMARTCITY_LIVE_MIN_SUPPORT", d.min_support, int),
        min_overlap=number("SMARTCITY_LIVE_OVERLAP", d.min_overlap),
        min_sharpness=number("SMARTCITY_LIVE_MIN_SHARPNESS", d.min_sharpness),
        max_frames=number("SMARTCITY_LIVE_FRAMES", d.max_frames, int),
        possible_conf=number("SMARTCITY_LIVE_POSSIBLE_CONF", d.possible_conf),
        poor_sharpness=number("SMARTCITY_LIVE_POOR_SHARPNESS", d.poor_sharpness),
        poor_blur_fraction=number("SMARTCITY_LIVE_POOR_BLUR_FRACTION", d.poor_blur_fraction),
        poor_dark=number("SMARTCITY_LIVE_POOR_DARK", d.poor_dark),
        poor_bright=number("SMARTCITY_LIVE_POOR_BRIGHT", d.poor_bright),
        time_budget=number("SMARTCITY_LIVE_BUDGET", d.time_budget),
    )


@dataclass(frozen=True)
class SceneVerdict:
    state: str                      # "garbage" | "possible" | "clean" | "blurry"
    message: str
    anchor_index: int | None = None  # frame that shows the garbage best
    support: int = 0
    frames: int = 0
    best_confidence: float = 0.0
    mean_confidence: float | None = None
    box_ratio: float | None = None
    rule: str = ""

    @property
    def is_garbage(self) -> bool:
        return self.state == "garbage"


CLEAN_MESSAGE = "✅ No garbage detected. This area looks clean."
BLURRY_MESSAGE = (
    "The picture is too blurry to check. Hold the phone steady, "
    "let it focus for a second and press Capture again."
)


def poor_quality(scan, settings: LiveSettings) -> str:
    """Why a frame is hard to judge ("" when it is fine): blurry, dark or washed out."""
    if scan.brightness < settings.poor_dark:
        return "dark"
    if scan.brightness > settings.poor_bright:
        return "washed out"
    if scan.sharpness < settings.poor_sharpness or scan.blur_fraction >= settings.poor_blur_fraction:
        return "blurry"
    return ""


def possible_message(percent: float) -> str:
    return (
        f"Possible garbage ({percent:.0f}% at best) but not clear enough. "
        "Move closer, improve the light and hold the camera steady."
    )


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a, b) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _same_place(a, b, min_overlap: float) -> bool:
    """Same box if they overlap enough, or the centre of one lies inside the other."""
    if _iou(a, b) >= min_overlap:
        return True

    def centre_inside(inner, outer):
        cx, cy = (inner[0] + inner[2]) / 2.0, (inner[1] + inner[3]) / 2.0
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    return centre_inside(a, b) or centre_inside(b, a)


def frame_has_candidate(scan, settings: LiveSettings) -> bool:
    return any(box[4] >= settings.far_conf for box in scan.boxes)


def decide_scene(scans, settings: LiveSettings) -> SceneVerdict:
    """Combine the scanned frames into one firm answer."""
    frames = len(scans)
    best_overall = max(
        (box[4] for scan in scans for box in scan.boxes), default=0.0
    )
    sharpest = max((scan.sharpness for scan in scans), default=0.0)

    def verdict(state, message, **extra):
        return SceneVerdict(
            state=state,
            message=message,
            frames=frames,
            best_confidence=best_overall,
            **extra,
        )

    # Candidate anchors: the strongest boxes anywhere.
    candidates = sorted(
        (
            box
            for scan in scans
            for box in scan.boxes
            if box[4] >= settings.far_conf
        ),
        key=lambda box: box[4],
        reverse=True,
    )[:8]

    best_plan = None
    for anchor in candidates:
        matches = []  # (confidence, frame index): best matching box per frame
        for frame_index, scan in enumerate(scans):
            hits = [
                box for box in scan.boxes
                if box[4] >= settings.far_conf
                and _same_place(anchor, box, settings.min_overlap)
            ]
            if hits:
                top = max(hits, key=lambda box: box[4])
                matches.append((top[4], frame_index))
        support = len(matches)
        if support < settings.min_support:
            continue

        confs = [conf for conf, _ in matches]
        mean_conf = sum(confs) / support
        top_conf = max(confs)
        near = _area(anchor) >= settings.near_area
        strong_frames = sum(1 for conf in confs if conf >= settings.strong_conf)

        rule = None
        if strong_frames >= min(2, settings.min_support):
            rule = "strong"
        elif near and mean_conf >= settings.near_mean and top_conf >= settings.near_best:
            rule = "near"
        elif (not near) and mean_conf >= settings.far_mean and top_conf >= settings.far_best:
            rule = "far"

        if rule:
            anchor_frame = max(matches)[1]
            plan = (top_conf, rule, anchor_frame, support, mean_conf, _area(anchor))
            if best_plan is None or plan[0] > best_plan[0]:
                best_plan = plan

    if best_plan:
        top_conf, rule, anchor_frame, support, mean_conf, box_area = best_plan
        return verdict(
            "garbage",
            "Garbage confirmed.",
            anchor_index=anchor_frame,
            support=support,
            mean_confidence=mean_conf,
            box_ratio=box_area,
            rule=rule,
        )

    # Nothing confirmed. If the model did see something garbage-like but the
    # picture is blurry / dark / washed out, say "possible garbage" with the
    # percentage so the user retakes it. A clean, sharp picture never gets here.
    if best_overall >= settings.possible_conf:
        top_scan = max(
            scans,
            key=lambda scan: max((box[4] for box in scan.boxes), default=0.0),
        )
        reason = poor_quality(top_scan, settings)
        if reason:
            return verdict(
                "possible",
                possible_message(best_overall),
                rule="possible-" + reason,
            )

    # Nothing confirmed. If nothing at all was seen AND the picture is mush,
    # say so instead of claiming the area is clean.
    if (
        settings.min_sharpness > 0
        and best_overall < settings.far_conf
        and sharpest < settings.min_sharpness
    ):
        return verdict("blurry", BLURRY_MESSAGE, rule="blurry")

    return verdict("clean", CLEAN_MESSAGE, rule="clean")


def can_still_confirm(
    frames_done: int,
    candidate_frames: int,
    total_frames: int,
    settings: LiveSettings,
) -> bool:
    """False once the remaining frames can no longer reach ``min_support``."""
    return candidate_frames + (total_frames - frames_done) >= settings.min_support
