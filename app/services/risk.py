from __future__ import annotations


# Risk level is decided by the detection percentage (average confidence):
#   HIGH   -> more than 80%
#   MEDIUM -> more than 45%
#   LOW    -> everything else
HIGH_RISK_ABOVE_PERCENT = 80.0
MEDIUM_RISK_ABOVE_PERCENT = 45.0


def validate_detection(detection_count, confidence, minimum_confidence=None):
    """Any detected garbage is a valid report; no garbage means no case."""
    if not detection_count:
        return False, "No garbage was detected in this image."
    if confidence is None:
        return True, "Garbage detected."
    return True, f"Garbage detected with {confidence:.1f}% average confidence."


def assess_risk(detection_count, confidence, box_area_ratio, repeated_location=False):
    """Return (level, score, reason). The score is the detection percentage."""
    if not detection_count:
        return "LOW", 0, "No garbage detected; no cleanup priority."

    percent = max(0.0, min(confidence or 0.0, 100.0))
    if percent > HIGH_RISK_ABOVE_PERCENT:
        level = "HIGH"
    elif percent > MEDIUM_RISK_ABOVE_PERCENT:
        level = "MEDIUM"
    else:
        level = "LOW"

    reason = f"{level.title()} risk: garbage detected with {percent:.1f}% confidence"
    if box_area_ratio:
        reason += f", covering about {min(box_area_ratio * 100, 100):.0f}% of the image"
    reason += "."
    return level, int(round(percent)), reason
