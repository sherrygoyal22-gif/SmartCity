from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The bundled weights are trusted local files. Newer PyTorch releases default
# to "weights_only" loading, which can reject YOLO checkpoints and made the
# model fail to load on some machines. Must be set before torch is imported.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# ultralytics (and therefore PyTorch) is imported lazily, the first time the model is
# needed. Importing it at start-up costs a few hundred MB of RAM, which crashes small
# hosted instances (512 MB) before anyone has even logged in.
YOLO = None
_IMPORT_ERROR = None
_import_lock = threading.Lock()


def _import_yolo():
    """Import ultralytics on first use; returns YOLO or None (error kept in _IMPORT_ERROR)."""
    global YOLO, _IMPORT_ERROR
    if YOLO is not None:
        return YOLO
    with _import_lock:
        if YOLO is None:
            try:
                from ultralytics import YOLO as _YOLO
                YOLO = _YOLO
                _IMPORT_ERROR = None
            except Exception as exc:  # missing / broken install: report it clearly later
                _IMPORT_ERROR = exc
    return YOLO

log = logging.getLogger("smartcity.detector")


class DetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectionResult:
    detection_count: int
    average_confidence: float | None
    highest_confidence: float | None
    box_area_ratio: float | None
    largest_box_ratio: float | None = None


def _limit_threads() -> None:
    """Keep PyTorch/OpenCV to a few threads (see wsgi.py): much faster on small hosted CPUs."""
    try:
        count = max(1, int(os.environ.get("SMARTCITY_TORCH_THREADS", "2")))
    except ValueError:
        count = 2
    try:
        import torch

        torch.set_num_threads(count)
    except Exception:
        pass
    try:
        cv2.setNumThreads(count)
    except Exception:
        pass


class GarbageDetector:
    """Lazy YOLO loader that renders one enclosing garbage box per image."""

    def __init__(
        self,
        model_path: Path,
        fallback_model_path: Path | None = None,
        allowed_class_names: tuple[str, ...] | None = None,
    ):
        self.model_path = Path(model_path)
        self.fallback_model_path = (
            Path(fallback_model_path)
            if fallback_model_path is not None
            else None
        )
        self._model = None
        self.active_model_path = None
        # One request at a time inside the model: the start-up warm-up and a user's
        # first request must not both load it (doubles memory), and concurrent
        # predict() calls on one model are not safe.
        self._lock = threading.RLock()
        self.last_inference_ms = None
        # False-positive guard: only boxes whose predicted class name matches
        # one of these (case-insensitive) are treated as real garbage.
        # None/empty means "accept every class" (unchanged legacy behaviour),
        # so this is safe to leave unset if a model's class names are unknown.
        self.allowed_class_names = (
            {name.lower() for name in allowed_class_names}
            if allowed_class_names
            else None
        )

    def self_check(self):
        """Load the model and run it once on a blank frame. Returns the model path used."""
        self._load_model()
        self._inspect_image(np.full((320, 320, 3), 255, dtype=np.uint8), 0.25, 320)
        return self.active_model_path

    def _load_model(self):
        with self._lock:
            return self._load_model_locked()

    def _load_model_locked(self):
        if self._model is None:
            candidates = [self.model_path]
            if self.fallback_model_path and self.fallback_model_path != self.model_path:
                candidates.append(self.fallback_model_path)

            if _import_yolo() is None:
                log.error("ultralytics could not be imported: %r", _IMPORT_ERROR)
                raise DetectionError(
                    "The detection engine (ultralytics) is not installed correctly. "
                    "Close the server and run:  python run.py  to repair it."
                ) from _IMPORT_ERROR

            errors = []
            for candidate in candidates:
                if not candidate.is_file():
                    errors.append(FileNotFoundError(candidate))
                    continue
                try:
                    self._model = YOLO(str(candidate))
                    self.active_model_path = candidate
                    break
                except Exception as exc:
                    log.exception("Could not load model %s", candidate)
                    errors.append(exc)

            if self._model is not None:
                _limit_threads()

            if self._model is None:
                missing = [str(e) for e in errors if isinstance(e, FileNotFoundError)]
                if len(missing) == len(errors) and errors:
                    message = "Detection model file not found: " + ", ".join(missing)
                else:
                    message = ("The detection model could not be loaded. Run  python run.py  "
                               "once to update the detection engine, then try again.")
                raise DetectionError(message) from (errors[-1] if errors else None)
        return self._model

    def detect(self, image_path: Path, result_path: Path, confidence: float, image_size: int):
        image = self._normalise_image(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED))
        image = self._limit_size(image)
        result, detections = self._inspect_image(image, confidence, image_size)

        if detections:
            x1 = min(item[0] for item in detections)
            y1 = min(item[1] for item in detections)
            x2 = max(item[2] for item in detections)
            y2 = max(item[3] for item in detections)
            label = f"Garbage {result.average_confidence:.1f}%"
            cv2.rectangle(image, (x1, y1), (x2, y2), (22, 163, 74), 3)
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1 - height - baseline - 8, 0)
            cv2.rectangle(image, (x1, top), (x1 + width + 10, top + height + baseline + 8), (22, 163, 74), -1)
            cv2.putText(image, label, (x1 + 5, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if not cv2.imwrite(str(result_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise DetectionError("The processed image could not be saved.")
        return result

    def inspect_bytes(self, image_bytes: bytes, confidence: float, image_size: int):
        """Inspect a camera frame without saving a report or result image."""
        image = self._normalise_image(
            cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED),
            error_message="A camera frame is not a valid image.",
        )
        result, _ = self._inspect_image(image, confidence, image_size)
        return result

    @staticmethod
    def _limit_size(image, longest_side: int = 1600):
        """Shrink huge phone photos: the model works at 640 px, so more pixels only slow everything down."""
        height, width = image.shape[:2]
        largest = max(height, width)
        if largest <= longest_side:
            return image
        scale = longest_side / largest
        return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _normalise_image(image, error_message="The uploaded file is not a valid image."):
        """Return BGR pixels, compositing transparent uploads onto white.

        OpenCV otherwise discards PNG alpha onto black. That changes the visual
        input for transparent artwork, even though browsers display it over a
        light page background, and can suppress valid V4 detections.
        """
        if image is None:
            raise DetectionError(error_message)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] != 4:
            return image

        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        foreground = image[:, :, :3].astype(np.float32)
        white_background = np.full_like(foreground, 255)
        return (foreground * alpha + white_background * (1 - alpha)).astype(np.uint8)

    def _inspect_image(self, image, confidence: float, image_size: int):
        try:
            with self._lock:
                started = time.perf_counter()
                prediction = self._load_model().predict(
    source=image,
    conf=confidence,
    imgsz=320,
    verbose=False,
    device="cpu",
    half=False,
)[0]
                self.last_inference_ms = round((time.perf_counter() - started) * 1000)
        except DetectionError:
            raise
        except Exception as exc:
            log.exception("YOLO prediction failed")
            raise DetectionError("Detection could not be completed. Please try another image.") from exc

        class_names = getattr(self._model, "names", {}) or {}
        detections = []
        for box in prediction.boxes or []:
            if self.allowed_class_names is not None:
                try:
                    class_id = int(box.cls[0])
                    class_name = str(class_names.get(class_id, "")).lower()
                except (TypeError, ValueError, IndexError):
                    class_name = ""
                # Reject anything that isn't a recognised garbage class
                # (background objects, roads, people, vehicles, etc.) even
                # though YOLO gave it a passing confidence score.
                if class_name not in self.allowed_class_names:
                    continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            score = float(box.conf[0]) * 100
            detections.append((x1, y1, x2, y2, score))

        box_area_ratio = None
        largest_box_ratio = None
        if detections:
            # The model can identify several pieces of litter in one scene.
            # Render one final municipal-report box enclosing all predictions.
            x1 = min(item[0] for item in detections)
            y1 = min(item[1] for item in detections)
            x2 = max(item[2] for item in detections)
            y2 = max(item[3] for item in detections)
            average_confidence = sum(item[4] for item in detections) / len(detections)
            image_height, image_width = image.shape[:2]
            box_area_ratio = (
                max(x2 - x1, 0) * max(y2 - y1, 0)
            ) / (image_width * image_height)
            # Largest SINGLE box. The merged box above can be inflated by two
            # tiny false hits in opposite corners; live-camera validation uses
            # this stricter value instead.
            largest_box_ratio = max(
                (max(d[2] - d[0], 0) * max(d[3] - d[1], 0)) / (image_width * image_height)
                for d in detections
            )
        return DetectionResult(
            detection_count=len(detections),
            average_confidence=average_confidence if detections else None,
            highest_confidence=max((item[4] for item in detections), default=None),
            box_area_ratio=box_area_ratio,
            largest_box_ratio=largest_box_ratio,
        ), detections
