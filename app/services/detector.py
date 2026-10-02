from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


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
    # Enclosing box of all detections, normalised 0..1 (x1, y1, x2, y2).
    # Used by the live camera to check detections stay in the same place.
    enclosing_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FrameScan:
    """Everything the live camera needs to know about one frame.

    ``boxes`` are (x1, y1, x2, y2, confidence_percent) with the corners
    normalised to 0..1 so frames of any size can be compared.
    """

    boxes: tuple
    sharpness: float
    zoomed: bool = False


class GarbageDetector:
    """Lightweight ONNX garbage detector for local and hosted CPU deployment."""

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

        self._onnx_session = None
        self._onnx_input_name = None
        self._onnx_output_name = None
        self._input_size = 640

        self.active_model_path = None

        self._lock = threading.RLock()
        self.last_inference_ms = None

        self.allowed_class_names = (
            {name.lower() for name in allowed_class_names}
            if allowed_class_names
            else None
        )

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def self_check(self):
        """Load the ONNX model and run one small blank-frame test."""

        self._load_model()

        self._inspect_image(
            np.full(
                (320, 320, 3),
                255,
                dtype=np.uint8,
            ),
            0.25,
            320,
        )

        return self.active_model_path

    def _load_model(self):
        with self._lock:
            return self._load_model_locked()

    def _load_model_locked(self):
        """
        Load the ONNX model.

        If the configured model is:

            app/model/best_garbage.pt

        this automatically looks for:

            app/model/best_garbage.onnx
        """

        if self._onnx_session is not None:
            return self._onnx_session

        candidates = []

        # Primary model:
        # best_garbage.pt -> best_garbage.onnx
        candidates.append(
            self.model_path.with_suffix(".onnx")
        )

        # Explicit ONNX model.
        candidates.append(
            self.model_path.parent
            / "best_garbage.onnx"
        )

        # Remove duplicate paths.
        unique_candidates = []

        for candidate in candidates:
            candidate = Path(candidate)

            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        last_error = None

        for candidate in unique_candidates:

            if not candidate.is_file():
                continue

            try:
                log.info(
                    "Loading ONNX detection model: %s",
                    candidate,
                )

                self._onnx_session = ort.InferenceSession(
                    str(candidate),
                    providers=[
                        "CPUExecutionProvider"
                    ],
                )

                inputs = (
                    self._onnx_session.get_inputs()
                )

                outputs = (
                    self._onnx_session.get_outputs()
                )

                if not inputs:
                    raise RuntimeError(
                        "ONNX model has no input tensor."
                    )

                if not outputs:
                    raise RuntimeError(
                        "ONNX model has no output tensor."
                    )

                self._onnx_input_name = (
                    inputs[0].name
                )

                # Use whatever square size the ONNX file was exported at
                # (320, 640, ...) so the app never mismatches the model.
                shape = inputs[0].shape
                self._input_size = (
                    int(shape[2])
                    if len(shape) == 4
                    and isinstance(shape[2], int)
                    else 640
                )

                self._onnx_output_name = (
                    outputs[0].name
                )

                self.active_model_path = candidate

                log.info(
                    "ONNX model loaded successfully: %s",
                    candidate,
                )

                log.info(
                    "ONNX input: %s",
                    self._onnx_input_name,
                )

                log.info(
                    "ONNX output: %s",
                    self._onnx_output_name,
                )

                return self._onnx_session

            except Exception as exc:
                last_error = exc

                log.exception(
                    "Could not load ONNX model %s",
                    candidate,
                )

        expected = [
            str(path)
            for path in unique_candidates
        ]

        raise DetectionError(
            "ONNX detection model not found or "
            "could not be loaded. Expected one of: "
            + ", ".join(expected)
        ) from last_error

    # ============================================================
    # IMAGE DETECTION
    # ============================================================

    def detect(
        self,
        image_path: Path,
        result_path: Path,
        confidence: float,
        image_size: int,
        deep: bool = False,
    ):
        image = self._normalise_image(
            cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )
        )

        image = self._limit_size(image)

        result, detections = self._inspect_image(
            image,
            confidence,
            image_size,
            deep=deep,
        )

        # --------------------------------------------------------
        # Draw ONE enclosing garbage box.
        # --------------------------------------------------------

        if detections:

            x1 = min(
                item[0]
                for item in detections
            )

            y1 = min(
                item[1]
                for item in detections
            )

            x2 = max(
                item[2]
                for item in detections
            )

            y2 = max(
                item[3]
                for item in detections
            )

            confidence_text = (
                result.average_confidence
                if result.average_confidence is not None
                else 0.0
            )

            label = (
                f"Garbage "
                f"{confidence_text:.1f}%"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (22, 163, 74),
                3,
            )

            (
                width,
                height,
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            top = max(
                y1 - height - baseline - 8,
                0,
            )

            cv2.rectangle(
                image,
                (
                    x1,
                    top,
                ),
                (
                    x1 + width + 10,
                    top + height + baseline + 8,
                ),
                (22, 163, 74),
                -1,
            )

            cv2.putText(
                image,
                label,
                (
                    x1 + 5,
                    top + height + 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if not cv2.imwrite(
            str(result_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85,
            ],
        ):
            raise DetectionError(
                "The processed image could not be saved."
            )

        return result

    # ============================================================
    # CAMERA / BYTES DETECTION
    # ============================================================

    def inspect_bytes(
        self,
        image_bytes: bytes,
        confidence: float,
        image_size: int,
    ):
        """Inspect a camera frame without saving a report image."""

        image = self._normalise_image(
            cv2.imdecode(
                np.frombuffer(
                    image_bytes,
                    dtype=np.uint8,
                ),
                cv2.IMREAD_UNCHANGED,
            ),
            error_message=(
                "A camera frame is not a valid image."
            ),
        )

        image = self._limit_size(image)

        result, _ = self._inspect_image(
            image,
            confidence,
            image_size,
        )

        return result

    # ============================================================
    # IMAGE HELPERS
    # ============================================================

    @staticmethod
    def _limit_size(
        image,
        longest_side: int = 1600,
    ):
        """
        Shrink very large phone photos.

        The model itself runs at 320x320, so extremely
        large original images only waste CPU and memory.
        """

        height, width = image.shape[:2]

        largest = max(
            height,
            width,
        )

        if largest <= longest_side:
            return image

        scale = (
            longest_side
            / largest
        )

        return cv2.resize(
            image,
            (
                round(width * scale),
                round(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _normalise_image(
        image,
        error_message=(
            "The uploaded file is not a valid image."
        ),
    ):
        """
        Return normal BGR pixels.

        Transparent PNG images are composited onto white.
        """

        if image is None:
            raise DetectionError(
                error_message
            )

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        if image.shape[2] != 4:
            return image

        alpha = (
            image[:, :, 3:4]
            .astype(np.float32)
            / 255.0
        )

        foreground = (
            image[:, :, :3]
            .astype(np.float32)
        )

        white_background = np.full_like(
            foreground,
            255,
        )

        return (
            foreground * alpha
            + white_background * (1 - alpha)
        ).astype(np.uint8)

    # ============================================================
    # ONNX PREPROCESSING
    # ============================================================

    @staticmethod
    def _prepare_input(
        image,
        image_size: int,
    ):
        """
        Letterbox the BGR image to [1, 3, S, S] float32 RGB 0..1.

        The aspect ratio is kept and the borders are padded with grey
        (114), exactly like Ultralytics does, so the model sees objects
        the same way it did during training.
        """
        height, width = image.shape[:2]

        ratio = min(
            image_size / float(height),
            image_size / float(width),
        )

        new_w = max(1, int(round(width * ratio)))
        new_h = max(1, int(round(height * ratio)))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_w = (image_size - new_w) / 2.0
        pad_h = (image_size - new_h) / 2.0

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(
            padded,
            cv2.COLOR_BGR2RGB,
        )

        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return (
            np.ascontiguousarray(
                tensor,
                dtype=np.float32,
            ),
            ratio,
            (left, top),
        )

    # ============================================================
    # ONNX OUTPUT DECODING
    # ============================================================

    @staticmethod
    def _sigmoid(value):
        value = np.clip(
            value,
            -50,
            50,
        )

        return 1.0 / (
            1.0 + np.exp(-value)
        )

    def _decode_predictions(
        self,
        output,
        original_width,
        original_height,
        image_size,
        confidence,
        ratio=None,
        pad=(0, 0),
    ):
        """
        Decode exported YOLO output.

        Current exported model:

            [1, 5, 2100]

        Five values:

            x
            y
            width
            height
            confidence

        This is treated as a single-class
        garbage detector.
        """

        output = np.asarray(
            output,
            dtype=np.float32,
        )

        if output.ndim == 3:
            output = output[0]

        if output.ndim != 2:
            raise DetectionError(
                "Unexpected ONNX output shape: "
                f"{output.shape}"
            )

        # [5, 2100] -> [2100, 5]
        if output.shape[0] == 5:
            predictions = output.T

        elif output.shape[1] == 5:
            predictions = output

        else:
            raise DetectionError(
                "Unexpected ONNX output shape: "
                f"{output.shape}"
            )

        if ratio is None:
            ratio = min(
                image_size / float(original_height),
                image_size / float(original_width),
            )

        pad_x, pad_y = pad

        boxes = []
        scores = []

        for row in predictions:

            if len(row) < 5:
                continue

            cx = float(row[0])
            cy = float(row[1])
            width = float(row[2])
            height = float(row[3])
            score = float(row[4])

            # Some exports return logits.
            if (
                score < 0.0
                or score > 1.0
            ):
                score = float(
                    self._sigmoid(score)
                )

            if score < confidence:
                continue

            x1 = (
                cx - width / 2.0 - pad_x
            ) / ratio

            y1 = (
                cy - height / 2.0 - pad_y
            ) / ratio

            x2 = (
                cx + width / 2.0 - pad_x
            ) / ratio

            y2 = (
                cy + height / 2.0 - pad_y
            ) / ratio

            x1 = max(
                0,
                min(
                    original_width - 1,
                    int(round(x1)),
                ),
            )

            y1 = max(
                0,
                min(
                    original_height - 1,
                    int(round(y1)),
                ),
            )

            x2 = max(
                0,
                min(
                    original_width - 1,
                    int(round(x2)),
                ),
            )

            y2 = max(
                0,
                min(
                    original_height - 1,
                    int(round(y2)),
                ),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                ]
            )

            scores.append(score)

        if not boxes:
            return []

        # --------------------------------------------------------
        # Non-Maximum Suppression
        # --------------------------------------------------------

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            float(confidence),
            0.45,
        )

        if indices is None:
            return []

        indices = np.asarray(
            indices
        ).reshape(-1)

        detections = []

        for index in indices:

            index = int(index)

            x, y, width, height = (
                boxes[index]
            )

            x1 = int(x)
            y1 = int(y)

            x2 = int(
                x + width
            )

            y2 = int(
                y + height
            )

            score_percent = (
                scores[index]
                * 100.0
            )

            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    score_percent,
                )
            )

        return detections

    # ============================================================
    # ZOOM PASSES (small / far-away garbage)
    # ============================================================
    #
    # The network sees a 640 px square. A phone frame that is 1024 px wide
    # is shrunk to ~60 %, so garbage that is far away becomes a few pixels
    # and is missed. Two overlapping crops that each cover 60 % of the long
    # side are enlarged to ~100 %, which makes distant items about 1.6x
    # bigger for the model without any new training.

    @staticmethod
    def _zoom_tiles(image, fraction: float = 0.6):
        height, width = image.shape[:2]
        if width >= height:
            tile = int(round(width * fraction))
            return [(0, 0, tile, height), (width - tile, 0, width, height)]
        tile = int(round(height * fraction))
        return [(0, 0, width, tile), (0, height - tile, width, height)]

    def _zoom_detections(self, image, confidence: float):
        found = []
        size = self._input_size
        for x1, y1, x2, y2 in self._zoom_tiles(image):
            tile = image[y1:y2, x1:x2]
            tensor, ratio, pad = self._prepare_input(tile, size)
            output = self._onnx_session.run(
                [self._onnx_output_name],
                {self._onnx_input_name: tensor},
            )[0]
            for a, b, c, d, score in self._decode_predictions(
                output,
                tile.shape[1],
                tile.shape[0],
                size,
                confidence,
                ratio,
                pad,
            ):
                found.append((a + x1, b + y1, c + x1, d + y1, score))
        return found

    @staticmethod
    def _merge_detections(detections, iou_threshold: float = 0.45):
        """Non-maximum suppression across the full-frame and zoomed passes."""
        if len(detections) < 2:
            return list(detections)
        boxes = [
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            for x1, y1, x2, y2, _ in detections
        ]
        scores = [float(item[4]) / 100.0 for item in detections]
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, float(iou_threshold))
        if keep is None or len(keep) == 0:
            return list(detections)
        return [detections[int(i)] for i in np.asarray(keep).reshape(-1)]

    # ============================================================
    # CAMERA FRAME SCAN
    # ============================================================

    @staticmethod
    def sharpness(image) -> float:
        """Variance of the Laplacian: low = blurry, high = crisp."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scale = 480.0 / max(height, width)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def scan_frame(
        self,
        image_bytes: bytes,
        confidence: float,
        image_size: int,
        solid_percent: float = 60.0,
        solid_area: float = 0.04,
    ) -> FrameScan:
        """Detect on one camera frame.

        The normal full-frame pass always runs. The two zoom passes are added
        only when that pass found nothing solid (a large, confident box),
        because that is exactly when small / far / blurry garbage hides.
        """
        image = self._limit_size(
            self._normalise_image(
                cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                ),
                error_message="A camera frame is not a valid image.",
            )
        )
        height, width = image.shape[:2]
        total = float(width * height)

        _, detections = self._inspect_image(image, confidence, image_size)
        zoomed = False
        solid = any(
            item[4] >= solid_percent
            and (item[2] - item[0]) * (item[3] - item[1]) / total >= solid_area
            for item in detections
        )
        if not solid:
            detections = self._merge_detections(
                list(detections)
                + self._zoom_detections_locked(image, confidence)
            )
            zoomed = True

        boxes = tuple(
            (
                item[0] / width,
                item[1] / height,
                item[2] / width,
                item[3] / height,
                float(item[4]),
            )
            for item in detections
        )
        return FrameScan(
            boxes=boxes,
            sharpness=self.sharpness(image),
            zoomed=zoomed,
        )

    def _zoom_detections_locked(self, image, confidence: float):
        with self._lock:
            self._load_model()
            return self._zoom_detections(image, confidence)

    # ============================================================
    # MAIN INFERENCE
    # ============================================================

    def _inspect_image(
        self,
        image,
        confidence: float,
        image_size: int,
        deep: bool = False,
    ):
        try:

            with self._lock:

                started = time.perf_counter()

                session = self._load_model()

                size = self._input_size

                input_tensor, ratio, pad = (
                    self._prepare_input(
                        image,
                        size,
                    )
                )

                outputs = session.run(
                    [
                        self._onnx_output_name
                    ],
                    {
                        self._onnx_input_name:
                            input_tensor
                    },
                )

                prediction = outputs[0]

                detections = (
                    self._decode_predictions(
                        prediction,
                        image.shape[1],
                        image.shape[0],
                        size,
                        confidence,
                        ratio,
                        pad,
                    )
                )

                if deep:
                    detections = self._merge_detections(
                        list(detections)
                        + self._zoom_detections(image, confidence)
                    )

                self.last_inference_ms = round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000
                )

        except DetectionError:
            raise

        except Exception as exc:

            log.exception(
                "ONNX prediction failed"
            )

            raise DetectionError(
                "Detection could not be completed. "
                "Please try another image."
            ) from exc

        # --------------------------------------------------------
        # Calculate detection statistics.
        # --------------------------------------------------------

        box_area_ratio = None
        largest_box_ratio = None
        enclosing_box = None

        if detections:

            x1 = min(
                item[0]
                for item in detections
            )

            y1 = min(
                item[1]
                for item in detections
            )

            x2 = max(
                item[2]
                for item in detections
            )

            y2 = max(
                item[3]
                for item in detections
            )

            average_confidence = (
                sum(
                    item[4]
                    for item in detections
                )
                / len(detections)
            )

            image_height, image_width = (
                image.shape[:2]
            )

            total_area = (
                image_width
                * image_height
            )

            box_area_ratio = (
                max(
                    x2 - x1,
                    0,
                )
                * max(
                    y2 - y1,
                    0,
                )
            ) / total_area

            largest_box_ratio = max(
                (
                    max(
                        item[2] - item[0],
                        0,
                    )
                    * max(
                        item[3] - item[1],
                        0,
                    )
                )
                / total_area
                for item in detections
            )

            enclosing_box = (
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            )

        else:
            average_confidence = None

        result = DetectionResult(
            detection_count=len(
                detections
            ),
            average_confidence=(
                average_confidence
                if detections
                else None
            ),
            highest_confidence=max(
                (
                    item[4]
                    for item in detections
                ),
                default=None,
            ),
            box_area_ratio=(
                box_area_ratio
            ),
            largest_box_ratio=(
                largest_box_ratio
            ),
            enclosing_box=enclosing_box,
        )

        return result, detections