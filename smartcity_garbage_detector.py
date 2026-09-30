
from ultralytics import YOLO
import cv2


class SmartCityGarbageDetector:

    def __init__(
        self,
        model_path,
        conf=0.10,
        imgsz=640,
        device=0
    ):

        print("🧠 Loading SmartCity model...")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

        print("✅ Model loaded!")


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, image_path):

        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        result = results[0]

        # No detections
        if result.boxes is None or len(result.boxes) == 0:

            return {
                "garbage_detected": False,
                "confidence": 0.0,
                "box": None,
                "internal_detections": 0
            }

        # Get all YOLO boxes
        boxes = result.boxes.xyxy.cpu().numpy()

        # Get confidences
        confidences = result.boxes.conf.cpu().numpy()

        # ====================================================
        # MERGE ALL DETECTIONS INTO ONE BOX
        # ====================================================

        x1 = int(min(box[0] for box in boxes))
        y1 = int(min(box[1] for box in boxes))

        x2 = int(max(box[2] for box in boxes))
        y2 = int(max(box[3] for box in boxes))

        # Highest confidence
        confidence = float(max(confidences))

        return {
            "garbage_detected": True,
            "confidence": confidence,

            # ONLY ONE FINAL BOX
            "box": [x1, y1, x2, y2],

            "internal_detections": len(boxes)
        }


    # ========================================================
    # PREDICT + DRAW
    # ========================================================

    def predict_and_draw(
        self,
        image_path,
        output_path
    ):

        # Run prediction
        result = self.predict(image_path)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"❌ Cannot read image: {image_path}"
            )

        # ====================================================
        # DRAW ONLY ONE BOX
        # ====================================================

        if result["garbage_detected"]:

            x1, y1, x2, y2 = result["box"]

            confidence = result["confidence"]

            label = (
                f"garbage {confidence:.2f}"
            )

            # ONE bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Label
            cv2.putText(
                image,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Save output
        cv2.imwrite(output_path, image)

        return result, output_path


print("✅ Permanent SmartCity detector module created successfully!")
print("📁 Location: /content/smartcity_garbage_detector.py")
print("🎯 Feature: Multiple YOLO boxes → ONE final garbage box")
