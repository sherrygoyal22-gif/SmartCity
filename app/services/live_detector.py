import cv2
from pathlib import Path
from ultralytics import YOLO
import time


# ============================================================
# SMARTCITY AI GARBAGE DETECTION - LIVE DETECTOR
# ============================================================

# Project structure:
#
# Version 2/
# └── app/
#     ├── model/
#     │   └── best_garbage.pt
#     └── services/
#         └── live_detector.py
#
# This file is inside:
# app/services/live_detector.py


# ------------------------------------------------------------
# 1. FIND PROJECT ROOT
# ------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()

# live_detector.py
#      ↓
# services
#      ↓
# app
#      ↓
# Version 2

PROJECT_ROOT = CURRENT_FILE.parents[2]


# ------------------------------------------------------------
# 2. MODEL PATH
# ------------------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "app" / "model" / "V4_best.pt"


# Check whether model exists
if not MODEL_PATH.exists():
    print("\nERROR: Model file not found!")
    print("Expected model location:")
    print(MODEL_PATH)
    print("\nMake sure best_garbage.pt is inside:")
    print("Version 2/app/model/")
    exit()


# ------------------------------------------------------------
# 3. LOAD YOLO MODEL
# ------------------------------------------------------------

print("\nLoading AI Garbage Detection Model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")
print("Model:", MODEL_PATH)


# ------------------------------------------------------------
# 4. SETTINGS
# ------------------------------------------------------------

# IMPORTANT:
# Only detections >= 0.75 confidence will be accepted.
CONFIDENCE_THRESHOLD = 0.75

# Number of consecutive frames required
# before we consider garbage confirmed.
REQUIRED_FRAMES = 5

# Camera number
CAMERA_ID = 0


# ------------------------------------------------------------
# 5. OPEN CAMERA
# ------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("\nERROR: Camera could not be opened.")
    print("Check that your webcam is connected.")
    exit()


# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


print("\n======================================")
print("SMARTCITY AI GARBAGE DETECTION")
print("======================================")
print("Live camera started.")
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Press Q to stop.")
print("======================================\n")


# ------------------------------------------------------------
# 6. VARIABLES
# ------------------------------------------------------------

garbage_frame_count = 0
alert_active = False

last_detection_time = 0

# Prevent console from printing the same alert continuously
ALERT_COOLDOWN = 10


# ------------------------------------------------------------
# 7. START LIVE DETECTION
# ------------------------------------------------------------

while True:

    # Read camera frame
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break


    # --------------------------------------------------------
    # YOLO PREDICTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )


    # --------------------------------------------------------
    # DETECTION STATUS
    # --------------------------------------------------------

    garbage_detected = False
    highest_confidence = 0.0


    # --------------------------------------------------------
    # PROCESS YOLO RESULTS
    # --------------------------------------------------------

    for result in results:

        # Bounding boxes
        boxes = result.boxes

        if boxes is None:
            continue


        for box in boxes:

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]


            # ------------------------------------------------
            # ONLY ACCEPT GARBAGE
            # ------------------------------------------------

            if class_name.lower() == "garbage" and confidence >= CONFIDENCE_THRESHOLD:

                garbage_detected = True

                highest_confidence = max(
                    highest_confidence,
                    confidence
                )


                # Bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # Detection label
                label = f"Garbage {confidence:.2f}"


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


    # --------------------------------------------------------
    # FRAME CONFIRMATION SYSTEM
    # --------------------------------------------------------

    if garbage_detected:

        garbage_frame_count += 1

    else:

        # Reset if garbage disappears
        garbage_frame_count = 0
        alert_active = False


    # --------------------------------------------------------
    # GARBAGE CONFIRMED
    # --------------------------------------------------------

    if garbage_frame_count >= REQUIRED_FRAMES:

        alert_active = True


        # Show alert on screen
        cv2.rectangle(
            frame,
            (10, 10),
            (430, 65),
            (0, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            "GARBAGE DETECTED!",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # ALERT MESSAGE IN TERMINAL
        # ----------------------------------------------------

        current_time = time.time()

        if current_time - last_detection_time >= ALERT_COOLDOWN:

            print("\n--------------------------------------")
            print("ALERT: GARBAGE DETECTED")
            print(
                f"Confidence: {highest_confidence:.2f}"
            )
            print("--------------------------------------")

            last_detection_time = current_time


    else:

        # Normal status
        cv2.putText(
            frame,
            "Monitoring...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # SHOW CONFIDENCE / STATUS
    # --------------------------------------------------------

    if garbage_detected:

        status_text = (
            f"Garbage confidence: "
            f"{highest_confidence:.2f}"
        )

        cv2.putText(
            frame,
            status_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


    # --------------------------------------------------------
    # SHOW CAMERA WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "SmartCity AI Garbage Detection",
        frame
    )


    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\nStopping live detection...")
        break


# ------------------------------------------------------------
# 8. RELEASE CAMERA
# ------------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Live detection stopped.")