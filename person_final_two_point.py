from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# ============================================================
# SETTINGS
# ============================================================

INPUT_VIDEO = "test_video.mp4"
OUTPUT_VIDEO = "person_final_two_point.mp4"

CONFIDENCE_THRESHOLD = 0.30

# Two-point calibration obtained from the same video
CALIBRATION_A = 1.124250
CALIBRATION_B = -1.489149


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")

yolo = YOLO("yolov8n.pt")

print("YOLO loaded!")

print("Loading Metric Indoor Depth model...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    device=0
)

print("Depth model loaded!")
print("")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print("")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    print(
        f"Processing frame {frame_number}..."
    )


    # ========================================================
    # YOLO PERSON DETECTION
    # ========================================================

    result = yolo(
        frame,
        verbose=False,
        device=0
    )[0]


    # ========================================================
    # DEPTH ESTIMATION
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_frame
    )


    depth_result = depth_pipe(
        pil_image
    )


    depth_image = (
        depth_result["predicted_depth"]
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )


    depth_image = cv2.resize(
        depth_image,
        (width, height)
    )


    # ========================================================
    # PROCESS DETECTED PEOPLE
    # ========================================================

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        object_name = yolo.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )


        # Only process people

        if object_name != "person":
            continue


        if confidence < CONFIDENCE_THRESHOLD:
            continue


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )


        # ----------------------------------------------------
        # Central region
        # ----------------------------------------------------

        box_width = x2 - x1
        box_height = y2 - y1


        inner_x1 = int(
            x1 + box_width * 0.25
        )

        inner_x2 = int(
            x2 - box_width * 0.25
        )

        inner_y1 = int(
            y1 + box_height * 0.25
        )

        inner_y2 = int(
            y2 - box_height * 0.25
        )


        # ----------------------------------------------------
        # Extract depth
        # ----------------------------------------------------

        person_depth = depth_image[
            inner_y1:inner_y2,
            inner_x1:inner_x2
        ]


        person_depth = person_depth[
            np.isfinite(person_depth)
        ]


        if len(person_depth) == 0:
            continue


        # ----------------------------------------------------
        # Raw depth
        # ----------------------------------------------------

        raw_depth = float(
            np.median(person_depth)
        )


        # ====================================================
        # APPLY TWO-POINT CALIBRATION
        # ====================================================

        estimated_distance = (
            CALIBRATION_A * raw_depth
            + CALIBRATION_B
        )


        # Prevent negative distance

        estimated_distance = max(
            0.0,
            estimated_distance
        )


        # ----------------------------------------------------
        # Center of person
        # ----------------------------------------------------

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


        # ====================================================
        # DRAW CENTER
        # ====================================================

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )


        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"Person | "
            f"{estimated_distance:.2f} m | "
            f"conf: {confidence:.2f}"
        )


        # ----------------------------------------------------
        # Text size
        # ----------------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )


        label_y = max(
            y1 - 8,
            text_height + 10
        )


        # ====================================================
        # LABEL BACKGROUND
        # ====================================================

        cv2.rectangle(
            frame,
            (
                x1,
                label_y - text_height - 8
            ),
            (
                x1 + text_width + 6,
                label_y + baseline - 4
            ),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # LABEL TEXT
        # ====================================================

        cv2.putText(
            frame,
            label,
            (
                x1 + 3,
                label_y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    out.write(frame)


# ============================================================
# FINISH
# ============================================================

cap.release()

out.release()


print("")
print("==========================================")
print("TWO-POINT PERSON DISTANCE SYSTEM COMPLETE")
print("==========================================")

print(
    f"Saved: {OUTPUT_VIDEO}"
)

print(
    "Calibration:"
)

print(
    "Distance = "
    "1.124250 × raw_depth - 1.489149"
)