from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# --------------------------------
# Settings
# --------------------------------

VIDEO = "bottle_test.mp4"

# Approximate region where the bottle appears
# Based on the first frame we inspected.
X1 = 120
Y1 = 400
X2 = 280
Y2 = 760

# --------------------------------
# Load metric depth model
# --------------------------------

print("Loading Metric Indoor Depth model...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    device=0
)

print("Depth model loaded!")
print("")

# --------------------------------
# Open video
# --------------------------------

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print("")

# --------------------------------
# Storage
# --------------------------------

depth_values = []
frame_numbers = []

# --------------------------------
# Process video
# --------------------------------

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------
    # Convert image
    # --------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_frame
    )

    # --------------------------------
    # Depth estimation
    # --------------------------------

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

    # Resize depth map
    depth_image = cv2.resize(
        depth_image,
        (
            frame.shape[1],
            frame.shape[0]
        )
    )

    # --------------------------------
    # Bottle region
    # --------------------------------

    bottle_region = depth_image[
        Y1:Y2,
        X1:X2
    ]

    # Remove invalid values
    bottle_region = bottle_region[
        np.isfinite(bottle_region)
    ]

    if len(bottle_region) > 0:

        depth_value = float(
            np.median(bottle_region)
        )

        depth_values.append(
            depth_value
        )

        frame_numbers.append(
            frame_number
        )

    # --------------------------------
    # Print every 10 frames
    # --------------------------------

    if frame_number % 10 == 0:

        if len(bottle_region) > 0:

            print(
                f"Frame {frame_number:03d} | "
                f"Depth: {depth_value:.3f} m"
            )

    # --------------------------------
    # Save sample frames
    # --------------------------------

    if frame_number in [1, 50, 100, 150, 200, 250, 300]:

        sample = frame.copy()

        # Draw selected region
        cv2.rectangle(
            sample,
            (X1, Y1),
            (X2, Y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            sample,
            f"Frame {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        filename = (
            f"calibration_frame_{frame_number}.jpg"
        )

        cv2.imwrite(
            filename,
            sample
        )

# --------------------------------
# Finish
# --------------------------------

cap.release()

print("")
print("--------------------------------")
print("DEPTH CALIBRATION TEST COMPLETE")
print("--------------------------------")

print(
    f"Frames processed: {frame_number}"
)

if len(depth_values) > 0:

    print(
        f"Minimum depth: "
        f"{min(depth_values):.3f} m"
    )

    print(
        f"Maximum depth: "
        f"{max(depth_values):.3f} m"
    )

    print(
        f"Average depth: "
        f"{np.mean(depth_values):.3f} m"
    )

else:

    print("No depth values obtained.")