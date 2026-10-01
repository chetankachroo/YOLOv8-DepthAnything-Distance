from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# --------------------------------
# Settings
# --------------------------------

VIDEO = "validation_test.mp4"

# Bottle region
X1 = 120
Y1 = 400
X2 = 280
Y2 = 760

# --------------------------------
# Known distances
# --------------------------------

# Time intervals in seconds
# and their actual measured distances

POSITIONS = [
    (0, 1, 0.90),    # 90 cm
    (5, 7, 0.75),    # 75 cm
    (11, 13, 0.60),  # 60 cm
    (17, 19, 0.30)   # 30 cm
]

# --------------------------------
# Load depth model
# --------------------------------

print("Loading Metric Indoor Depth model...")

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    device=0
)

print("Model loaded!")
print("")

# --------------------------------
# Open video
# --------------------------------

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open validation_test.mp4")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print("")

# --------------------------------
# Create storage
# --------------------------------

depth_data = {}

for start, end, actual_distance in POSITIONS:

    depth_data[actual_distance] = []

# --------------------------------
# Process video
# --------------------------------

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    time_seconds = frame_number / fps

    # --------------------------------
    # Find which distance interval
    # this frame belongs to
    # --------------------------------

    current_distance = None

    for start, end, actual_distance in POSITIONS:

        if start <= time_seconds <= end:

            current_distance = actual_distance
            break

    # Ignore frames outside our
    # four measurement intervals

    if current_distance is None:
        continue

    # --------------------------------
    # Convert frame to RGB
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

    bottle_region = bottle_region[
        np.isfinite(bottle_region)
    ]

    if len(bottle_region) == 0:
        continue

    # --------------------------------
    # Median depth
    # --------------------------------

    depth_value = float(
        np.median(bottle_region)
    )

    depth_data[current_distance].append(
        depth_value
    )

# --------------------------------
# Finish
# --------------------------------

cap.release()

print("")
print("--------------------------------")
print("VALIDATION RESULTS")
print("--------------------------------")
print("")

# --------------------------------
# Calculate median for each distance
# --------------------------------

results = []

for actual_distance in [0.90, 0.75, 0.60, 0.30]:

    values = depth_data[actual_distance]

    if len(values) == 0:

        print(
            f"{actual_distance:.2f} m: "
            f"No samples found"
        )

        continue

    median_depth = float(
        np.median(values)
    )

    mean_depth = float(
        np.mean(values)
    )

    minimum_depth = float(
        np.min(values)
    )

    maximum_depth = float(
        np.max(values)
    )

    results.append(
        (
            actual_distance,
            median_depth
        )
    )

    print(
        f"Actual distance: "
        f"{actual_distance:.2f} m"
    )

    print(
        f"Samples: "
        f"{len(values)}"
    )

    print(
        f"Median model depth: "
        f"{median_depth:.3f} m"
    )

    print(
        f"Mean model depth: "
        f"{mean_depth:.3f} m"
    )

    print(
        f"Range: "
        f"{minimum_depth:.3f} - "
        f"{maximum_depth:.3f} m"
    )

    print("")

# --------------------------------
# Summary
# --------------------------------

print("--------------------------------")
print("SUMMARY")
print("--------------------------------")

for actual_distance, median_depth in results:

    print(
        f"Actual: {actual_distance:.2f} m"
        f"  ->  "
        f"Model: {median_depth:.3f} m"
    )

print("")
print("VALIDATION ANALYSIS COMPLETE")