import numpy as np

# --------------------------------
# IMPORTANT:
# These are the frame ranges based
# on your video timing.
# --------------------------------

# 90 cm: 0-1 seconds
# 60 cm: 3-5 seconds
# 30 cm: 9-11 seconds

# We will use the depth values saved
# by running depth_calibration.py again,
# but only for the relevant frame ranges.

from transformers import pipeline
from PIL import Image
import cv2

VIDEO = "bottle_test.mp4"

# Bottle region from our previous test
X1 = 120
Y1 = 400
X2 = 280
Y2 = 760

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

fps = cap.get(cv2.CAP_PROP_FPS)

print(f"FPS: {fps:.2f}")
print("")

# --------------------------------
# Storage
# --------------------------------

depth_90cm = []
depth_60cm = []
depth_30cm = []

frame_number = 0

# --------------------------------
# Process video
# --------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------
    # Determine which section
    # --------------------------------

    time_seconds = frame_number / fps

    if 0 <= time_seconds <= 1:
        current_group = depth_90cm

    elif 3 <= time_seconds <= 5:
        current_group = depth_60cm

    elif 9 <= time_seconds <= 11:
        current_group = depth_30cm

    else:
        continue

    # --------------------------------
    # Convert frame
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

    if len(bottle_region) > 0:

        depth_value = float(
            np.median(bottle_region)
        )

        current_group.append(
            depth_value
        )

# --------------------------------
# Finish
# --------------------------------

cap.release()

print("")
print("--------------------------------")
print("CALIBRATION ANALYSIS")
print("--------------------------------")
print("")

# --------------------------------
# 90 cm
# --------------------------------

if len(depth_90cm) > 0:

    median_90 = np.median(depth_90cm)

    print(
        f"Actual distance: 0.90 m"
    )

    print(
        f"Number of samples: "
        f"{len(depth_90cm)}"
    )

    print(
        f"Median model depth: "
        f"{median_90:.3f} m"
    )

    print("")

# --------------------------------
# 60 cm
# --------------------------------

if len(depth_60cm) > 0:

    median_60 = np.median(depth_60cm)

    print(
        f"Actual distance: 0.60 m"
    )

    print(
        f"Number of samples: "
        f"{len(depth_60cm)}"
    )

    print(
        f"Median model depth: "
        f"{median_60:.3f} m"
    )

    print("")

# --------------------------------
# 30 cm
# --------------------------------

if len(depth_30cm) > 0:

    median_30 = np.median(depth_30cm)

    print(
        f"Actual distance: 0.30 m"
    )

    print(
        f"Number of samples: "
        f"{len(depth_30cm)}"
    )

    print(
        f"Median model depth: "
        f"{median_30:.3f} m"
    )

    print("")

# --------------------------------
# Final summary
# --------------------------------

print("--------------------------------")
print("SUMMARY")
print("--------------------------------")

if len(depth_90cm) > 0:
    print(
        f"0.90 m actual  -> "
        f"{np.median(depth_90cm):.3f} m model"
    )

if len(depth_60cm) > 0:
    print(
        f"0.60 m actual  -> "
        f"{np.median(depth_60cm):.3f} m model"
    )

if len(depth_30cm) > 0:
    print(
        f"0.30 m actual  -> "
        f"{np.median(depth_30cm):.3f} m model"
    )

print("")
print("ANALYSIS COMPLETE")