# YOLOv8 + Depth Anything V2: Distance Estimation

## Overview

This project combines YOLOv8 object detection with Depth Anything V2 Metric Indoor depth estimation to estimate the distance of a person from a monocular camera.

## Method

1. YOLOv8 detects the person in the video.
2. Depth Anything V2 estimates the depth from the monocular image.
3. The depth values within the detected person's bounding box are sampled.
4. A two-point calibration is applied using known reference distances.
5. The calibrated distance is displayed in metres.

## Models Used

- YOLOv8 for person detection
- Depth Anything V2 Metric Indoor for metric depth estimation

## Dataset

The Depth Anything V2 Metric Indoor model was fine-tuned using the synthetic Hypersim dataset.

## Calibration

Two known reference distances were used:

- 5.55 m
- 0.90 m

The resulting calibration equation was:

Distance = 1.124250 × raw_depth - 1.489149

## Validation

An independent reference point was used for validation:

- Actual distance: 2.85 m
- Predicted distance: 2.95 m
- Absolute error: 0.10 m
- Percentage error: 3.51%

## Main Code

`person_final_two_point.py` contains the final implementation of the detection, depth estimation, calibration and distance measurement pipeline.
