# YOLOv8 + Depth Anything V2: Distance Estimation

## Overview

This project combines YOLOv8 object detection with Depth Anything V2 Metric Indoor depth estimation to estimate the distance of a person from a monocular camera.

The system takes video from a single camera, detects a person using YOLOv8, estimates scene depth using Depth Anything V2, and converts the depth estimate into an approximate metric distance using calibration.

## Method

The overall pipeline consists of the following steps:

1. YOLOv8 detects the person in the video.
2. Depth Anything V2 estimates depth from the monocular image.
3. Depth values from the detected person's bounding box are sampled.
4. A calibration relationship is applied to convert raw depth into an approximate physical distance.
5. The estimated distance is displayed in metres along with the detected person's bounding box.

## Models Used

- **YOLOv8** for person detection
- **Depth Anything V2 Metric Indoor** for metric depth estimation

## Dataset

The Depth Anything V2 Metric Indoor model used in this project was fine-tuned using the synthetic **Hypersim** dataset.

YOLOv8 is a pre-trained object detection model and was used for detecting the person in the input video.

## Bottle Calibration Experiment

An initial calibration experiment was performed using a bottle placed at known distances from the camera.

The bottle was recorded at approximately:

- 0.90 m
- 0.60 m
- 0.30 m

The corresponding raw depth estimates from Depth Anything V2 were analyzed to understand the relationship between the model's depth output and the actual physical distance.

The scripts related to this experiment are:

- `depth_calibration.py`
- `calibration_analysis.py`
- `validation_analysis.py`

This experiment demonstrated that the raw depth output could be related to physical distance through calibration. It also showed that a calibration obtained from one object/setup should not automatically be assumed to generalize to another object or scene.

## Person Distance Calibration

For the final person-distance system, a separate two-point calibration was performed using known distances from the person video.

The two reference distances were:

- 5.55 m
- 0.90 m

The median raw depth values at these reference points were used to derive the calibration relationship:

**Distance = 1.124250 × raw_depth - 1.489149**

This calibration was then applied to the detected person throughout the video.

## Validation

An additional reference point from the person video was used as an independent validation measurement.

The result was:

- Actual distance: **2.85 m**
- Predicted distance: **2.95 m**
- Absolute error: **0.10 m**
- Percentage error: **3.51%**

The 2.85 m measurement was not used to derive the two-point calibration.

## Code Structure

### Final system

`person_final_two_point.py`

Contains the final YOLOv8 + Depth Anything V2 pipeline, including:

- Person detection
- Depth estimation
- Depth extraction from the detected person
- Two-point calibration
- Metric distance estimation
- Result visualization

### Calibration experiments

`depth_calibration.py`

Performs depth extraction from the bottle video.

`calibration_analysis.py`

Analyzes the bottle measurements at known distances and calculates the calibration relationship.

`validation_analysis.py`

Analyzes the second bottle experiment to examine how the calibration performs at additional distances.

## Requirements

The main Python packages used in this project are listed in `requirements.txt`.

The project was developed using Python 3.12 with GPU acceleration through NVIDIA CUDA.

## Important Note

Depth Anything V2 provides model-based depth estimation. The final metric distance values depend on the camera setup, scene conditions, and calibration used in this experiment.

Therefore, the calibration used in this project is specific to the experimental setup and should not be assumed to provide the same accuracy for every camera, object, or environment.
