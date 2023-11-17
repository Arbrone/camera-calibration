# Camera Calibration Repository

This repository contains Python scripts for camera calibration. It provides two main scripts: `camera-calibration.py` for calibrating a single camera and `stereo-camera-calibration.py` for calibrating a stereo camera setup.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Prerequisites

Before using these calibration scripts, make sure you have the following prerequisites installed on your system:

- Python 3.x
- OpenCV (Open Source Computer Vision Library)

You can install OpenCV using pip:

```bash
pip install opencv-python
```

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/camera-calibration.git
```

Navigate to the repository's directory:

```bash
cd camera-calibration
```

## Usage
**Camera Calibration**

To calibrate a single camera, use the camera-calibration.py script. It requires two arguments:

* **-d** or **--device**: The device you want to calibrate (e.g., camera name or identifier).
* **-n** or **--name**: A custom name to identify the device.

**Example usage**:

```bash
python camera-calibration.py -d my_camera -n MyCamera --rows n --columns n
```

## Stereo Camera Calibration

To calibrate a stereo camera setup, use the stereo-camera-calibration.py script. It requires similar arguments as the camera-calibration.py script, but you'll need to specify parameters for both left and right cameras.

**Example usage**:

```bash
python stereo-camera-calibration.py --left lef_cam --right right_cam --rows n --columns n
```