---
description: Repository Information Overview
alwaysApply: true
---

# Gait Recognition System Information

## Summary
This project is a gait recognition system that uses computer vision techniques to detect, track, and identify people based on their walking patterns. It combines person detection models (YOLOv5, YOLOv8, SSD), tracking algorithms (ByteTrack/SORT), and silhouette extraction for gait recognition.

## Structure
- **model/**: Contains detection and extraction models
  - **person_det/**: Person detection models (YOLOv5, YOLOv8, SSD)
  - **person_ext/**: Person extraction models (RVM)
- **person_tracking/**: Person tracking implementation using ByteTrack/SORT
- **config.py**: Global configuration settings

## Language & Runtime
**Language**: Python
**Version**: Python >= 3.6 (as specified in requirements.txt)
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- PyTorch (torch >= 1.10.0, torchvision >= 0.10.0)
- OpenCV (opencv-python >= 4.1.1)
- Ultralytics (>= 8.0.0) for YOLOv8
- NumPy (>= 1.18.5)
- FastAPI (>= 0.68.0)
- Uvicorn (>= 0.15.0)
- Kornia (>= 0.6.0)
- FilterPy for tracking algorithms

**Development Dependencies**:
- IPython
- Tensorboard (>= 2.4.1)

## Build & Installation
```bash
pip install -r requirements.txt
```

## Components

### Person Detection
**Models**: 
- YOLOv5 (model/person_det/yolov5/)
- YOLOv8 (model/person_det/yolov8/)
- SSD MobileNet (model/person_det/SSD/)

**Usage**:
```python
from model.person_det.yolov8.detect_person import detect_person
# or
from model.person_det.SSD.model_data.detect_person import ssd_detect_person
```

### Person Tracking
**Framework**: ByteTrack/SORT implementation
**Main Files**: 
- person_tracking/tracking_pipeline.py
- person_tracking/bytetrack_tracker.py
- person_tracking/yolo_detector.py

**Usage**:
```python
from person_tracking.tracking_pipeline import process_video
process_video("path/to/video.mp4")
```

### Person Extraction
**Model**: Robust Video Matting (RVM)
**Location**: model/person_ext/rvm/
**Main File**: model/person_ext/rvm/person_ext.py

## Testing
**Framework**: Manual testing scripts
**Test Files**: person_tracking/test_tracking.py
**Run Command**:
```bash
python person_tracking/test_tracking.py
```

## Configuration
**Main Config**: config.py
**Key Settings**:
- CUDA_VISIBLE_DEVICES: GPU configuration
- UPLOAD_FOLDER: Path for uploaded videos
- PRE_METHOD: Preprocessing method (default: "rvm")
- RECOGNITION_SOURCE: Input source configuration (webcam, RTSP, upload)