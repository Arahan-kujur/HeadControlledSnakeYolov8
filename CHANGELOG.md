# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-XX

### Added
- Initial release of AI-Controlled Snake Game
- Head movement control using YOLOv8 pose estimation
- Real-time camera preview with head tracking visualization
- GPU acceleration support for improved performance
- Automatic camera detection
- Calibration system for neutral head position
- Smooth movement detection with rolling average filtering
- Game controls: head movement for direction, gestures for boost/pause
- Modern Pygame UI with live camera feed overlay

### Technical Details
- YOLOv8 pose model integration
- COCO keypoint format for head detection (nose keypoint)
- OpenCV for camera handling and frame processing
- Pygame for game rendering and UI
- PyTorch for GPU acceleration

[1.0.0]: https://github.com/yourusername/snake-head-control/releases/tag/v1.0.0

