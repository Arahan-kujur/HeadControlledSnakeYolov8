# 🐍 AI-Controlled Snake Game - Head Movement Control

A real-time head-controlled Snake game powered by YOLOv8 pose estimation. Control the snake by moving your head instead of using keyboard or mouse!
[![Watch the demo](demo.png)](https://youtu.be/J-1SKO85cjU)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange.svg)

## ✨ Features

- 🎮 **Head Movement Control**: Control snake direction by moving your head up, down, left, or right
- 🚀 **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- 📹 **Real-time Tracking**: Smooth, lag-free gameplay with live camera preview
- 🎯 **Pose Detection**: Uses YOLOv8 pose estimation for accurate head tracking
- 🎨 **Modern UI**: Clean Pygame interface with live camera feed overlay

## 🎬 Demo

The game tracks your head position in real-time and translates head movements into snake direction controls. Move your head to guide the snake and collect food!

## 📋 Requirements

- **Python**: 3.8 or higher
- **Webcam**: Any USB webcam or built-in camera
- **GPU** (Recommended): NVIDIA GPU with CUDA support for best performance
- **OS**: Windows, Linux, or macOS

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/snake-head-control.git
   cd snake-head-control
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The YOLOv8 pose model (`yolov8s-pose.pt`) will be automatically downloaded on first run.

## 🎮 Usage

1. **Run the game**
   ```bash
   python main.py
   ```

2. **Calibration**
   - The game will start with a 3-second calibration period
   - Keep your head still and face the camera during calibration

3. **Play**
   - Move your head in the direction you want the snake to go
   - Head up → Snake moves UP
   - Head down → Snake moves DOWN
   - Head left → Snake moves LEFT
   - Head right → Snake moves RIGHT

## 🎯 Controls

| Action | Control |
|--------|---------|
| Move Up | Move head upward |
| Move Down | Move head downward |
| Move Left | Move head to the left |
| Move Right | Move head to the right |
| Restart | Press SPACE (when game over) |
| Quit | Press ESC or close window |

## 📷 Camera Setup

For best results:
- Place camera at face-level
- Maintain 1-1.5 meters distance from camera
- Ensure good lighting (avoid backlighting)
- Keep your face fully visible in the camera frame
- Use a plain background if possible

## 🏗️ Project Structure

```
snake-head-control/
├── main.py                 # Main game controller
├── vision.py              # YOLOv8 pose detection and head tracking
├── gesture_recognition.py # Head movement analysis
├── control_bridge.py      # Converts head movements to game controls
├── game_engine.py         # Snake game logic
├── ui.py                  # Pygame UI and rendering
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Technical Details

- **Computer Vision**: YOLOv8 pose estimation model (COCO format)
- **Head Tracking**: Nose keypoint (index 0) from pose detection
- **Movement Detection**: Tracks head position changes over time
- **Smoothing**: Rolling average filter for stable controls
- **GPU Support**: Automatic CUDA detection and GPU acceleration

## 🐛 Troubleshooting

### Camera not detected
- Check if camera is connected and not being used by another application
- Try different camera indices (the game auto-detects)
- Verify camera permissions in system settings

### Low FPS / Laggy performance
- Ensure GPU drivers are up to date
- Check if CUDA is properly installed (for GPU acceleration)
- Reduce camera resolution in `vision.py` if needed
- Close other applications using the camera

### Head not detected
- Improve lighting conditions
- Move closer to the camera
- Ensure face is fully visible
- Check camera focus

### Model download issues
- Ensure stable internet connection for first-time model download
- Model file (~22MB) will be cached after first download

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**don anna**

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Pygame](https://www.pygame.org/) for game framework
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📊 Performance

- **FPS**: 45-60 FPS on RTX 4070 Super
- **Latency**: <50ms head tracking delay
- **Accuracy**: ~90%+ head detection accuracy

---

⭐ If you find this project interesting, please give it a star!
