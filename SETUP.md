# Setup Guide

## Quick Start

1. **Install Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/snake-head-control.git
   cd snake-head-control
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Game**
   ```bash
   python main.py
   ```

## First Run

On the first run, YOLOv8 will automatically download the pose model (~22MB). This is a one-time download and the model will be cached for future use.

## GPU Setup (Optional but Recommended)

For best performance with GPU acceleration:

1. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Install the appropriate version for your GPU

2. **Install PyTorch with CUDA**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify GPU Detection**
   - Run the game and check console output
   - Should see: "Using GPU: [Your GPU Name]"

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Camera Issues
- Check camera permissions
- Try different camera indices
- Ensure no other application is using the camera

### Model Download Issues
- Check internet connection
- Model file location: `yolov8s-pose.pt` in project root
- Delete and re-run to re-download if corrupted

