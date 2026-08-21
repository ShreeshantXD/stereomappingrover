#!/bin/bash
set -e

echo "=== Stereo Mapping Environment Setup ==="
echo "Target: Raspberry Pi 5 (4GB RAM, 32GB storage)"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $PRETTY_NAME"
else
    echo "Warning: Cannot detect OS version"
fi

# Check for Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Raspberry Pi detected"
    IS_PI=true
else
    echo "Not running on a Raspberry Pi (cross-development?)"
    IS_PI=false
fi

echo ""
echo "--- Updating package lists ---"
sudo apt-get update

echo ""
echo "--- Installing build essentials ---"
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git

echo ""
echo "--- Installing OpenCV development libraries ---"
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv

echo ""
echo "--- Installing Python dependencies ---"
sudo apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-yaml

# Picamera2 should already be present on Raspberry Pi OS with libcamera
echo ""
echo "--- Checking picamera2 ---"
python3 -c "from picamera2 import Picamera2; print('picamera2: OK')" 2>/dev/null || {
    echo "picamera2 not found, installing..."
    sudo apt-get install -y python3-picamera2
}

echo ""
echo "--- Installing Open3D for point cloud visualization (optional, may be slow) ---"
read -p "Install Open3D for PLY visualization? (y/N): " INSTALL_OPEN3D
if [ "$INSTALL_OPEN3D" = "y" ] || [ "$INSTALL_OPEN3D" = "Y" ]; then
    pip3 install open3d || echo "Warning: Open3D install failed, PLY viewing will need external tool"
fi

echo ""
echo "--- Installing PLY file support ---"
pip3 install plyfile || echo "Warning: plyfile install failed"

echo ""
echo "--- Verifying installation ---"
echo -n "C++ compiler: "
g++ --version | head -1
echo -n "CMake: "
cmake --version | head -1
echo -n "OpenCV: "
python3 -c "import cv2; print(f'{cv2.__version__} (Python)')"
pkg-config --modversion opencv4 2>/dev/null && echo "OpenCV C++: $(pkg-config --modversion opencv4)" || echo "OpenCV C++: pkg-config not finding opencv4"
echo -n "NumPy: "
python3 -c "import numpy; print(numpy.__version__)"
echo -n "Picamera2: "
python3 -c "from picamera2 import Picamera2; print('available')" 2>/dev/null || echo "NOT AVAILABLE"

echo ""
echo "=== Environment setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Build the C++ pipeline:  cd build && cmake .. && make -j\$(nproc)"
echo "  2. Test camera capture:     python3 scripts/camera_verify.py"
echo "  3. Capture calibration:     python3 scripts/calibrate_stereo.py --capture"
echo "  4. Run calibration:         python3 scripts/calibrate_stereo.py --calibrate"
echo "  5. Run stereo pipeline:     ./build/stereo_pipeline"
