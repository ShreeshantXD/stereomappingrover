# Stereo 3D Mapping — Raspberry Pi 5 Rover

Stereo vision pipeline for 3D environment mapping using dual IMX708 CSI cameras on a Raspberry Pi 5.

## Hardware

| Component | Detail |
|-----------|--------|
| Platform | Raspberry Pi 5, 4 GB RAM, 32 GB storage |
| Cameras | 2x Sony IMX708 (CSI, not USB) |
| Resolution | 4608x2592 native, 1536x864 for stereo |
| Baseline | ~85 mm horizontal separation |
| OS | Linux (Raspberry Pi OS, kernel 6.18) |
| Libcamera | v0.7.2+rpt20260817 |

## Pipeline Architecture

```
IMX708 Left  ─┐
               ├─→ Stereo Capture (picamera2) ─→ Stereo Calibration ─→ Calib Data (YAML)
IMX708 Right ─┘
                                              ─→ Stereo Rectification
                                              ─→ Disparity (SGBM + WLS)
                                              ─→ Depth Map
                                              ─→ 3D Point Cloud (PLY)
```

## Quick Start

```bash
# 1. Install dependencies
chmod +x setup_env.sh && ./setup_env.sh

# 2. Verify both cameras work
python3 scripts/camera_verify.py

# 3. Capture calibration pairs (move target around, 30 pairs)
python3 scripts/calibrate_stereo.py capture --num-pairs 30

# 4. Run calibration
python3 scripts/calibrate_stereo.py calibrate

# 5. Validate calibration
python3 scripts/calibrate_stereo.py validate

# 6. Build C++ pipeline
chmod +x build.sh && ./build.sh

# 7. Capture and process a stereo pair
python3 scripts/run_pipeline.py

# Or process existing images:
python3 scripts/run_pipeline.py --from-files captures/left.png captures/right.png
```

## Project Structure

```
stereo_mapping/
├── CMakeLists.txt              # C++ build
├── build.sh                    # Build helper
├── setup_env.sh                # Dependency installer
├── config/
│   ├── stereo_config.yaml      # Main configuration
│   └── calibration/            # Calibration output (stereo_calib.yaml)
├── src/
│   ├── main.cpp                # C++ pipeline entry point
│   └── stereo_pipeline.cpp     # Core stereo processing (C++)
├── include/
│   └── stereo_pipeline.h       # C++ header
├── scripts/
│   ├── camera_verify.py        # Stage 1: Camera testing
│   ├── stereo_capture.py       # Stage 2: Stereo capture
│   ├── calibrate_stereo.py     # Stage 3: Calibration
│   ├── run_pipeline.py         # Full pipeline runner
│   └── visualize_outputs.py    # Offline visualization
├── captures/                   # Captured stereo pairs
├── calibration/                # Raw calibration data
├── output/                     # Pipeline outputs (PLY, images)
└── build/                      # C++ build directory
```

## Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `camera_verify.py` | Detect cameras, test capture |
| 2 | `stereo_capture.py` | Acquire stereo image pairs |
| 3 | `calibrate_stereo.py` | ChArUco/checkerboard calibration |
| 4 | `stereo_pipeline` (C++) | Rectification, disparity, depth, PLY |
| 5 | `visualize_outputs.py` | Offline viewing, HTML report |

## Configuration

Edit `config/stereo_config.yaml` for:

- Camera IDs and rotation
- Capture resolution (1536x864 recommended)
- Calibration board parameters (must match your physical target)
- Disparity algorithm parameters (SGBM tuning)
- Depth range limits
- Point cloud settings

## Calibration Target

Default: **ChArUco board** (5x7 squares, 30mm square size)

You can print a ChArUco board using:
```python
import cv2.aruco as aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 7), 30.0, 22.0, dictionary)
img = board.generateImage((1500, 2100))
cv2.imwrite("charuco_target.png", img)
```

## Viewing Point Clouds

Transfer `output/pointcloud.ply` to a PC and open with:

- **MeshLab**: File > Import Mesh
- **CloudCompare**: File > Open (best for point clouds)
- **Blender**: File > Import > PLY
- **Python**: `open3d.io.read_point_cloud("pointcloud.ply")`

## Known Limitations

1. **No hardware sync**: Left/right CSI captures are sequential (~10-50ms gap). Static scenes only for accurate depth.
2. **4 GB RAM**: Keep resolution at 1536x864. Avoid full 4608x2592 for stereo.
3. **CPU-only**: No CUDA on Pi5. SGBM is the most practical algorithm.
4. **Calibration required**: Do not use without running calibration first.

## Future Work (Not Implemented Yet)

- Visual odometry / SLAM
- Persistent 3D mapping across rover movement
- LiDAR fusion (sensor not yet identified)
- ROS 2 integration
- Real-time streaming
