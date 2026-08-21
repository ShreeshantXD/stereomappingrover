#!/usr/bin/env python3
"""
Full Stereo Pipeline Runner
Captures a stereo pair, runs calibration (if needed), and generates depth + point cloud.

Usage:
    python3 run_pipeline.py                     # Full pipeline with live capture
    python3 run_pipeline.py --from-files L R    # Process existing image pair
    python3 run_pipeline.py --calibrate         # Only run calibration workflow
"""

import os
import sys
import time
import argparse
import subprocess
import yaml
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV required")
    sys.exit(1)


def load_config(config_path="config/stereo_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_camera_verify():
    """Stage 1: Verify both cameras work."""
    print("=" * 60)
    print("STAGE 1: Camera Verification")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "scripts/camera_verify.py"],
        capture_output=False
    )
    return result.returncode == 0


def run_stereo_capture(num_pairs=1):
    """Stage 2: Capture stereo pairs."""
    print("=" * 60)
    print("STAGE 2: Stereo Capture")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "scripts/stereo_capture.py",
         "--num-pairs", str(num_pairs)],
        capture_output=False
    )
    return result.returncode == 0


def run_calibration(skip_capture=False, num_cal_pairs=30):
    """Stage 3: Calibration workflow."""
    print("=" * 60)
    print("STAGE 3: Stereo Calibration")
    print("=" * 60)

    if not skip_capture:
        print("\n--- Capturing calibration pairs ---")
        result = subprocess.run(
            [sys.executable, "scripts/calibrate_stereo.py", "capture",
             "--num-pairs", str(num_cal_pairs)],
            capture_output=False
        )
        if result.returncode != 0:
            print("Calibration capture failed")
            return False

    print("\n--- Running calibration ---")
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_stereo.py", "calibrate"],
        capture_output=False
    )
    if result.returncode != 0:
        print("Calibration failed")
        return False

    print("\n--- Validating calibration ---")
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_stereo.py", "validate"],
        capture_output=False
    )
    return result.returncode == 0


def run_stereo_depth(left_path, right_path, output_dir="output"):
    """Stages 4-7: Rectify, disparity, depth, point cloud using C++ pipeline."""
    print("=" * 60)
    print("STAGES 4-7: Stereo Depth Pipeline (C++)")
    print("=" * 60)

    exe = "build/stereo_pipeline"
    if not os.path.exists(exe):
        print("C++ pipeline not built yet. Building...")
        os.makedirs("build", exist_ok=True)
        result = subprocess.run(
            ["cmake", ".."], cwd="build"
        )
        if result.returncode != 0:
            print("ERROR: cmake failed")
            return False
        result = subprocess.run(
            ["make", "-j4"], cwd="build"
        )
        if result.returncode != 0:
            print("ERROR: build failed")
            return False

    result = subprocess.run(
        [exe,
         "--left", left_path,
         "--right", right_path,
         "--output-dir", output_dir,
         "--save-all"],
        capture_output=False
    )
    return result.returncode == 0


def run_stereo_depth_python(left_path, right_path, output_dir="output"):
    """
    Fallback: Full stereo depth pipeline in Python using OpenCV.
    Used if C++ build is not available.
    """
    print("=" * 60)
    print("STAGES 4-7: Stereo Depth Pipeline (Python fallback)")
    print("=" * 60)

    config = load_config()
    calib_path = config['calibration']['output_file']

    if not os.path.exists(calib_path):
        print(f"ERROR: Calibration file not found: {calib_path}")
        print("Run calibration first: python3 scripts/calibrate_stereo.py calibrate")
        return False

    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)

    # Load calibration matrices
    K_left = np.array(calib['camera_left']['K'])
    dist_left = np.array(calib['camera_left']['dist'])
    K_right = np.array(calib['camera_right']['K'])
    dist_right = np.array(calib['camera_right']['dist'])
    R1 = np.array(calib['rectification']['R1'])
    R2 = np.array(calib['rectification']['R2'])
    P1 = np.array(calib['rectification']['P1'])
    P2 = np.array(calib['rectification']['P2'])
    Q = np.array(calib['rectification']['Q'])

    w = calib['resolution']['width']
    h = calib['resolution']['height']
    baseline_mm = calib['estimated_baseline_mm']
    focal_px = P1[0, 0]

    print(f"  Baseline: {baseline_mm:.2f} mm")
    print(f"  Focal length: {focal_px:.2f} px")

    # Step 1: Rectify
    print("\n[1/4] Rectifying...")
    t0 = time.time()

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, (w, h), cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, (w, h), cv2.CV_32FC1
    )

    rect_left = cv2.remap(left, map_left_x, map_left_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map_right_x, map_right_y, cv2.INTER_LINEAR)
    print(f"  Done in {(time.time()-t0)*1000:.0f} ms")

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/rectified_left.png", rect_left)
    cv2.imwrite(f"{output_dir}/rectified_right.png", rect_right)

    # Epipolar lines
    sidebyside = np.zeros((h, w * 2, 3), dtype=np.uint8)
    sidebyside[:, :w] = rect_left
    sidebyside[:, w:] = rect_right
    for y_line in range(0, h, 40):
        cv2.line(sidebyside, (0, y_line), (w * 2, y_line), (0, 255, 0), 1)
    cv2.imwrite(f"{output_dir}/rectified_epipolar.png", sidebyside)

    # Step 2: Disparity
    print("[2/4] Computing disparity...")
    t1 = time.time()

    disp_cfg = config['disparity']['sgbm']
    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=disp_cfg['min_disparity'],
        numDisparities=disp_cfg['num_disparities'],
        blockSize=disp_cfg['block_size'],
        P1=disp_cfg['P1'],
        P2=disp_cfg['P2'],
        disp12MaxDiff=disp_cfg['disp12_max_diff'],
        uniquenessRatio=disp_cfg['uniqueness_ratio'],
        speckleWindowSize=disp_cfg['speckle_window_size'],
        speckleRange=disp_cfg['speckle_range'],
        preFilterCap=disp_cfg['pre_filter_cap'],
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(gray_l, gray_r)

    # WLS filter if available
    if config['disparity'].get('wls_filter', False) and hasattr(cv2, 'ximgproc'):
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            disp_right = right_matcher.compute(gray_r, gray_l)
            wls = cv2.ximgproc.createDisparityWLSFilter(stereo)
            wls.setLambda(config['disparity']['wls_lambda'])
            wls.setSigmaColor(config['disparity']['wls_sigma'])
            disparity = wls.filter(disparity, gray_l, disparity, disp_right)
            print("  WLS filter applied")
        except Exception as e:
            print(f"  WLS filter unavailable: {e}")

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/disparity.png", disp_vis)
    print(f"  Done in {(time.time()-t1)*1000:.0f} ms")

    # Step 3: Depth
    print("[3/4] Computing depth...")
    t2 = time.time()

    depth_min = config['depth']['min_depth_mm']
    depth_max = config['depth']['max_depth_mm']

    depth = np.zeros(disparity.shape, dtype=np.float64)
    valid_disp = disparity > 0
    disp_vals = disparity[valid_disp].astype(np.float64) / 16.0
    Z = (focal_px * baseline_mm) / disp_vals
    Z[Z < depth_min] = 0
    Z[Z > depth_max] = 0
    depth[valid_disp] = Z

    valid_depth = depth > 0
    print(f"  Valid depth pixels: {valid_depth.sum()} / {depth.size} "
          f"({100*valid_depth.sum()/depth.size:.1f}%)")

    depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
    if valid_depth.any():
        d_min = depth[valid_depth].min()
        d_max = depth[valid_depth].max()
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        depth_norm[valid_depth] = ((depth[valid_depth] - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/depth.png", depth_vis)
    print(f"  Done in {(time.time()-t2)*1000:.0f} ms")

    # Step 4: Point cloud
    print("[4/4] Generating point cloud...")
    t3 = time.time()

    fx = P1[0, 0]
    fy = P1[1, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]

    ys, xs = np.where(valid_depth)
    Z_vals = depth[ys, xs]
    X_vals = (xs - cx) * Z_vals / fx
    Y_vals = (ys - cy) * Z_vals / fy

    points = np.column_stack([X_vals, Y_vals, Z_vals]).astype(np.float32)

    # Get colors from left image
    if rect_left.shape[2] == 3:
        colors_rgb = rect_left[ys, xs][:, ::-1]  # BGR -> RGB
    else:
        colors_rgb = np.full((len(xs), 3), 255, dtype=np.uint8)

    # Downsample if too many points
    max_pts = config['pointcloud']['max_points']
    if len(points) > max_pts:
        step = len(points) // max_pts
        points = points[::step]
        colors_rgb = colors_rgb[::step]

    # Save PLY
    ply_path = f"{output_dir}/pointcloud.ply"
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                    f"{colors_rgb[i,0]} {colors_rgb[i,1]} {colors_rgb[i,2]}\n")

    print(f"  Points: {len(points)}")
    print(f"  PLY saved: {ply_path}")
    print(f"  Done in {(time.time()-t3)*1000:.0f} ms")

    print(f"\nAll outputs saved to {output_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run stereo mapping pipeline")
    parser.add_argument("--from-files", nargs=2, metavar=("LEFT", "RIGHT"),
                        help="Process existing image files instead of live capture")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run calibration workflow")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify cameras work")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip camera verification")
    parser.add_argument("--skip-calibrate", action="store_true",
                        help="Skip calibration (use existing)")
    parser.add_argument("--use-python", action="store_true",
                        help="Use Python pipeline instead of C++")
    parser.add_argument("--config", type=str, default="config/stereo_config.yaml")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--num-cal-pairs", type=int, default=30,
                        help="Number of calibration pairs to capture")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("STEREO MAPPING PIPELINE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print()

    # Stage 1: Camera verification
    if not args.skip_verify and not args.from_files:
        if not run_camera_verify():
            print("\nFATAL: Camera verification failed. Fix camera setup first.")
            sys.exit(1)
        print()

    if args.verify_only:
        return

    # Stage 3: Calibration (if needed)
    calib_file = load_config(args.config)['calibration']['output_file']
    if not args.skip_calibrate and not os.path.exists(calib_file):
        print("No calibration file found. Running calibration...")
        if not run_calibration(num_cal_pairs=args.num_cal_pairs):
            print("\nFATAL: Calibration failed.")
            sys.exit(1)
        print()

    if args.calibrate_only:
        return

    # Stages 2, 4-7: Capture + Process
    if args.from_files:
        left_path, right_path = args.from_files
        print(f"Processing: {left_path}, {right_path}")
    else:
        # Capture a stereo pair
        if not run_stereo_capture(num_pairs=1):
            print("\nFATAL: Stereo capture failed.")
            sys.exit(1)

        # Find the most recent capture
        import glob
        left_files = sorted(glob.glob("captures/left_*.png"))
        right_files = sorted(glob.glob("captures/right_*.png"))
        if not left_files or not right_files:
            print("ERROR: No captures found")
            sys.exit(1)
        left_path = left_files[-1]
        right_path = right_files[-1]
        print(f"\nUsing captured pair:\n  Left:  {left_path}\n  Right: {right_path}")

    # Run stereo depth pipeline
    if args.use_python or not os.path.exists("build/stereo_pipeline"):
        if not os.path.exists("build/stereo_pipeline"):
            print("C++ pipeline not built, using Python fallback")
        success = run_stereo_depth_python(left_path, right_path, args.output_dir)
    else:
        success = run_stereo_depth(left_path, right_path, args.output_dir)

    if success:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutputs in: {args.output_dir}/")
        print("Next steps:")
        print("  1. Check rectified_epipolar.png for alignment")
        print("  2. Inspect disparity.png and depth.png")
        print("  3. Open pointcloud.ply in MeshLab or CloudCompare")
        print("  4. Run visualize_outputs.py --dir output/ for a report")
    else:
        print("\nPIPELINE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
