#!/usr/bin/env python3
"""
Stereo Camera Calibration Tool (Stage 3)
Uses ChArUco or checkerboard pattern for stereo calibration.

Workflow:
  1. --capture: Capture stereo pairs with calibration target
  2. --calibrate: Process captured pairs and compute calibration
  3. --validate: Show reprojection error and baseline estimate

Chessboard/ChArUco dimensions and square size MUST match your physical target.
"""

import os
import sys
import glob
import json
import time
import yaml
import numpy as np

try:
    import cv2
    import cv2.aruco as aruco
except ImportError:
    print("ERROR: OpenCV with ArUco support required")
    sys.exit(1)


class NumpyDumper(yaml.SafeDumper):
    pass


def _numpy_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:float', float(data))


NumpyDumper.add_representer(np.floating, _numpy_representer)
NumpyDumper.add_representer(np.integer, lambda d, v: d.represent_scalar('tag:yaml.org,2002:int', int(v)))
NumpyDumper.add_representer(np.ndarray, lambda d, v: d.represent_list(v.tolist()))

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 required for capture mode")
    Picamera2 = None


class StereoCalibrator:
    """Stereo calibration using ChArUco or checkerboard patterns."""

    def __init__(self, config_path="config/stereo_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.board_type = self.config['calibration']['board_type']
        self.output_file = self.config['calibration']['output_file']

        if self.board_type == "charuco":
            board_cfg = self.config['calibration']['charuco']
            self.board_width = board_cfg['squares_x']
            self.board_height = board_cfg['squares_y']
            self.square_size = board_cfg['square_size_mm']
            self.marker_size = board_cfg['marker_size_mm']
            dict_name = board_cfg['dictionary']
        elif self.board_type == "checkerboard":
            board_cfg = self.config['calibration']['checkerboard']
            self.board_width = board_cfg['corners_x']
            self.board_height = board_cfg['corners_y']
            self.square_size = board_cfg['square_size_mm']
            self.marker_size = None
            dict_name = None
        else:
            raise ValueError(f"Unknown board type: {self.board_type}")

        self.resolution = (
            self.config['capture']['width'],
            self.config['capture']['height']
        )

        # Initialize ArUco dictionary if ChArUco
        if self.board_type == "charuco":
            self.aruco_dict = aruco.getPredefinedDictionary(
                getattr(aruco, dict_name)
            )
            self.charuco_board = aruco.CharucoBoard(
                (self.board_width, self.board_height),
                self.square_size,
                self.marker_size,
                self.aruco_dict
            )
            self.detector_params = aruco.DetectorParameters()
        else:
            self.charuco_board = None
            self.detector_params = None

    def detect_corners(self, image):
        """
        Detect calibration corners in an image.

        Returns:
            (corners, ids) tuple, or (None, None) if detection fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if self.board_type == "charuco":
            marker_corners, marker_ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
            if marker_ids is None or len(marker_ids) < 4:
                return None, None

            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.charuco_board
            )
            if not ret or charuco_corners is None or len(charuco_corners) < 6:
                return None, None

            return charuco_corners, charuco_ids

        else:  # checkerboard
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, (self.board_width, self.board_height), flags)
            if not ret:
                return None, None

            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners, None

    def capture_calibration_pairs(self, num_pairs=30, save_dir="captures/calibration"):
        """Interactive capture of calibration stereo pairs."""
        if Picamera2 is None:
            print("ERROR: picamera2 not available for capture")
            return

        left_id = self.config['cameras']['left']['camera_id']
        right_id = self.config['cameras']['right']['camera_id']

        print(f"Capturing calibration pairs at {self.resolution[0]}x{self.resolution[1]}")
        print(f"Board type: {self.board_type}")
        print(f"Target: {num_pairs} valid pairs")
        print(f"Move the calibration target to different positions and orientations.")
        print(f"Press Enter to capture, 'q' to finish early\n")

        cam_left = Picamera2(left_id)
        cam_right = Picamera2(right_id)

        config_left = cam_left.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        config_right = cam_right.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )

        cam_left.configure(config_left)
        cam_right.configure(config_right)

        cam_left.start()
        time.sleep(0.3)
        cam_right.start()
        time.sleep(2.0)

        os.makedirs(os.path.join(save_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "right"), exist_ok=True)

        valid_count = 0
        total_attempts = 0

        try:
            while valid_count < num_pairs:
                user_input = input(
                    f"Pair {valid_count + 1}/{num_pairs} [Enter to capture, q to quit]: "
                ).strip()
                if user_input.lower() == 'q':
                    break

                # Capture
                frame_left = cam_left.capture_array()
                frame_right = cam_right.capture_array()

                # Convert to BGR for OpenCV
                left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
                right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

                # Detect corners in both
                l_corners, l_ids = self.detect_corners(left_bgr)
                r_corners, r_ids = self.detect_corners(right_bgr)

                total_attempts += 1

                if l_corners is not None and r_corners is not None:
                    # Save the pair
                    idx = f"{valid_count:04d}"
                    cv2.imwrite(os.path.join(save_dir, "left", f"cal_{idx}.png"), left_bgr)
                    cv2.imwrite(os.path.join(save_dir, "right", f"cal_{idx}.png"), right_bgr)
                    valid_count += 1
                    print(f"  OK - corners detected in both ({len(l_corners)}L / {len(r_corners)}R)")
                else:
                    l_msg = f"{len(l_corners)} corners" if l_corners is not None else "NONE"
                    r_msg = f"{len(r_corners)} corners" if r_corners is not None else "NONE"
                    print(f"  SKIP - corners: L={l_msg}, R={r_msg}. Reposition target.")

        except KeyboardInterrupt:
            print(f"\nInterrupted after {valid_count} valid pairs.")
        finally:
            cam_left.stop()
            cam_left.close()
            cam_right.stop()
            cam_right.close()

        print(f"\nCaptured {valid_count} valid pairs from {total_attempts} attempts")
        print(f"Saved to: {os.path.abspath(save_dir)}/")

        # Save capture metadata
        meta = {
            "board_type": self.board_type,
            "resolution": list(self.resolution),
            "num_valid_pairs": valid_count,
            "num_attempts": total_attempts,
            "save_dir": save_dir,
        }
        with open(os.path.join(save_dir, "capture_info.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        return valid_count

    def calibrate(self, calibration_dir="captures/calibration", min_corners=6):
        """Run stereo calibration on captured pairs."""
        left_dir = os.path.join(calibration_dir, "left")
        right_dir = os.path.join(calibration_dir, "right")

        left_images = sorted(glob.glob(os.path.join(left_dir, "cal_*.png")))
        right_images = sorted(glob.glob(os.path.join(right_dir, "cal_*.png")))

        if len(left_images) != len(right_images):
            print(f"ERROR: Mismatched image counts: {len(left_images)} left, {len(right_images)} right")
            return False

        num_pairs = len(left_images)
        if num_pairs < self.config['calibration']['min_pairs']:
            print(f"ERROR: Need at least {self.config['calibration']['min_pairs']} pairs, "
                  f"found {num_pairs}")
            return False

        print(f"Calibrating with {num_pairs} stereo pairs...")
        print(f"Board type: {self.board_type}")
        print(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")

        # Detect corners in all pairs
        obj_points = []  # 3D world coordinates
        img_points_left = []
        img_points_right = []

        # Generate object points for the board
        if self.board_type == "charuco":
            objp = self.charuco_board.getChessboardCorners()
        else:
            objp = np.zeros((self.board_width * self.board_height, 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.board_width, 0:self.board_height].T.reshape(-1, 2)
            objp *= self.square_size

        valid_pairs = 0
        for i in range(num_pairs):
            left_img = cv2.imread(left_images[i])
            right_img = cv2.imread(right_images[i])

            if left_img is None or right_img is None:
                print(f"  Pair {i}: FAILED to read images")
                continue

            l_corners, l_ids = self.detect_corners(left_img)
            r_corners, r_ids = self.detect_corners(right_img)

            if l_corners is None or r_corners is None:
                print(f"  Pair {i}: corner detection failed")
                continue

            if self.board_type == "charuco":
                # Match corners by ID
                l_id_set = set(l_ids.flatten())
                r_id_set = set(r_ids.flatten())
                common_ids = sorted(l_id_set & r_id_set)

                if len(common_ids) < min_corners:
                    print(f"  Pair {i}: only {len(common_ids)} common corners (< {min_corners}, skipped)")
                    continue

                l_indices = [np.where(l_ids.flatten() == cid)[0][0] for cid in common_ids]
                r_indices = [np.where(r_ids.flatten() == cid)[0][0] for cid in common_ids]

                matched_l = l_corners[l_indices]
                matched_r = r_corners[r_indices]
                matched_obj = objp[common_ids]

                obj_points.append(matched_obj)
                img_points_left.append(matched_l)
                img_points_right.append(matched_r)
                valid_pairs += 1
                print(f"  Pair {i}: OK ({len(common_ids)} corners)")

            else:
                obj_points.append(objp.copy())
                img_points_left.append(l_corners)
                img_points_right.append(r_corners)
                valid_pairs += 1
                print(f"  Pair {i}: OK ({len(l_corners)} corners)")

        if valid_pairs < self.config['calibration']['min_pairs']:
            print(f"\nERROR: Only {valid_pairs} valid pairs. "
                  f"Need at least {self.config['calibration']['min_pairs']}.")
            return False

        print(f"\nRunning calibration with {valid_pairs} valid pairs...")

        h, w = self.resolution[1], self.resolution[0]

        # Calibrate individual cameras
        print("  Calibrating left camera...")
        ret_l, K_left, dist_left, rvecs_l, tvecs_l = cv2.calibrateCamera(
            obj_points, img_points_left, (w, h), None, None
        )
        print(f"  Left reprojection error: {ret_l:.4f} pixels")

        print("  Calibrating right camera...")
        ret_r, K_right, dist_right, rvecs_r, tvecs_r = cv2.calibrateCamera(
            obj_points, img_points_right, (w, h), None, None
        )
        print(f"  Right reprojection error: {ret_r:.4f} pixels")

        # Stereo calibration
        print("  Running stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC  # Use already-calibrated intrinsics
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_left, img_points_right,
            K_left, dist_left,
            K_right, dist_right,
            (w, h),
            criteria=criteria,
            flags=flags
        )
        print(f"  Stereo reprojection error: {ret_stereo:.4f} pixels")

        # Stereo rectification
        print("  Computing stereo rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, dist_left,
            K_right, dist_right,
            (w, h), R, T,
            alpha=0,  # 0 = crop, 1 = black borders
        )

        # Rectification maps are NOT saved - they are derived data, recomputed
        # from K/dist/R/P by whichever tool loads this file.

        # Compute baseline from full translation vector magnitude.
        # abs(T[0][0]) would assume cameras are perfectly parallel with zero
        # vertical/depth offset. Real mounts have tilt, so use the full vector.
        estimated_baseline = np.linalg.norm(T)
        physical_baseline = self.config['stereo']['baseline_mm']

        # Extract focal length from projection matrix
        depth_params_f = P1[0, 0]

        print(f"\n{'='*60}")
        print("CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Board type:          {self.board_type}")
        print(f"Resolution:          {w}x{h}")
        print(f"Valid pairs:         {valid_pairs}")
        print(f"Left cam error:      {ret_l:.4f} px")
        print(f"Right cam error:     {ret_r:.4f} px")
        print(f"Stereo error:        {ret_stereo:.4f} px")
        print(f"Estimated baseline:  {estimated_baseline:.2f} mm")
        print(f"Physical baseline:   {physical_baseline:.1f} mm (for reference)")

        # Validate reprojection error
        ERROR_THRESHOLD_WARN = 1.0  # pixels
        ERROR_THRESHOLD_FAIL = 2.0  # pixels

        print()
        if ret_stereo > ERROR_THRESHOLD_FAIL:
            print(f"*** FAIL: Stereo reprojection error {ret_stereo:.4f} px exceeds "
                  f"failure threshold ({ERROR_THRESHOLD_FAIL} px). ***")
            print("  Calibration is unreliable. Possible causes:")
            print("  - Poor corner detection (bad lighting, blurry images)")
            print("  - Too few calibration pairs")
            print("  - Incorrect board dimensions in config")
            print("  - Camera movement between captures")
            print("  RECOMMENDATION: Re-capture calibration pairs and re-run.")
        elif ret_stereo > ERROR_THRESHOLD_WARN:
            print(f"WARNING: Stereo reprojection error {ret_stereo:.4f} px exceeds "
                  f"warning threshold ({ERROR_THRESHOLD_WARN} px).")
            print("  Results may be usable but not ideal.")
            print("  Consider re-capturing with better lighting, more pairs,")
            print("  or more varied board positions/orientations.")
        else:
            print(f"OK: Stereo reprojection error {ret_stereo:.4f} px is within "
                  f"acceptable range (< {ERROR_THRESHOLD_WARN} px).")

        # Also check individual camera errors
        if ret_l > ERROR_THRESHOLD_WARN:
            print(f"WARNING: Left camera error {ret_l:.4f} px is high. "
                  f"Check left camera focus and calibration images.")
        if ret_r > ERROR_THRESHOLD_WARN:
            print(f"WARNING: Right camera error {ret_r:.4f} px is high. "
                  f"Check right camera focus and calibration images.")

        # Validate baseline sanity
        baseline_diff = abs(estimated_baseline - physical_baseline)
        if baseline_diff > 10:
            print(f"\nWARNING: Baseline difference {baseline_diff:.1f}mm is large "
                  f"(estimated {estimated_baseline:.2f}mm vs physical {physical_baseline:.1f}mm).")
            print("  Check board dimensions and units in config.")
            print("  The calibrated baseline will be used for depth computation.")
        elif baseline_diff > 5:
            print(f"  Baseline difference: {baseline_diff:.1f}mm "
                  f"(estimated {estimated_baseline:.2f}mm vs physical {physical_baseline:.1f}mm). "
                  f"Within acceptable range.")
        else:
            print(f"  Baseline: {estimated_baseline:.2f}mm estimated vs "
                  f"{physical_baseline:.1f}mm physical — good agreement.")

        # Save calibration file location
        calib_full_path = os.path.abspath(self.output_file)
        print(f"\n  Calibration saved to: {self.output_file}")
        print(f"  Full path: {calib_full_path}")

        # Summary of what will be used
        print(f"\n  The calibrated intrinsic/extrinsic parameters will be used for:")
        print(f"    - Stereo rectification")
        print(f"    - Disparity computation")
        print(f"    - Depth map generation (f={depth_params_f:.1f}px, B={estimated_baseline:.2f}mm)")
        print(f"{'='*60}")

        # Save calibration data
        calib_data = {
            "board_type": self.board_type,
            "resolution": {"width": w, "height": h},
            "num_valid_pairs": valid_pairs,
            "reprojection_error": {
                "left_camera": float(ret_l),
                "right_camera": float(ret_r),
                "stereo": float(ret_stereo),
            },
            "estimated_baseline_mm": float(estimated_baseline),
            "physical_baseline_mm": float(physical_baseline),
            "camera_left": {
                "K": K_left.tolist(),
                "dist": dist_left.tolist(),
            },
            "camera_right": {
                "K": K_right.tolist(),
                "dist": dist_right.tolist(),
            },
            "stereo": {
                "R": R.tolist(),
                "T": T.tolist(),
                "E": E.tolist(),
                "F": F.tolist(),
            },
            "rectification": {
                "R1": R1.tolist(),
                "R2": R2.tolist(),
                "P1": P1.tolist(),
                "P2": P2.tolist(),
                "Q": Q.tolist(),
                "roi1": list(roi1),
                "roi2": list(roi2),
            },
        }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            yaml.dump(calib_data, f, Dumper=NumpyDumper, default_flow_style=False)
        print(f"\nCalibration saved to: {self.output_file}")

        return True

    def validate(self, calibration_path=None):
        """Validate calibration by showing rectified images and reprojection."""
        cal_path = calibration_path or self.output_file

        if not os.path.exists(cal_path):
            print(f"ERROR: Calibration file not found: {cal_path}")
            return False

        with open(cal_path, 'r') as f:
            calib = yaml.safe_load(f)

        print(f"Calibration file: {cal_path}")
        print(f"Resolution: {calib['resolution']['width']}x{calib['resolution']['height']}")
        print(f"Valid pairs: {calib['num_valid_pairs']}")
        print(f"Left error: {calib['reprojection_error']['left_camera']:.4f} px")
        print(f"Right error: {calib['reprojection_error']['right_camera']:.4f} px")
        print(f"Stereo error: {calib['reprojection_error']['stereo']:.4f} px")
        print(f"Baseline: {calib['estimated_baseline_mm']:.2f} mm")

        # Load calibration matrices
        K_left = np.array(calib['camera_left']['K'])
        dist_left = np.array(calib['camera_left']['dist'])
        K_right = np.array(calib['camera_right']['K'])
        dist_right = np.array(calib['camera_right']['dist'])
        R1 = np.array(calib['rectification']['R1'])
        R2 = np.array(calib['rectification']['R2'])
        P1 = np.array(calib['rectification']['P1'])
        P2 = np.array(calib['rectification']['P2'])

        w = calib['resolution']['width']
        h = calib['resolution']['height']

        # Compute rectification maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, (w, h), cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, (w, h), cv2.CV_32FC1
        )

        # Find a recent stereo pair to test
        cal_dir = "captures/calibration"
        left_images = sorted(glob.glob(os.path.join(cal_dir, "left", "cal_*.png")))
        right_images = sorted(glob.glob(os.path.join(cal_dir, "right", "cal_*.png")))

        if len(left_images) == 0:
            print("\nNo calibration images found for validation.")
            return True

        # Use last pair
        idx = len(left_images) - 1
        left_img = cv2.imread(left_images[idx])
        right_img = cv2.imread(right_images[idx])

        # Apply rectification
        rect_left = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_LINEAR)

        # Save rectified pair
        os.makedirs("output/calibration", exist_ok=True)
        cv2.imwrite("output/calibration/rectified_left.png", rect_left)
        cv2.imwrite("output/calibration/rectified_right.png", rect_right)

        # Create side-by-side with epipolar lines
        h_line = 2
        sidebyside = np.zeros((h, w * 2, 3), dtype=np.uint8)
        sidebyside[:, :w] = rect_left
        sidebyside[:, w:] = rect_right

        for y in range(0, h, 40):
            cv2.line(sidebyside, (0, y), (w * 2, y), (0, 255, 0), h_line)

        cv2.imwrite("output/calibration/rectified_epipolar.png", sidebyside)
        print(f"\nRectified images saved to output/calibration/")
        print("  rectified_left.png")
        print("  rectified_right.png")
        print("  rectified_epipolar.png (with epipolar lines)")
        print("\nCheck that corresponding features lie on the same horizontal line.")

        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("--config", type=str, default="config/stereo_config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    # Capture subcommand
    cap_parser = subparsers.add_parser("capture", help="Capture calibration pairs")
    cap_parser.add_argument("--num-pairs", type=int, default=30,
                            help="Number of pairs to capture")
    cap_parser.add_argument("--save-dir", type=str, default="captures/calibration")

    # Calibrate subcommand
    cal_parser = subparsers.add_parser("calibrate", help="Run stereo calibration")
    cal_parser.add_argument("--data-dir", type=str, default="captures/calibration")
    cal_parser.add_argument("--min-corners", type=int, default=6,
                            help="Skip pairs with fewer common corners (default: 6)")

    # Validate subcommand
    val_parser = subparsers.add_parser("validate", help="Validate calibration")
    val_parser.add_argument("--calibration", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    calibrator = StereoCalibrator(args.config)

    if args.command == "capture":
        calibrator.capture_calibration_pairs(args.num_pairs, args.save_dir)
    elif args.command == "calibrate":
        calibrator.calibrate(args.data_dir, args.min_corners)
    elif args.command == "validate":
        calibrator.validate(args.calibration)


if __name__ == "__main__":
    main()
