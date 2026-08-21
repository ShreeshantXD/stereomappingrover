#!/usr/bin/env python3
"""
Camera Verification Script (Stage 1)
Detects both IMX708 CSI cameras and captures test images.
Verifies cameras can produce usable frames.
"""

import os
import sys
import time
from datetime import datetime

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 not available. Install with: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not available. Install with: sudo apt install python3-opencv")
    sys.exit(1)


def list_cameras():
    """List all available Picamera2 cameras and their info, mapped to left/right."""
    print("=" * 60)
    print("Camera Detection")
    print("=" * 60)

    cameras = Picamera2.global_camera_info()
    print(f"\nFound {len(cameras)} camera(s):")

    # Default mapping: camera_id 0 = left, camera_id 1 = right
    # This matches config/stereo_config.yaml
    left_id = 0
    right_id = 1
    role_map = {left_id: "LEFT", right_id: "RIGHT"}

    camera_info = {}
    for i, cam in enumerate(cameras):
        role = role_map.get(i, f"CAM{i}")
        num = cam.get('Num', i)
        location = cam.get('Location', 'unknown')
        rotation = cam.get('Rotation', 'unknown')
        cam_id = cam.get('Id', 'unknown')
        model = cam.get('Model', 'unknown')

        print(f"\n  Camera {i} [{role}]:")
        print(f"    Picamera2 Num:        {num}")
        print(f"    Model:                {model}")
        print(f"    Location:             {location}")
        print(f"    Rotation:             {rotation}")
        print(f"    Id (libcamera path):  {cam_id}")
        print(f"    Assigned role:        {role}")

        camera_info[i] = {
            'role': role,
            'num': num,
            'location': location,
            'rotation': rotation,
            'id': cam_id,
            'model': model,
        }

    if len(cameras) < 2:
        print(f"\n  WARNING: Expected 2 cameras for stereo, found {len(cameras)}")
    else:
        print(f"\n  Stereo assignment: Camera {left_id} = LEFT, Camera {right_id} = RIGHT")
        print(f"  (as configured in config/stereo_config.yaml)")

    print()

    return cameras, camera_info


def test_camera(camera_id, role_label, resolution=(1536, 864), save_dir="captures"):
    """Test a single camera by capturing and saving an image with unambiguous label."""
    print(f"--- Testing Camera {camera_id} [{role_label}] at {resolution[0]}x{resolution[1]} ---")

    try:
        picam = Picamera2(camera_id)
    except Exception as e:
        print(f"  FAIL: Cannot open camera {camera_id} [{role_label}]: {e}")
        return False

    try:
        config = picam.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()

        # Let auto-exposure/auto-white-balance settle
        time.sleep(2.0)

        frame = picam.capture_array()
        picam.stop()

        if frame is None or frame.size == 0:
            print(f"  FAIL: Camera {camera_id} [{role_label}] returned empty frame")
            return False

        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")

        # Check for blank/black/dead frame
        frame_mean = frame.mean()
        frame_std = frame.std()
        print(f"  Mean pixel: {frame_mean:.1f}, Std dev: {frame_std:.1f}")

        if frame_mean < 5.0:
            print(f"  FAIL: Camera {camera_id} [{role_label}] returned near-black frame "
                  f"(mean={frame_mean:.1f}). Check exposure/connector/IR filter.")
            return False

        if frame_std < 2.0:
            print(f"  FAIL: Camera {camera_id} [{role_label}] returned uniform frame "
                  f"(std={frame_std:.1f}). Sensor may be covered or malfunctioning.")
            return False

        # Save the image with unambiguous label
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"verify_{role_label.lower()}_{timestamp}.png")

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
        print(f"  Saved: {filename}")
        print(f"  PASS: Camera {camera_id} [{role_label}] captured {frame.shape[1]}x{frame.shape[0]} frame")

        picam.close()
        return True

    except Exception as e:
        print(f"  FAIL: Camera {camera_id} [{role_label}] error: {e}")
        try:
            picam.stop()
        except:
            pass
        try:
            picam.close()
        except:
            pass
        return False


def test_stereo_pair(resolution=(1536, 864), save_dir="captures"):
    """Capture a near-simultaneous stereo pair and verify overlap."""
    print("=" * 60)
    print("Stereo Pair Capture & Overlap Test")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cam_left = None
    cam_right = None
    try:
        # Open both cameras
        cam_left = Picamera2(0)
        cam_right = Picamera2(1)

        config_left = cam_left.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        config_right = cam_right.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )

        cam_left.configure(config_left)
        cam_right.configure(config_right)

        # Start both cameras (but capture sequentially due to CSI limitation)
        cam_left.start()
        time.sleep(0.5)
        cam_right.start()
        time.sleep(2.0)  # let AE settle

        # Capture left
        t_start = time.time()
        frame_left = cam_left.capture_array()
        t_left = time.time()

        # Capture right (as close to left as possible)
        frame_right = cam_right.capture_array()
        t_right = time.time()

        gap_ms = (t_right - t_left) * 1000
        total_ms = (t_right - t_start) * 1000

        print(f"\n  Left frame:  {frame_left.shape}")
        print(f"  Right frame: {frame_right.shape}")
        print(f"  Time gap (L->R): {gap_ms:.1f} ms")
        print(f"  Total capture time: {total_ms:.1f} ms")

        if gap_ms > 100:
            print(f"  WARNING: {gap_ms:.0f}ms gap between captures. "
                  f"Static scenes only for accurate depth.")

        # --- Content sanity checks ---
        checks_passed = True

        # 1. Check neither frame is blank/black
        left_mean = frame_left.mean()
        right_mean = frame_right.mean()
        left_std = frame_left.std()
        right_std = frame_right.std()

        print(f"\n  Left brightness:  mean={left_mean:.1f}, std={left_std:.1f}")
        print(f"  Right brightness: mean={right_mean:.1f}, std={right_std:.1f}")

        if left_mean < 5.0:
            print("  FAIL: Left frame is near-black")
            checks_passed = False
        if right_mean < 5.0:
            print("  FAIL: Right frame is near-black")
            checks_passed = False
        if left_std < 2.0:
            print("  FAIL: Left frame is uniform (sensor covered?)")
            checks_passed = False
        if right_std < 2.0:
            print("  FAIL: Right frame is uniform (sensor covered?)")
            checks_passed = False

        # 2. Check brightness difference
        brightness_diff = abs(left_mean - right_mean)
        print(f"  Brightness diff:  {brightness_diff:.1f}")
        if brightness_diff > 50:
            print("  WARNING: Large brightness difference between cameras. "
                  "Check exposure/gain settings.")

        # 3. Basic overlap check: compare horizontal structure
        #    Both images should have similar spatial frequency content
        #    if they're pointed at the same scene.
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY) if len(frame_left.shape) == 3 else frame_left
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY) if len(frame_right.shape) == 3 else frame_right

        # Compute horizontal gradient magnitudes as a proxy for edge content
        grad_left = cv2.Sobel(gray_left, cv2.CV_64F, 1, 0, ksize=3)
        grad_right = cv2.Sobel(gray_right, cv2.CV_64F, 1, 0, ksize=3)
        grad_left_mag = np.abs(grad_left).mean()
        grad_right_mag = np.abs(grad_right).mean()

        print(f"  Left edge energy:  {grad_left_mag:.1f}")
        print(f"  Right edge energy: {grad_right_mag:.1f}")

        if grad_left_mag < 1.0 or grad_right_mag < 1.0:
            print("  WARNING: Very low edge energy. Images may be out of focus or pointed at blank wall.")
        else:
            # Check that both cameras see roughly similar scene complexity
            edge_ratio = min(grad_left_mag, grad_right_mag) / max(grad_left_mag, grad_right_mag)
            print(f"  Edge energy ratio: {edge_ratio:.2f} (1.0 = identical)")
            if edge_ratio < 0.2:
                print("  WARNING: Very different scene content between cameras. "
                      "Check that both cameras point at the same area.")
            else:
                print("  OK: Both cameras see comparable scene content.")

        if not checks_passed:
            print("\n  FAIL: One or more content checks failed.")
            return False

        # Save stereo pair with unambiguous labels
        left_path = os.path.join(save_dir, f"verify_left_{timestamp}.png")
        right_path = os.path.join(save_dir, f"verify_right_{timestamp}.png")

        cv2.imwrite(left_path, cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(right_path, cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR))

        print(f"\n  Saved LEFT:  {left_path}")
        print(f"  Saved RIGHT: {right_path}")

        # Create side-by-side comparison with labels
        sidebyside = np.hstack([
            cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        ])
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(sidebyside, "LEFT (cam0)", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(sidebyside, "RIGHT (cam1)", (resolution[0] + 10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(sidebyside, f"Gap: {gap_ms:.0f}ms", (10, 60), font, 0.6, (0, 255, 255), 1)

        sbs_path = os.path.join(save_dir, f"verify_stereo_sidebyside_{timestamp}.png")
        cv2.imwrite(sbs_path, sidebyside)
        print(f"  Saved side-by-side: {sbs_path}")

        print("\n  PASS: Stereo pair captured and content checks passed")
        return True

    except Exception as e:
        print(f"  FAIL: Stereo capture error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for cam in (cam_left, cam_right):
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
                try:
                    cam.close()
                except Exception:
                    pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify stereo camera setup")
    parser.add_argument("--width", type=int, default=1536, help="Capture width")
    parser.add_argument("--height", type=int, default=864, help="Capture height")
    parser.add_argument("--save-dir", type=str, default="captures", help="Output directory")
    parser.add_argument("--skip-stereo", action="store_true", help="Skip stereo pair test")
    args = parser.parse_args()

    resolution = (args.width, args.height)

    # Step 1: List cameras with left/right role mapping
    cameras, camera_info = list_cameras()
    if len(cameras) < 2:
        print("ERROR: Expected 2 cameras, found", len(cameras))
        print("Check CSI camera connections and libcamera configuration.")
        sys.exit(1)

    # Step 2: Test each camera individually with left/right labels
    results = []
    for cam_id in range(len(cameras)):
        role_label = camera_info[cam_id]['role']
        ok = test_camera(cam_id, role_label, resolution, args.save_dir)
        results.append(ok)

    if not all(results):
        print("\nERROR: One or more cameras failed individual test.")
        sys.exit(1)

    # Step 3: Test stereo pair with overlap verification
    if not args.skip_stereo:
        stereo_ok = test_stereo_pair(resolution, args.save_dir)
        if not stereo_ok:
            print("\nWARNING: Stereo pair test failed.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nImages saved to: {os.path.abspath(args.save_dir)}/")
    print("Verify manually that both images show the same scene with sufficient overlap.")


if __name__ == "__main__":
    main()
