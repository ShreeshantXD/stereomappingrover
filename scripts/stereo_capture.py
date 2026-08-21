#!/usr/bin/env python3
"""
Stereo Capture Utility (Stage 2)
Captures synchronized/near-synchronized stereo image pairs.
Separate from processing logic - only handles acquisition.
"""

import os
import sys
import time
import json
import yaml
from datetime import datetime

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 not available")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not available")
    sys.exit(1)


class StereoCapturer:
    """
    Manages stereo camera capture from dual CSI IMX708 cameras.

    Synchronization note:
    The two CSI cameras on Pi5 are connected through separate RP1 CFE
    interfaces. Picamera2 cannot guarantee hardware synchronization.
    Frames are captured sequentially (left then right), introducing a
    small time gap (typically 10-50ms). This is acceptable for static
    or slow-moving scenes but should be documented as a limitation.
    """

    def __init__(self, config_path="config/stereo_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.left_id = self.config['cameras']['left']['camera_id']
        self.right_id = self.config['cameras']['right']['camera_id']
        self.width = self.config['capture']['width']
        self.height = self.config['capture']['height']
        self.format = self.config['capture'].get('format', 'RGB888')
        self.max_gap_ms = self.config['sync']['max_time_gap_ms']

        self.cam_left = None
        self.cam_right = None
        self._initialized = False

    def initialize(self):
        """Initialize both cameras with configured resolution."""
        print(f"Initializing stereo capture at {self.width}x{self.height}...")

        self.cam_left = Picamera2(self.left_id)
        self.cam_right = Picamera2(self.right_id)

        config_left = self.cam_left.create_still_configuration(
            main={"size": (self.width, self.height), "format": self.format}
        )
        config_right = self.cam_right.create_still_configuration(
            main={"size": (self.width, self.height), "format": self.format}
        )

        self.cam_left.configure(config_left)
        self.cam_right.configure(config_right)

        # Start cameras and let auto-exposure settle
        self.cam_left.start()
        time.sleep(0.3)
        self.cam_right.start()
        time.sleep(2.0)

        self._initialized = True
        print("Stereo capture initialized.")

    def shutdown(self):
        """Stop and release both cameras."""
        if self.cam_left:
            try:
                self.cam_left.stop()
                self.cam_left.close()
            except:
                pass
        if self.cam_right:
            try:
                self.cam_right.stop()
                self.cam_right.close()
            except:
                pass
        self._initialized = False
        print("Stereo capture shut down.")

    def capture_pair(self):
        """
        Capture a near-simultaneous stereo pair.

        Returns:
            dict with keys: left, right, timestamp, gap_ms, metadata
            Returns None on failure.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before capturing")

        try:
            t0 = time.time()
            frame_left = self.cam_left.capture_array()
            t_left = time.time()

            frame_right = self.cam_right.capture_array()
            t_right = time.time()

            gap_ms = (t_right - t_left) * 1000

            if gap_ms > self.max_gap_ms:
                print(f"WARNING: Capture gap {gap_ms:.1f}ms exceeds {self.max_gap_ms}ms threshold")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            metadata = {
                "timestamp": timestamp,
                "gap_ms": round(gap_ms, 2),
                "left_shape": list(frame_left.shape),
                "right_shape": list(frame_right.shape),
                "left_mean_brightness": round(float(frame_left.mean()), 1),
                "right_mean_brightness": round(float(frame_right.mean()), 1),
                "capture_time_ms": round((t_right - t0) * 1000, 1),
            }

            return {
                "left": frame_left,
                "right": frame_right,
                "timestamp": timestamp,
                "gap_ms": gap_ms,
                "metadata": metadata,
            }

        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def save_pair(self, pair, save_dir="captures"):
        """Save a stereo pair to disk."""
        os.makedirs(save_dir, exist_ok=True)

        ts = pair["timestamp"]
        left_path = os.path.join(save_dir, f"left_{ts}.png")
        right_path = os.path.join(save_dir, f"right_{ts}.png")

        left_bgr = cv2.cvtColor(pair["left"], cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(pair["right"], cv2.COLOR_RGB2BGR)

        cv2.imwrite(left_path, left_bgr)
        cv2.imwrite(right_path, right_bgr)

        # Save metadata
        meta_path = os.path.join(save_dir, f"meta_{ts}.json")
        with open(meta_path, 'w') as f:
            json.dump(pair["metadata"], f, indent=2)

        return left_path, right_path, meta_path


def capture_loop(config_path="config/stereo_config.yaml", save_dir="captures",
                 num_pairs=None, delay=1.0):
    """
    Interactive capture loop. Press Enter to capture, 'q' to quit.
    If num_pairs is set, capture that many pairs automatically.
    """
    capturer = StereoCapturer(config_path)
    capturer.initialize()

    print(f"\nStereo Capture Mode")
    print(f"Resolution: {capturer.width}x{capturer.height}")
    print(f"Save directory: {os.path.abspath(save_dir)}")
    if num_pairs:
        print(f"Target pairs: {num_pairs}")
    print("Press Enter to capture, 'q' to quit\n")

    count = 0
    try:
        while True:
            if num_pairs and count >= num_pairs:
                print(f"\nCaptured {count} pairs. Done.")
                break

            if num_pairs:
                print(f"Capturing pair {count + 1}/{num_pairs}...", end=" ", flush=True)
            else:
                user_input = input("Press Enter to capture (q to quit): ").strip()
                if user_input.lower() == 'q':
                    break
                print("Capturing...", end=" ", flush=True)

            pair = capturer.capture_pair()
            if pair is None:
                print("FAILED")
                continue

            left_path, right_path, meta_path = capturer.save_pair(pair, save_dir)
            count += 1

            print(f"OK [{count}] gap={pair['gap_ms']:.1f}ms "
                  f"L_bright={pair['metadata']['left_mean_brightness']:.0f} "
                  f"R_bright={pair['metadata']['right_mean_brightness']:.0f}")

            if num_pairs and count < num_pairs:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\nInterrupted after {count} pairs.")
    finally:
        capturer.shutdown()

    print(f"\nSaved {count} stereo pairs to {os.path.abspath(save_dir)}/")
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stereo camera capture utility")
    parser.add_argument("--config", type=str, default="config/stereo_config.yaml",
                        help="Configuration file path")
    parser.add_argument("--save-dir", type=str, default="captures",
                        help="Directory to save captures")
    parser.add_argument("--num-pairs", type=int, default=None,
                        help="Number of pairs to capture (default: interactive)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between auto captures (seconds)")
    parser.add_argument("--resolution", type=str, default=None,
                        help="Override resolution as WxH (e.g., 1536x864)")
    args = parser.parse_args()

    if args.resolution:
        w, h = args.resolution.split('x')
        # Override config - write to a temp config or modify in-memory
        import tempfile
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['capture']['width'] = int(w)
        config['capture']['height'] = int(h)
        tmp_config = os.path.join(tempfile.gettempdir(), 'stereo_config_tmp.yaml')
        with open(tmp_config, 'w') as f:
            yaml.dump(config, f)
        args.config = tmp_config

    capture_loop(args.config, args.save_dir, args.num_pairs, args.delay)


if __name__ == "__main__":
    main()
