#!/usr/bin/env python3
"""
Generate ChArUco calibration board for printing.
Must match config/stereo_config.yaml calibration parameters.

Usage:
    python3 generate_charuco_board.py                     # Print to file
    python3 generate_charuco_board.py --output board.png  # Custom output
    python3 generate_charuco_board.py --dpi 300            # Print-resolution
"""

import argparse
import cv2
import cv2.aruco as aruco
import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate ChArUco calibration board")
    parser.add_argument("--config", type=str, default="config/stereo_config.yaml")
    parser.add_argument("--output", type=str, default="charuco_target.png")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for print resolution")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config['calibration']['board_type'] != 'charuco':
        print("Config uses checkerboard, not ChArUco. Showing checkerboard instead.")
        squares_x = config['calibration']['checkerboard']['corners_x'] + 1
        squares_y = config['calibration']['checkerboard']['corners_y'] + 1
        square_mm = config['calibration']['checkerboard']['square_size_mm']

        # Convert mm to pixels
        px_per_mm = args.dpi / 25.4
        w = int(squares_x * square_mm * px_per_mm)
        h = int(squares_y * square_mm * px_per_mm)

        img = cv2.cvtColor(
            cv2.imread(cv2.data.haarcascades + "smartphone.png", 0),
            cv2.COLOR_GRAY2BGR
        ) if False else None

        # Create simple checkerboard
        cell_w = w // squares_x
        cell_h = h // squares_y
        board = 255 * np.ones((squares_y * cell_h, squares_x * cell_w), dtype=np.uint8)
        for r in range(squares_y):
            for c in range(squares_x):
                if (r + c) % 2 == 0:
                    board[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = 0

        cv2.imwrite(args.output, board)
        print(f"Checkerboard saved: {args.output}")
        print(f"  Size: {squares_x}x{squares_y} squares, {square_mm}mm each")
        print(f"  Print size: {squares_x * square_mm:.0f} x {squares_y * square_mm:.0f} mm")
        return

    # ChArUco board
    board_cfg = config['calibration']['charuco']
    squares_x = board_cfg['squares_x']
    squares_y = board_cfg['squares_y']
    square_size = board_cfg['square_size_mm']
    marker_size = board_cfg['marker_size_mm']
    dict_name = board_cfg['dictionary']

    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size,
        marker_size,
        aruco_dict
    )

    px_per_mm = args.dpi / 25.4
    img_w = int(squares_x * square_size * px_per_mm)
    img_h = int(squares_y * square_size * px_per_mm)

    img = board.generateImage((img_w, img_h))
    cv2.imwrite(args.output, img)

    print(f"ChArUco board saved: {args.output}")
    print(f"  Board: {squares_x}x{squares_y} squares")
    print(f"  Square size: {square_size}mm")
    print(f"  Marker size: {marker_size}mm")
    print(f"  Dictionary: {dict_name}")
    print(f"  Image size: {img_w}x{img_h} px")
    print(f"  Print size: {squares_x * square_size:.0f} x {squares_y * square_size:.0f} mm")
    print(f"  DPI: {args.dpi}")
    print()
    print("IMPORTANT: Print at actual size (no scaling). Measure square size with")
    print("calipers and update config if it differs from the configured value.")


if __name__ == "__main__":
    import numpy as np
    main()
