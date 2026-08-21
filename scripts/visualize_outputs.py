#!/usr/bin/env python3
"""
Offline PLY Viewer / Comparison Tool
Since the Pi has no GUI, this script helps:
1. View PLY files on a development PC
2. Generate thumbnail previews of outputs
3. Compare multiple results

Usage on development PC:
    python visualize_outputs.py --dir output/
    python visualize_outputs.py --ply output/pointcloud.ply

Usage on Pi (headless):
    python visualize_outputs.py --dir output/ --generate-report
"""

import os
import sys
import glob
import json
import argparse

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False


def load_ply_points(ply_path):
    """Load a PLY file and return points and colors."""
    if not HAS_PLYFILE:
        print("ERROR: plyfile not installed. Run: pip3 install plyfile")
        return None, None

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    points = np.column_stack([x, y, z])

    colors = None
    if 'red' in vertices.data.dtype.names:
        r = vertices['red']
        g = vertices['green']
        b = vertices['blue']
        colors = np.column_stack([r, g, b])

    return points, colors


def create_ply_preview(ply_path, output_path, width=800, height=600):
    """Create a 2D projection preview of a PLY point cloud."""
    points, colors = load_ply_points(ply_path)
    if points is None:
        return False

    print(f"PLY: {ply_path}")
    print(f"  Points: {len(points)}")

    if len(points) == 0:
        print("  Empty point cloud")
        return False

    # Simple orthographic projection (top-down and front view)
    img_top = np.zeros((height, width, 3), dtype=np.uint8)
    img_front = np.zeros((height, width, 3), dtype=np.uint8)

    # Remove invalid points
    valid = np.all(np.isfinite(points), axis=1) & (np.abs(points).max(axis=1) < 100000)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    if len(points) == 0:
        print("  No valid points after filtering")
        return False

    # Top-down view (XZ plane)
    x_range = points[:, 0].max() - points[:, 0].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    scale_top = min((width - 40) / max(x_range, 1), (height - 40) / max(z_range, 1))

    x_off = (width - x_range * scale_top) / 2
    z_off = (height - z_range * scale_top) / 2

    for i in range(len(points)):
        px = int((points[i, 0] - points[:, 0].min()) * scale_top + x_off)
        pz = int((points[i, 2] - points[:, 2].min()) * scale_top + z_off)
        if 0 <= px < width and 0 <= pz < height:
            c = tuple(map(int, colors[i])) if colors is not None else (255, 255, 255)
            cv2.circle(img_top, (px, pz), 1, c, -1)

    cv2.putText(img_top, f"Top-down ({len(points)} pts)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Front view (XY plane, Z as depth -> color)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    scale_front = min((width - 40) / max(x_range, 1), (height - 40) / max(y_range, 1))

    x_off = (width - x_range * scale_front) / 2
    y_off = (height - y_range * scale_front) / 2

    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = max(z_max - z_min, 1)

    for i in range(len(points)):
        px = int((points[i, 0] - points[:, 0].min()) * scale_front + x_off)
        py = int((points[i, 1] - points[:, 1].min()) * scale_front + y_off)
        if 0 <= px < width and 0 <= py < height:
            # Color by depth
            depth_norm = (points[i, 2] - z_min) / z_range
            r = int(255 * depth_norm)
            b = int(255 * (1 - depth_norm))
            c = (b, 0, r)
            cv2.circle(img_front, (px, py), 1, c, -1)

    cv2.putText(img_front, f"Front view (depth-colored)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Combine
    combined = np.vstack([img_top, img_front])
    cv2.imwrite(output_path, combined)
    print(f"  Preview saved: {output_path}")
    return True


def generate_html_report(output_dir):
    """Generate an HTML report of all outputs for easy viewing."""
    html = """<!DOCTYPE html>
<html>
<head>
<title>Stereo Mapping Output Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
h1, h2 { color: #4fc3f7; }
.image-pair { display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; }
.image-pair img { max-width: 45%; border: 1px solid #444; }
.image-full img { max-width: 90%; border: 1px solid #444; margin: 10px 0; }
.info { background: #2a2a2a; padding: 10px; border-radius: 5px; margin: 10px 0; }
code { background: #333; padding: 2px 6px; border-radius: 3px; }
</style>
</head>
<body>
<h1>Stereo Mapping Output Report</h1>
"""
    # Find all output files
    images = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    plys = sorted(glob.glob(os.path.join(output_dir, "*.ply")))

    html += f"<p>Found {len(images)} images, {len(plys)} point clouds</p>\n"

    # Show paired images
    for img_path in images:
        basename = os.path.basename(img_path)
        rel_path = os.path.relpath(img_path, output_dir)
        html += f'<div class="image-full">\n'
        html += f'<h3>{basename}</h3>\n'
        html += f'<img src="{rel_path}" alt="{basename}">\n'
        html += f'</div>\n'

    # Show PLY info
    for ply_path in plys:
        basename = os.path.basename(ply_path)
        if HAS_PLYFILE:
            try:
                plydata = PlyData.read(ply_path)
                n_verts = len(plydata['vertex'].data)
                html += f'<div class="info"><h3>{basename}</h3>\n'
                html += f'<p>Points: {n_verts}</p>\n'
                html += f'</div>\n'
            except:
                html += f'<div class="info"><h3>{basename}</h3><p>(could not read)</p></div>\n'
        else:
            html += f'<div class="info"><h3>{basename}</h3><p>(install plyfile to preview)</p></div>\n'

    html += """
<h2>How to View PLY Files</h2>
<div class="info">
<p>Transfer the PLY file to a PC and view with:</p>
<ul>
<li><code>MeshLab</code> (free, cross-platform): File > Import Mesh</li>
<li><code>CloudCompare</code> (free, best for point clouds): File > Open</li>
<li><code>Blender</code> (free): File > Import > PLY</li>
<li><code>Python + Open3D</code>:</li>
</ul>
<pre><code>import open3d as o3d
pcd = o3d.io.read_point_cloud("pointcloud.ply")
o3d.visualization.draw_geometries([pcd])</code></pre>
</div>
</body></html>
"""
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"HTML report saved: {report_path}")
    return report_path


def print_ply_info(ply_path):
    """Print detailed information about a PLY file."""
    points, colors = load_ply_points(ply_path)
    if points is None:
        return

    print(f"\n{'='*50}")
    print(f"PLY File: {ply_path}")
    print(f"{'='*50}")
    print(f"Points: {len(points)}")

    valid = np.all(np.isfinite(points), axis=1)
    print(f"Valid points: {valid.sum()}")

    if valid.sum() > 0:
        pts = points[valid]
        print(f"\nBounding box:")
        print(f"  X: [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}] mm")
        print(f"  Y: [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}] mm")
        print(f"  Z: [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}] mm")
        print(f"\nDepth range: {pts[:, 2].min():.1f} - {pts[:, 2].max():.1f} mm")
        print(f"Mean depth: {pts[:, 2].mean():.1f} mm")

    if colors is not None:
        c = colors[valid]
        print(f"\nColor range:")
        print(f"  R: [{c[:, 0].min()}, {c[:, 0].max()}]")
        print(f"  G: [{c[:, 1].min()}, {c[:, 1].max()}]")
        print(f"  B: [{c[:, 2].min()}, {c[:, 2].max()}]")


def main():
    parser = argparse.ArgumentParser(description="View stereo mapping outputs")
    parser.add_argument("--dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--ply", type=str, default=None,
                        help="Specific PLY file to analyze")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate HTML report (headless Pi)")
    parser.add_argument("--preview", action="store_true",
                        help="Generate 2D preview of PLY files")
    args = parser.parse_args()

    if not HAS_OPENCV:
        print("ERROR: OpenCV required")
        sys.exit(1)

    if args.ply:
        print_ply_info(args.ply)
        if args.preview:
            preview_path = args.ply.replace('.ply', '_preview.png')
            create_ply_preview(args.ply, preview_path)
        return

    if not os.path.exists(args.dir):
        print(f"Directory not found: {args.dir}")
        sys.exit(1)

    if args.generate_report:
        report = generate_html_report(args.dir)
        print(f"\nOpen {report} in a browser to view results.")

    if args.preview:
        plys = sorted(glob.glob(os.path.join(args.dir, "*.ply")))
        for ply in plys:
            preview = ply.replace('.ply', '_preview.png')
            create_ply_preview(ply, preview)

    # Print info about all PLYs
    plys = sorted(glob.glob(os.path.join(args.dir, "*.ply")))
    for ply in plys:
        print_ply_info(ply)


if __name__ == "__main__":
    main()
