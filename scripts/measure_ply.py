#!/usr/bin/env python3
"""Single-frame PLY accuracy verification (no stitching, no pose logic).

All coordinates are millimetres in the rectified LEFT camera frame:
  X = right, Y = down, Z = forward along the optical axis.
A tape-measure distance "wall to lens" therefore corresponds to Z.

Commands:
  stats <ply>                          point count, XYZ min/max/mean/std, health flags
  check <ply> EXPECTED_MM              compare robust depth against a tape measurement
  plane <ply>                          plane fit: normal, tilt vs camera Z, RMS roughness
  dist <ply> X1 Y1 Z1 X2 Y2 Z2         3D distance between two point neighbourhoods
                                       (--r snap radius, default 100 mm)

Examples:
  python3 scripts/measure_ply.py stats output/pointcloud.ply
  python3 scripts/measure_ply.py check output/pointcloud.ply 1000
  python3 scripts/measure_ply.py plane output/pointcloud.ply
  python3 scripts/measure_ply.py dist output/pointcloud.ply -300 0 1000 300 0 1000 --r 80
"""
import argparse
import sys

import numpy as np


def load_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            if not line:
                sys.exit(f"ERROR: {path}: unterminated PLY header")
            header.append(line)
            if line == "end_header":
                break
        fmt = next((l for l in header if l.startswith("format")), "")
        if "ascii" not in fmt:
            sys.exit(f"ERROR: {path}: only ascii PLY supported "
                     f"(got: {fmt})")
        nv_line = next((l for l in header if l.startswith("element vertex")), None)
        if nv_line is None:
            sys.exit(f"ERROR: {path}: no vertex element")
        nv = int(nv_line.split()[-1])
        props = [l.split()[-1] for l in header if l.startswith("property")]

    data = np.loadtxt(path, skiprows=len(header), max_rows=nv, ndmin=2)
    if data.shape[0] != nv:
        sys.exit(f"ERROR: {path}: header says {nv} points, file has {data.shape[0]}")
    return {p: data[:, i] for i, p in enumerate(props)}


def xyz(ply):
    for ax in ("x", "y", "z"):
        if ax not in ply:
            sys.exit(f"ERROR: PLY missing property '{ax}'")
    return np.column_stack([ply["x"], ply["y"], ply["z"]])


def cmd_stats(path):
    pts = xyz(load_ply(path))
    n = len(pts)
    print(f"=== PLY Stats: {path} ===")
    print(f"  Points: {n}")
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    for i, ax in enumerate("XYZ"):
        print(f"  {ax}: min={lo[i]:9.1f} max={hi[i]:9.1f} span={hi[i]-lo[i]:9.1f} "
              f"mean={mean[i]:9.1f} std={std[i]:7.2f} mm")

    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]
    rms = float(np.sqrt(max(evals[0], 0.0)))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    resid = (pts - pts.mean(axis=0)) @ normal
    within5 = 100.0 * np.count_nonzero(np.abs(resid) < 5.0) / n
    print(f"  Dominant plane: normal=({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}) "
          f"tilt={tilt:.1f} deg from camera Z")
    print(f"  Plane RMS roughness: {rms:.2f} mm | {within5:.1f}% of points within 5mm of plane")

    flags = []
    if n < 1000:
        flags.append("too few points (<1000)")
    if np.all(std < 0.5):
        flags.append("collapsed cloud (all axis stds < 0.5mm)")
    if hi[2] - lo[2] < 1.0:
        flags.append("zero Z range")
    if flags:
        print(f"  Flags: DEGENERATE -> {'; '.join(flags)}")
        sys.exit(1)
    print("  Flags: none (cloud looks healthy)")


def cmd_check(path, expected_mm):
    z = xyz(load_ply(path))[:, 2]
    med = float(np.median(z))
    q1, q3 = np.percentile(z, [25, 75])
    err = med - expected_mm
    pct = 100.0 * err / expected_mm
    print(f"=== Depth Check: {path} ===")
    print(f"  Points used: {len(z)}")
    print(f"  Robust depth (median): {med:.1f} mm  (IQR {q1:.1f} .. {q3:.1f})")
    print(f"  Tape measurement:      {expected_mm:.1f} mm")
    print(f"  Error: {err:+.1f} mm ({pct:+.2f}%)")

    tol_mm = max(5.0, 0.01 * expected_mm)
    inside = np.count_nonzero(np.abs(z - expected_mm) <= tol_mm)
    print(f"  {100.0 * inside / len(z):.1f}% of points within +/-{tol_mm:.1f} mm of tape value")
    if abs(err) <= tol_mm:
        print(f"  VERDICT: PASS (|error| <= {tol_mm:.1f} mm tolerance)")
    elif abs(err) <= 2 * tol_mm:
        print(f"  VERDICT: MARGINAL (error {abs(err):.1f} mm > {tol_mm:.1f} mm tolerance)")
    else:
        print(f"  VERDICT: FAIL (error {abs(err):.1f} mm >> {tol_mm:.1f} mm tolerance)")
        sys.exit(1)


def cmd_plane(path):
    cmd_stats(path)


def cmd_dist(path, seed_a, seed_b, radius):
    pts = xyz(load_ply(path))

    def snap(seed, label):
        d = np.linalg.norm(pts - np.array(seed, dtype=np.float64), axis=1)
        sel = pts[d < radius]
        if len(sel) < 20:
            sys.exit(f"ERROR: only {len(sel)} points within {radius}mm of {label} "
                     f"{seed} - wrong seed or radius too small")
        med = np.median(sel, axis=0)
        spread = np.std(sel[:, 2])
        print(f"  {label}: snapped to ({med[0]:.1f}, {med[1]:.1f}, {med[2]:.1f}) mm "
              f"[{len(sel)} pts, Z std {spread:.2f} mm]")
        return med

    print(f"=== Distance: {path} ===")
    a = snap(seed_a, "A")
    b = snap(seed_b, "B")
    delta = b - a
    dist = float(np.linalg.norm(delta))
    print(f"  dx={delta[0]:+.1f} dy={delta[1]:+.1f} dz={delta[2]:+.1f} mm")
    print(f"  3D distance A->B: {dist:.1f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="point count, XYZ stats, health flags")
    p.add_argument("ply")

    p = sub.add_parser("check", help="compare median depth against tape measurement")
    p.add_argument("ply")
    p.add_argument("expected_mm", type=float)

    p = sub.add_parser("plane", help="plane fit summary (same as stats)")
    p.add_argument("ply")

    p = sub.add_parser("dist", help="3D distance between two point neighbourhoods")
    p.add_argument("ply")
    p.add_argument("a", type=float, nargs=3, metavar="AX AY AZ")
    p.add_argument("b", type=float, nargs=3, metavar="BX BY BZ")
    p.add_argument("--r", type=float, default=100.0,
                   help="snap radius in mm (default 100)")

    args = ap.parse_args()
    if args.cmd == "stats":
        cmd_stats(args.ply)
    elif args.cmd == "check":
        cmd_check(args.ply, args.expected_mm)
    elif args.cmd == "plane":
        cmd_plane(args.ply)
    elif args.cmd == "dist":
        cmd_dist(args.ply, args.a, args.b, args.r)


if __name__ == "__main__":
    main()
