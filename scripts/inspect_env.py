#!/usr/bin/env python3
"""
Inspect Raspberry Pi Environment
Runs diagnostics to verify the Pi is ready for stereo mapping.
Run this before starting development to check hardware and software.
"""

import os
import sys
import glob
import subprocess
import platform


def run_cmd(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "(timeout)", 1
    except Exception as e:
        return str(e), 1


def check_system():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")

    # OS version
    out, _ = run_cmd("cat /etc/os-release")
    for line in out.split('\n'):
        if line.startswith(('PRETTY_NAME', 'VERSION_ID')):
            print(f"  {line}")

    # Kernel
    out, _ = run_cmd("uname -r")
    print(f"  Kernel: {out}")

    # CPU info
    out, _ = run_cmd("nproc")
    print(f"  CPU cores: {out}")

    out, _ = run_cmd("cat /proc/cpuinfo | grep 'Model' | head -1")
    print(f"  CPU model: {out}")

    # RAM
    out, _ = run_cmd("free -h | grep Mem")
    print(f"  RAM: {out}")

    # Storage
    out, _ = run_cmd("df -h / | tail -1")
    print(f"  Storage: {out}")


def check_compilers():
    print("\n" + "=" * 60)
    print("COMPILERS & BUILD TOOLS")
    print("=" * 60)

    tools = {
        "g++": "g++ --version | head -1",
        "cmake": "cmake --version | head -1",
        "make": "make --version | head -1",
        "pkg-config": "pkg-config --version",
        "git": "git --version",
    }

    for name, cmd in tools.items():
        out, rc = run_cmd(cmd)
        status = "OK" if rc == 0 else "MISSING"
        print(f"  {name:15s}: {out if rc == 0 else 'NOT FOUND'} [{status}]")


def check_opencv():
    print("\n" + "=" * 60)
    print("OPENCV")
    print("=" * 60)

    # Python OpenCV
    try:
        import cv2
        print(f"  Python OpenCV: {cv2.__version__} [OK]")
        # Check specific capabilities we need
        checks = {
            "imgcodecs (imread/imwrite)": hasattr(cv2, 'imread'),
            "calib3d (calibration)": hasattr(cv2, 'calibrateCamera'),
            "imgproc (resize/cvtColor)": hasattr(cv2, 'cvtColor'),
            "ximgproc (WLS filter)": hasattr(cv2, 'ximgproc'),
            "StereoSGBM": hasattr(cv2, 'StereoSGBM_create'),
            "ArUco": hasattr(cv2, 'aruco'),
        }
        for desc, ok in checks.items():
            print(f"    {desc}: {'available' if ok else 'NOT found'}")
    except ImportError:
        print("  Python OpenCV: NOT FOUND")

    # C++ OpenCV via pkg-config
    out, rc = run_cmd("pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null")
    if rc == 0:
        print(f"  C++ OpenCV: {out} [OK]")
    else:
        print("  C++ OpenCV: NOT FOUND (install libopencv-dev)")


def check_python():
    print("\n" + "=" * 60)
    print("PYTHON")
    print("=" * 60)

    print(f"  Python: {sys.version}")

    libs = ["numpy", "yaml", "cv2"]

    for name in libs:
        out, rc = run_cmd(f'{sys.executable} -c "import {name}; print({name}.__version__)"')
        if rc == 0:
            print(f"  {name:12s}: {out} [OK]")
        else:
            print(f"  {name:12s}: NOT FOUND")


def check_picamera2():
    print("\n" + "=" * 60)
    print("PICAMERA2 & LIBCAMERA")
    print("=" * 60)

    # Picamera2
    try:
        from picamera2 import Picamera2
        print("  Picamera2: available [OK]")

        cameras = Picamera2.global_camera_info()
        print(f"  Detected cameras: {len(cameras)}")
        for i, cam in enumerate(cameras):
            print(f"    Camera {i}:")
            for k, v in cam.items():
                print(f"      {k}: {v}")
    except ImportError:
        print("  Picamera2: NOT FOUND")
    except Exception as e:
        print(f"  Picamera2 error: {e}")

    # libcamera
    out, rc = run_cmd("libcamera-hello --version 2>/dev/null || rpicam-hello --version 2>/dev/null")
    if rc == 0:
        print(f"  libcamera/rpicam: {out} [OK]")
    else:
        print("  libcamera/rpicam: not found in PATH")

    # Camera devices
    out, rc = run_cmd("rpicam-hello --list-cameras 2>&1")
    if rc == 0 or "imx" in out.lower() or "camera" in out.lower():
        print(f"  Camera list:\n{out}")


def check_existing_projects():
    """Scan for pre-existing rover/project directories without touching anything."""
    print("\n" + "=" * 60)
    print("EXISTING PROJECTS / DIRECTORIES (read-only scan)")
    print("=" * 60)

    home = os.path.expanduser("~")

    # Directories to scan for
    search_patterns = [
        ("~/rover*", "rover*"),
        ("~/catkin_ws", "catkin_ws"),
        ("~/ros2_ws", "ros2_ws"),
        ("~/stereo_*", "stereo_*"),
        ("~/mapping*", "mapping*"),
        ("~/pi_*", "pi_*"),
        ("~/projects", "projects"),
        ("~/workspace", "workspace"),
        ("~/src", "src"),
    ]

    found_any = False
    for search_path, label in search_patterns:
        expanded = os.path.expanduser(search_path)
        matches = glob.glob(expanded)
        for match in matches:
            if not os.path.isdir(match):
                continue
            # Quick content scan
            try:
                entries = os.listdir(match)
            except PermissionError:
                entries = ["<permission denied>"]

            # Check for interesting files
            interesting = []
            for e in entries:
                el = e.lower()
                if el.endswith(('.py', '.cpp', '.h', '.cmake', '.yaml', '.json',
                                '.launch', '.xml', '.urdf', '.xacro', '.msg',
                                '.srv', '.action')):
                    interesting.append(e)

            rel = os.path.relpath(match, home)
            print(f"\n  {rel}/")
            print(f"    Total entries: {len(entries)}")
            if interesting:
                print(f"    Code/config files: {len(interesting)}")
                for f in interesting[:10]:
                    print(f"      {f}")
                if len(interesting) > 10:
                    print(f"      ... and {len(interesting) - 10} more")
            else:
                print(f"    No code/config files detected")
            found_any = True

    # Also check current working directory
    cwd = os.getcwd()
    # Don't scan our own stereo_mapping dir
    cwd_name = os.path.basename(cwd)
    if cwd_name != "stereo_mapping":
        try:
            entries = os.listdir(cwd)
            code_files = [e for e in entries if e.lower().endswith(
                ('.py', '.cpp', '.h', '.cmake', '.yaml', '.json'))]
            if code_files:
                print(f"\n  CWD ({cwd}):")
                print(f"    Code/config files: {len(code_files)}")
                for f in code_files[:10]:
                    print(f"      {f}")
                found_any = True
        except PermissionError:
            pass

    # Check for ROS workspaces specifically
    print("\n  ROS workspace check:")
    for ws_name in ["catkin_ws", "ros2_ws", "colcon_ws"]:
        ws_path = os.path.join(home, ws_name)
        if os.path.isdir(ws_path):
            src_dir = os.path.join(ws_path, "src")
            if os.path.isdir(src_dir):
                pkgs = os.listdir(src_dir)
                print(f"    {ws_name}/src contains: {len(pkgs)} entries")
                for p in pkgs[:10]:
                    print(f"      {p}")
            else:
                print(f"    {ws_name} exists but no src/ directory")
        else:
            print(f"    {ws_name}: not found")

    if not found_any:
        print("  No pre-existing project directories found.")
    print()
    print("  NOTE: This is a read-only scan. Nothing was modified.")


def check_performance():
    print("\n" + "=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    # CPU frequency
    out, _ = run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null")
    if out.isdigit():
        print(f"  CPU freq: {int(out)/1000:.0f} MHz")

    # Temperature
    out, _ = run_cmd("vcgencmd measure_temp 2>/dev/null")
    if out:
        print(f"  Temperature: {out}")

    # GPU memory
    out, _ = run_cmd("vcgencmd get_mem gpu 2>/dev/null")
    if out:
        print(f"  GPU mem: {out}")

    # Camera ports
    out, _ = run_cmd("vcgencmd get_camera 2>/dev/null")
    if out and "detected" in out:
        print(f"  Camera status: {out}")


def main():
    print("\n" + "#" * 60)
    print("#  Raspberry Pi Environment Inspection")
    print("#  Stereo Mapping Project")
    print("#" * 60 + "\n")

    check_system()
    check_compilers()
    check_opencv()
    check_python()
    check_picamera2()
    check_existing_projects()
    check_performance()

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print("\nIf any critical items show MISSING, install them with:")
    print("  sudo apt update && sudo apt install <package>")


if __name__ == "__main__":
    main()
