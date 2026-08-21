# Stereo Mapping Project - Full Context

## Hardware
- Raspberry Pi 5 Model B Rev 1.0, 4GB RAM, aarch64
- OS: Debian 13 (trixie), kernel 6.18.34+rpt-rpi-2712
- 2x Sony IMX708 CSI cameras, ~85mm baseline
- Camera 0 = LEFT, Camera 1 = RIGHT (verified via hand test)
- No monitor on Pi — SSH only (antriksh@192.168.137.126)

## Software Already Installed on Pi (NO NEED to reinstall)
- g++ 14.2.0, cmake 3.31.6, make 4.4.1
- OpenCV 4.10.0 (Python + C++), all modules OK (calib3d, ximgproc, StereoSGBM, ArUco)
- Python 3.13.5, numpy 2.2.4, PyYAML 6.0.2
- Picamera2, libcamera v0.7.2
- setup_env.sh is NOT needed — everything already installed

## Pi Directory Structure
```
~/stereo_mapping/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── setup_env.sh          # NOT NEEDED, skip it
├── build.sh
├── config/
│   ├── stereo_config.yaml  # master config (square_size_mm=28.0, NOT 30.0)
│   └── calibration/
│       └── stereo_calib.yaml  # calibration output (may be stale/incomplete)
├── src/
│   ├── main.cpp
│   └── stereo_pipeline.cpp
├── include/
│   └── stereo_pipeline.h
├── scripts/
│   ├── inspect_env.py           # env check (fixed for Python 3.13)
│   ├── camera_verify.py         # camera detection + overlap test
│   ├── stereo_live_view.py      # MJPEG web stream + capture button
│   ├── calibrate_stereo.py      # ChArUco calibration (has numpy YAML fix)
│   ├── capture_stereo.py        # stereo pair capture
│   ├── run_pipeline.py          # full pipeline runner
│   ├── generate_charuco_board.py # printable board generator
│   └── visualize_outputs.py     # PLY viewer + HTML report
├── captures/
│   └── calibration/
│       ├── left/       # cal_*.png for calibration (currently EMPTY)
│       └── right/      # cal_*.png for calibration (currently EMPTY)
├── calibration/        # live view saves calib_NNNN_left.png here (currently EMPTY)
├── build/
└── output/
```

## ChArUco Board
- Board PNG generated at: `~/stereo_mapping/charuco_target.png`
- 5x7 squares, DICT_6X6_250
- Screen square size: **28mm** (measured with ruler from laptop screen)
- Marker size: 20.5mm
- This was updated in config from original 30mm

## Current Status: WORKING — depth maps confirmed good on textured scenes
- DIRECTION DECISION (user): multi-frame stitching / pose estimation (Stage 9) SKIPPED for now. Project mode = single-shot scanning: one capture -> one PLY from one viewpoint. Nothing more to build for scan mode; workflow = dashboard CAPTURE -> pipeline --save-all -> output/pointcloud.ply.
- Single-frame accuracy gate: STILL UNVERIFIED. Attempt 1 ran on room scene without tape value (invalid). Attempt 2: tape=440mm -> check FAIL +193% but TEST WAS INVALID: 440mm < 658mm rig minimum (impossible depth) AND scene was cluttered room not wall-filled (Z span 721-4999mm, plane tilt 84.5deg). Tool behaved correctly. REDO at ~1000mm: textured target filling frame, rig perpendicular, tape to LEFT lens front; sanity signal first = Depth Statistics Z mean within +-30mm of tape and small Z span.

## Git
- Repo: https://github.com/ShreeshantXD/stereomappingrover.git (branch main). Local Windows mirror initialized as git repo 2026-08-21, initial commit 36d6a66 on top of remote LICENSE commit 26c5ac0.
- SECURITY: Pi SSH password was in CONTEXT.md - REDACTED to "(redacted for public repo)" before push. Real password NOT in repo (user knows it).
- .gitignore excludes build/, captures/, output/, calibration/ - only source/scripts/config/docs are tracked.

### Depth-quality investigation findings
- Capture #31 (smooth metal scale, <658mm): NO math bug. Verified Z_calc==Z_stored on sample grid; /16 scaling consistent; only legit max_depth clip fired. SGBM SATURATED at max disparity (raw +4068..4076 = 254-255px ceiling) over textureless/too-close region -> flat blob. Old depth colormap rendered near=blue AND invalid=blue -> looked contradictory vs disparity map. FIXED: near=red, far=blue, invalid=black.
- Robustness added: texture mask (mean Sobel magnitude < depth.texture_min_gradient=8.0 -> invalid, prints masked %), per-run depth stats (valid %, Z min/max/mean), raw disparity sample grid with per-pixel status (INVALID(disp<=0)/CLIPPED(min_depth)/CLIPPED(max_depth)/MASKED(texture)/valid). WLS + LR check (disp12_max_diff=1) confirmed already enabled.
- Capture #32 (textured room scene): valid 38.7%, mean Z 1309mm, NO saturation blob (disparities vary smoothly 28-252px), far wall >5m correctly CLIPPED(max_depth). Mask removed 48.9% - if too aggressive lower texture_min_gradient to ~5.
- Pending: PLY preview (apt python3-plyfile doesn't exist in trixie; pip --break-system-packages slow); realtime perf (disparity 5.7s = bottleneck; needs lower res or fewer disparities later).
- rover_backend process holds cameras when running - pkill -f rover_backend before live view.
- Dashboard PROCESS button (stereo_live_view.py /process): captures pair, runs C++ pipeline, shows raw L/R + disparity + depth + stats in browser.
- Validity masking: REVERTED to gradient-only. WLS confidence-map masking (v1 CV_16U reinterpret, v2 CV_32F + stats/sweep, v3 MORPH_CLOSE+HoleDiag) was removed entirely after block-hole artifacts resisted two fix rounds (thr 0.1->0.05 had zero visible effect). Current state = the confirmed-good wall+switchboard behavior: Sobel texture mask only, depth.texture_min_gradient=8.0. Kept learnings if ever revisited: getConfidenceMap() returns CV_32FC1 (never reinterpret bytes); conf is HIGHER at half res (38% vs 23.5% above thr at downscale 2); block holes have threshold-insensitive confidence; full SGBM vs 3WAY mode test exposed create() arg-order bug.
- main.cpp --probe x1,y1,... prints raw disparity/Z/status at arbitrary pixels (used for glossy-surface diagnosis, e.g. charger plug specular mismatch).
- MAJOR BUG FIXED: initStereoMatcher passed StereoSGBM::create args in wrong order post-disp12MaxDiff (preFilterCap got 10, uniquenessRatio got 100, speckle params scrambled, mode NEVER applied - always default MODE_SGBM). All prior disparity quality analysis ran under wrong params. Mode test (SGBM vs 3WAY identical results) exposed it.
- Charger anomaly CONFIRMED as specular mismatch: plug reads 820.7mm vs wall 812.0mm behind it; both pixels flagged by confidence mask.
- Pi measurements: OpenCV has NEON+DOTPROD dispatch, TBB parallel framework, but pipeline runs ~100% single core during disparity (5.7s). num_disparities=128 disqualified (scene at 660-1100mm needs 150-255px; 128 caps at min depth 1310mm -> 0.4% valid).
- Confidence threshold sweep (superseded by revert): plateau ~29% of pixels at 1k-8k, cliff after; 32768 too strict (7.6% valid), 8192 chosen, later 0.05 float-domain. All removed.
- Mode verdict: SGBM_3WAY stays (full SGBM +33% time for +0.26pp valid). WLS costs ~+1500ms (disparity alone 2261ms). Pipeline runs single-core despite TBB/NEON present.
- Speed fixes added: left/right matchers run concurrently (std::thread); --downscale N flag computes disparity at 1/N res then upscales xN (bonus: effective min depth /N, e.g. 2 -> ~330mm).
- STRIPE BUG ROOT-CAUSED+FIXED: getConfidenceMap() returns CV_32FC1 (not CV_16S as older docs say). Old code reinterpreted its bytes as CV_16U -> float bits split into 2x uint16 streams -> alternating valid/black rows in depth mask. Fixed: use map directly as CV_32F.
- Confidence mask v3: REVERTED (see Validity masking note above). Was: stats + float sweep + MORPH_CLOSE 5x5 + HoleDiag; block holes persisted at every threshold.
- Downscale-2 bonus: confidence HIGHER at half res (38% vs 23.5% above thr) - half-res matching more robust per block, not just faster.
- Single-frame accuracy validation (pre-stitching gate): main.cpp prints Point Cloud Statistics every run (count, XYZ min/max/mean/std, PCA dominant-plane normal/tilt/RMS roughness, DEGENERATE flags: <1000 pts / collapsed stds / zero Z range). scripts/measure_ply.py (numpy-only, ascii PLY): stats | check <ply> <tape_mm> (median Z vs tape, error mm+%, PASS/MARGINAL/FAIL vs max(5mm,1%) tol) | plane | dist X1 Y1 Z1 X2 Y2 Z2 --r (3D distance between snapped neighbourhoods). Verified on synthetic 1000mm wall + 600mm baseline-distance cloud. NOTE: config downsample_voxel_mm is UNUSED by C++ generatePointCloud (only max_points striding) - PLY is faithful valid-pixel sample. Theory sigma_Z = Z^2*sigma_d/(f*B), f*B=167785: ~0.9mm @700mm, 1.8mm @1m, 4mm @1.5m (sigma_d=0.3px); calibration systematic ~1% dominates. Tape measured to LEFT lens front, rig perpendicular, textured target, never closer than 660mm (255px disparity ceiling; 330mm with --downscale 2).

### TF-Luna 1D LiDAR integration — IN PROGRESS, no data yet
- Hardware: Benewake TF-Luna (0.2-8m, 6-pin JST), purpose: (1) automated ground-truth for stereo accuracy check (replaces tape measure - point cameras+Luna at same wall, compare PLY median Z vs Luna distance), (2) rover obstacle ranging on textureless/glossy surfaces where stereo fails. Default protocol UART 115200, 9-byte frames: 0x54 0x59 distL distH signalL signalH tempL tempH checksum.
- User-reported wiring (physical pin numbers): Luna RX -> Pi pin7 (GPIO4), Luna TX -> Pi pin29 (GPIO5), GND -> pin6, VCC -> pin2 (5V rail; TF-Luna accepts 5V or 3.3V), mystery "i2c enable" wire -> Pi pin14 (GPIO10). TX/RX orientation as described is CORRECT cross-connection IF GPIO4=TX/GPIO5=RX.
- Pi 5 serial facts discovered: /dev/serial0 -> ttyAMA10 = GPIO14/15 header UART, console/getty OFF so port free. I2C bus has ONE device at 0x68 (RTC or IMU - NOT the Luna; Luna I2C would be 0x10). No USB-serial adapters.
- `sudo dtoverlay uart2` runtime-loaded OK and created /dev/ttyAMA2 (NOTE: root-only perms crw------- root root, needs sudo to read; NOTE: runtime overlay is LOST ON REBOOT - persist with `echo "dtoverlay=uart2" | sudo tee -a /boot/firmware/config.txt` once sensor works).
- STATUS: ZERO bytes received on both ttyAMA10 and ttyAMA2 at 115200 (sudo stty -F ... raw -echo then sudo timeout 3 od -A x -t x1 ...). Sensor silent.
- NEXT DEBUG STEPS (in order): (1) `pinctrl get 4; pinctrl get 5; sudo dtoverlay -l` - verify uart2 actually claimed GPIO4/5 (Pi5 RP1 uart-to-pin mapping may differ from Pi4). (2) Physically SWAP the two signal wires (pin7<->pin29) and retry od - TX/RX mislabel is most common cause of silence. (3) Clarify mystery i2c-enable wire: bare TF-Luna has NO such pin (pads: 5V, 3.3V, SDA/TX, GND, RX/SCL, Signal) - read silkscreen labels; if sensor got strapped into I2C mode its UART stays silent. (4) Verify power (any warmth/voltage at Luna VCC-GND).
- Once data flows: write scripts/tf_luna.py (open /dev/ttyAMA2 115200, parse 9-byte frames w/ checksum, print dist mm + signal strength), then a stereo-vs-luna cross-check (compare measure_ply.py median Z against Luna reading at same target).

### History
1. **Calibration** ✅ DONE (18 valid pairs, --min-corners 14):
   - Stereo error: 1.16 px | Baseline: 77.86mm (vs 79mm measured lens-center) | f=2154.9px
   - NOTE: PyYAML dumps keys alphabetically (board_type, camera_left, camera_right, ..., resolution last) — file IS complete
   - Raw K fx≈1720 vs rectified P1 fx=2155 are BOTH correct (different quantities)
2. **C++ pipeline** ✅ BUILDS AND RUNS (after fixing: ximgproc include, fake RightDeviceInfoMatcher class, 3-arg compute(), cv::CV_8UC1 scoping, vector<vector<double>> Mat conversion via toMat(), mkdirs→createDirectories, nested dist/T lists)
3. **First end-to-end run** ✅ cal_1 pair: 354,597 points, total 6.66s
   - Timing: rectify 22ms | disparity 5990ms | depth 19ms | pointcloud 59ms
   - Disparity dominates (SGBM_3WAY, 256 disp @ 1536x864). For realtime: lower res or fewer disparities.
4. **Capture naming**: live view saves calib_0001.. ; reorganize strips padding -> captures/calibration/{left,right}/cal_1.png .. cal_32.png

## IMPORTANT: Disparity Range Issue (fix when tuning pipeline)
- f*B = 2154.9 * 77.86 ≈ 167,800 px·mm
- With num_disparities=128 → min measurable depth ≈ 1310mm (config's min_depth_mm=200 unreachable!)
- For rover range 0.5-5m: set num_disparities=256 (min ~656mm) or 384 (min ~437mm) in stereo_config.yaml AND C++ defaults

## Known Bugs Fixed
- main.cpp Point Cloud Statistics: cv::meanStdDev on Nx3 single-channel Mat returns ONE value (per-channel, not per-column) -> Y/Z printed mean=0 std=0. Fixed: meanStdDev per pts.col(a) loop.
- OpenCV 4 removed old enum names: cv::PCA_DATA_AS_ROW -> cv::PCA::DATA_AS_ROW.
- `inspect_env.py`: `eval()` can't run `import` on Python 3.13 → switched to `subprocess`
- `inspect_env.py`: OpenCV module check `hasattr(cv2, 'core')` wrong → check specific functions
- `stereo_live_view.py`: `global CAPTURE_DIR` SyntaxError on Python 3.13 → removed globals
- `stereo_live_view.py`: `HTTPServer` single-threaded blocks POST during stream → `ThreadedHTTPServer`
- `stereo_live_view.py`: `create_still_configuration` slow → `create_video_configuration`
- `stereo_live_view.py`: /capture saved 768x432 stream frames → now calls grab_full_pair() (1536x864)
- `stereo_live_view.py`: Ctrl+C deadlocked (server.shutdown() from serve_forever thread) → os._exit(0)
- `stereo_live_view.py`: laggy stream (sequential captures, re-encoding unchanged frames) → parallel threads + dup-skip + fps cap + TCP_NODELAY
- `calibrate_stereo.py`: numpy types crash yaml.dump → NumpyDumper custom representer
- YAML stored 50MB of rectification maps → removed; recomputed at load (Python validate/run_pipeline already did; C++ loadCalibration now computes them)

## What To Do Next (STEPS)
1. Re-run calibration with slim YAML: `python3 scripts/calibrate_stereo.py calibrate --min-corners 14 && python3 scripts/calibrate_stereo.py validate` (fast now)
2. Check output/calibration/rectified_epipolar.png — features on same horizontal lines in both halves
3. Build C++ pipeline: `chmod +x build.sh && ./build.sh`
4. Update num_disparities 128→256 in stereo_config.yaml for closer range
5. End-to-end test: process a capture pair, check disparity/depth/PLY outputs

## Reorganize Command (run after capturing)
```bash
cd ~/stereo_mapping/captures/calibration && rm -rf left right && mkdir -p left right
for f in ~/stereo_mapping/calibration/calib_*_left.png; do
  idx=$(basename "$f" | sed 's/calib_0*\([0-9]*\)_left.png/\1/')
  cp "$f" "left/cal_${idx}.png"
done
for f in ~/stereo_mapping/calibration/calib_*_right.png; do
  idx=$(basename "$f" | sed 's/calib_0*\([0-9]*\)_right.png/\1/')
  cp "$f" "right/cal_${idx}.png"
done
echo "Done: $(ls left/ | wc -l) pairs"
```

## Calibration Capture Tips
- Both cameras MUST see the full board in every capture
- Vary positions: near, far, tilted, center, corners of frame
- Hold screen steady when clicking CAPTURE
- Board fills ~40-80% of frame
- 30 captures minimum, more is better

## Key Config Values (stereo_config.yaml)
- Resolution: 1536x864
- Square size: 28.0mm (screen-measured)
- Marker size: 20.5mm
- Board: 5x7 ChArUco, DICT_6X6_250
- SGBM: num_disparities=128, block_size=7, WLS filter enabled
- Depth range: 200mm - 5000mm
- Output: PLY format, voxel downsample 5mm

## SSH Access
- Host: 192.168.137.126
- User: antriksh
- Password: (redacted for public repo)
- SSH only, no monitor — user runs commands I provide

## Important Notes
- DO NOT run setup_env.sh — everything is already installed
- DO NOT modify files on Pi without asking — user wants to be asked before system changes
- Camera 0 = LEFT, Camera 1 = RIGHT (verified by hand test)
- Physical baseline ~85mm (measured with calipers)
- Screen calibration: square size is 28mm (measured with ruler), NOT 30mm
- Python 3.13 on this Pi — watch for eval() and yaml.dump() issues
