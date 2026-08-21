/*
 * Stereo Pipeline Main
 * Reads a stereo image pair, runs full rectification -> disparity -> depth -> PLY pipeline.
 *
 * Usage:
 *   ./stereo_pipeline --left <left.png> --right <right.png> [--config <config.yaml>] [--calibration <calib.yaml>]
 *                     [--probe x1,y1,x2,y2,...]
 *
 * This is a standalone offline processor. Use stereo_capture.py to acquire images.
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "stereo_pipeline.h"

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --left <path>          Left stereo image\n"
              << "  --right <path>         Right stereo image\n"
              << "  --calibration <path>   Calibration file (YAML)\n"
              << "  --config <path>        Config file (YAML, default: config/stereo_config.yaml)\n"
              << "  --output-dir <path>    Output directory (default: output)\n"
              << "  --probe x1,y1,...      Print raw disparity/Z/status at given pixels\n"
              << "  --downscale N          Match disparity at 1/N resolution (default: 1)\n"
              << "  --save-rectified       Save rectified images\n"
              << "  --save-disparity       Save disparity image\n"
              << "  --save-depth           Save depth image\n"
              << "  --save-ply             Save point cloud as PLY\n"
              << "  --save-all             Save all outputs\n";
}

static void printProbeRow(int x, int y, const cv::Mat& disparity, const cv::Mat& depth,
                          const stereo::DepthParams& dp) {
    short raw = disparity.at<short>(y, x);
    double d = raw / 16.0;
    double z_calc = d > 0 ? dp.focal_length_px * dp.baseline_mm / d : 0.0;
    double z_stored = depth.at<double>(y, x);
    std::string status;
    if (raw <= 0) {
        status = "INVALID(disp<=0)";
    } else if (z_calc < dp.min_depth_mm) {
        status = "CLIPPED(min_depth)";
    } else if (z_calc > dp.max_depth_mm) {
        status = "CLIPPED(max_depth)";
    } else if (z_stored <= 0) {
        status = "MASKED(texture)";
    } else {
        status = "valid";
    }
    printf("  %4d %4d | %+7d | %7.2f | %9.1f | %11.1f | %s\n",
           x, y, raw, d, z_calc, z_stored, status.c_str());
}

int main(int argc, char** argv) {
    std::string left_path, right_path;
    std::string calib_path = "config/calibration/stereo_calib.yaml";
    std::string config_path = "config/stereo_config.yaml";
    std::string output_dir = "output";
    std::string probe_spec;
    int downscale = 1;
    bool save_rectified = false;
    bool save_disparity = false;
    bool save_depth = false;
    bool save_ply = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--left" && i + 1 < argc) left_path = argv[++i];
        else if (arg == "--right" && i + 1 < argc) right_path = argv[++i];
        else if (arg == "--calibration" && i + 1 < argc) calib_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--probe" && i + 1 < argc) probe_spec = argv[++i];
        else if (arg == "--downscale" && i + 1 < argc) downscale = std::stoi(argv[++i]);
        else if (arg == "--save-rectified") save_rectified = true;
        else if (arg == "--save-disparity") save_disparity = true;
        else if (arg == "--save-depth") save_depth = true;
        else if (arg == "--save-ply") save_ply = true;
        else if (arg == "--save-all") {
            save_rectified = true;
            save_disparity = true;
            save_depth = true;
            save_ply = true;
        }
        else {
            printUsage(argv[0]);
            return 1;
        }
    }

    if (left_path.empty() || right_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    // Create output directory
    cv::utils::fs::createDirectories(output_dir);

    std::cout << "=== Stereo Pipeline ===" << std::endl;
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Left: " << left_path << std::endl;
    std::cout << "Right: " << right_path << std::endl;
    std::cout << "Calibration: " << calib_path << std::endl;
    std::cout << std::endl;

    // Load stereo pipeline
    stereo::StereoPipeline pipeline;

    if (!pipeline.loadConfig(config_path)) {
        std::cerr << "ERROR: Failed to load config" << std::endl;
        return 1;
    }

    if (!pipeline.loadCalibration(calib_path)) {
        std::cerr << "ERROR: Failed to load calibration" << std::endl;
        return 1;
    }

    if (downscale > 1) {
        pipeline.setDownscale(downscale);
        std::cout << "Downscale: disparity computed at 1/" << downscale
                  << " resolution" << std::endl;
    }

    // Load images
    cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);

    if (left.empty() || right.empty()) {
        std::cerr << "ERROR: Failed to load images" << std::endl;
        return 1;
    }

    std::cout << "Left image: " << left.cols << "x" << left.rows << std::endl;
    std::cout << "Right image: " << right.cols << "x" << right.rows << std::endl;

    // Step 1: Rectify
    auto t_start = std::chrono::high_resolution_clock::now();

    std::cout << "\n[1/4] Rectifying..." << std::endl;
    cv::Mat rect_left, rect_right;
    pipeline.rectify(left, right, rect_left, rect_right);

    auto t_rectify = std::chrono::high_resolution_clock::now();
    double ms_rectify = std::chrono::duration<double, std::milli>(t_rectify - t_start).count();
    std::cout << "  Done in " << ms_rectify << " ms" << std::endl;

    if (save_rectified) {
        cv::imwrite(output_dir + "/rectified_left.png", rect_left);
        cv::imwrite(output_dir + "/rectified_right.png", rect_right);

        // Epipolar line visualization
        int h = rect_left.rows;
        int w = rect_left.cols;
        cv::Mat sidebyside(h, w * 2, CV_8UC3);
        rect_left.copyTo(sidebyside(cv::Rect(0, 0, w, h)));
        rect_right.copyTo(sidebyside(cv::Rect(w, 0, w, h)));
        for (int y = 0; y < h; y += 40) {
            cv::line(sidebyside, cv::Point(0, y), cv::Point(w * 2, y),
                     cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite(output_dir + "/rectified_epipolar.png", sidebyside);
        std::cout << "  Saved rectified images" << std::endl;
    }

    // Step 2: Disparity
    std::cout << "\n[2/4] Computing disparity..." << std::endl;
    cv::Mat disparity, disparity_vis;
    pipeline.computeDisparity(rect_left, rect_right, disparity, disparity_vis);

    auto t_disp = std::chrono::high_resolution_clock::now();
    double ms_disp = std::chrono::duration<double, std::milli>(t_disp - t_rectify).count();
    std::cout << "  Done in " << ms_disp << " ms" << std::endl;

    if (save_disparity) {
        pipeline.saveDisparityImage(output_dir + "/disparity.png", disparity_vis);
    }

    // Step 3: Depth
    std::cout << "\n[3/4] Computing depth..." << std::endl;
    cv::Mat depth;
    pipeline.computeDepth(disparity, depth);

    auto t_depth = std::chrono::high_resolution_clock::now();
    double ms_depth = std::chrono::duration<double, std::milli>(t_depth - t_disp).count();
    std::cout << "  Done in " << ms_depth << " ms" << std::endl;

    if (save_depth) {
        pipeline.saveDepthImage(output_dir + "/depth.png", depth);
    }

    // === Diagnostics: depth statistics + raw disparity samples ===
    {
        const auto& dp = pipeline.getDepthParams();
        const auto& dpar = pipeline.getDisparityParams();

        long valid_count = 0;
        double zmin = 1e18, zmax = 0.0, zsum = 0.0;
        for (int y = 0; y < depth.rows; y++) {
            for (int x = 0; x < depth.cols; x++) {
                double z = depth.at<double>(y, x);
                if (z > 0) {
                    valid_count++;
                    zsum += z;
                    if (z < zmin) zmin = z;
                    if (z > zmax) zmax = z;
                }
            }
        }
        long total_px = static_cast<long>(depth.rows) * depth.cols;
        std::cout << "\n=== Depth Statistics ===" << std::endl;
        std::cout << "  Valid pixels: " << valid_count << " / " << total_px
                  << " (" << 100.0 * valid_count / total_px << "%)" << std::endl;
        if (valid_count > 0) {
            std::cout << "  Z min: " << zmin << " mm | max: " << zmax
                      << " mm | mean: " << zsum / valid_count << " mm" << std::endl;
        }
        std::cout << "  f*B = " << dp.focal_length_px * dp.baseline_mm
                  << " px*mm | min possible Z at max disparity ("
                  << dpar.num_disparities - 1 << " px): "
                  << dp.focal_length_px * dp.baseline_mm / (dpar.num_disparities - 1)
                  << " mm" << std::endl;

        std::cout << "\n=== Raw Disparity Samples (grid, step 128) ===" << std::endl;
        std::cout << "      x    y | raw_fix  disp_px | Z_calc_mm | Z_stored_mm | status" << std::endl;
        for (int y = 64; y < disparity.rows; y += 128) {
            for (int x = 64; x < disparity.cols; x += 128) {
                printProbeRow(x, y, disparity, depth, dp);
            }
        }
    }

    if (!probe_spec.empty()) {
        std::vector<int> coords;
        std::string item;
        std::stringstream ss(probe_spec);
        while (std::getline(ss, item, ',')) {
            coords.push_back(std::stoi(item));
        }
        const auto& dp = pipeline.getDepthParams();
        std::cout << "\n=== Probed Points (" << probe_spec << ") ===" << std::endl;
        std::cout << "      x    y | raw_fix  disp_px | Z_calc_mm | Z_stored_mm | status" << std::endl;
        for (size_t k = 0; k + 1 < coords.size(); k += 2) {
            int x = coords[k], y = coords[k + 1];
            if (x < 0 || y < 0 || x >= disparity.cols || y >= disparity.rows) {
                printf("  %4d %4d | out of bounds\n", x, y);
                continue;
            }
            printProbeRow(x, y, disparity, depth, dp);
        }
    }

    // Step 4: Point cloud
    std::cout << "\n[4/4] Generating point cloud..." << std::endl;
    std::vector<cv::Vec3f> points;
    std::vector<cv::Vec3b> colors;
    pipeline.generatePointCloud(rect_left, depth, disparity, points, colors);

    auto t_pc = std::chrono::high_resolution_clock::now();
    double ms_pc = std::chrono::duration<double, std::milli>(t_pc - t_depth).count();
    std::cout << "  Generated " << points.size() << " points in " << ms_pc << " ms" << std::endl;

    // === Point cloud statistics (printed on every generation) ===
    if (!points.empty()) {
        std::cout << "\n=== Point Cloud Statistics ===" << std::endl;
        std::cout << "  Points: " << points.size() << std::endl;

        cv::Mat pts(static_cast<int>(points.size()), 3, CV_32F);
        for (size_t i = 0; i < points.size(); ++i) {
            pts.at<float>(static_cast<int>(i), 0) = points[i][0];
            pts.at<float>(static_cast<int>(i), 1) = points[i][1];
            pts.at<float>(static_cast<int>(i), 2) = points[i][2];
        }

        const char* axes[3] = {"X", "Y", "Z"};
        double lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            double mn = 1e18, mx = -1e18;
            for (int i = 0; i < pts.rows; ++i) {
                double v = pts.at<float>(i, a);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            lo[a] = mn;
            hi[a] = mx;
        }
        cv::Mat pmean(3, 1, CV_64F), pstd(3, 1, CV_64F);
        for (int a = 0; a < 3; ++a) {
            cv::Scalar m, s;
            cv::meanStdDev(pts.col(a), m, s);
            pmean.at<double>(a) = m[0];
            pstd.at<double>(a) = s[0];
        }
        for (int a = 0; a < 3; ++a) {
            printf("  %s: min=%9.1f max=%9.1f span=%9.1f mean=%9.1f std=%7.2f mm\n",
                   axes[a], lo[a], hi[a], hi[a] - lo[a],
                   pmean.at<double>(a), pstd.at<double>(a));
        }

        // Dominant plane via PCA: normal = eigenvector of smallest eigenvalue.
        cv::PCA pca(pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
        cv::Vec3f nrm(pca.eigenvectors.at<float>(2, 0),
                      pca.eigenvectors.at<float>(2, 1),
                      pca.eigenvectors.at<float>(2, 2));
        float plane_rms = std::sqrt(std::max(0.0f, pca.eigenvalues.at<float>(2, 0)));
        double tilt = std::acos(std::min(1.0, std::abs(static_cast<double>(nrm[2]))))
                      * 180.0 / CV_PI;
        printf("  Dominant plane: normal=(%.3f, %.3f, %.3f) tilt=%.1f deg from camera Z, "
               "RMS roughness=%.2f mm\n",
               nrm[0], nrm[1], nrm[2], tilt, plane_rms);

        std::vector<std::string> flags;
        if (points.size() < 1000)
            flags.push_back("too few points (<1000)");
        if (pstd.at<double>(0) < 0.5 && pstd.at<double>(1) < 0.5 && pstd.at<double>(2) < 0.5)
            flags.push_back("collapsed cloud (all axis stds < 0.5mm)");
        if (hi[2] - lo[2] < 1.0)
            flags.push_back("zero Z range");
        if (flags.empty()) {
            std::cout << "  Flags: none (cloud looks healthy)" << std::endl;
        } else {
            std::cout << "  Flags: DEGENERATE -> ";
            for (size_t f = 0; f < flags.size(); ++f)
                std::cout << flags[f] << (f + 1 < flags.size() ? "; " : "");
            std::cout << std::endl;
        }
    }

    if (save_ply && !points.empty()) {
        pipeline.savePLY(output_dir + "/pointcloud.ply", points, colors);
    }

    // Summary
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\n=== Timing Summary ===" << std::endl;
    std::cout << "  Rectify:  " << ms_rectify << " ms" << std::endl;
    std::cout << "  Disparity: " << ms_disp << " ms" << std::endl;
    std::cout << "  Depth:    " << ms_depth << " ms" << std::endl;
    std::cout << "  PointCloud: " << ms_pc << " ms" << std::endl;
    std::cout << "  Total:    " << ms_total << " ms" << std::endl;
    std::cout << "  Effective FPS: " << 1000.0 / ms_total << std::endl;

    const auto& calib = pipeline.getCalibration();
    std::cout << "\n=== Calibration Summary ===" << std::endl;
    std::cout << "  Stereo error: " << calib.reprojection_error << " px" << std::endl;
    std::cout << "  Baseline: " << calib.estimated_baseline_mm << " mm" << std::endl;
    std::cout << "  Focal length: " << pipeline.getDepthParams().focal_length_px << " px" << std::endl;

    std::cout << "\nOutput saved to: " << output_dir << "/" << std::endl;

    return 0;
}
