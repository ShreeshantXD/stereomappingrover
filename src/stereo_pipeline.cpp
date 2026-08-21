#include "stereo_pipeline.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace stereo {

static cv::Mat toMat(const std::vector<std::vector<double>>& v) {
    if (v.empty() || v[0].empty()) {
        return cv::Mat();
    }
    cv::Mat m(static_cast<int>(v.size()), static_cast<int>(v[0].size()), CV_64FC1);
    for (int r = 0; r < static_cast<int>(v.size()); ++r) {
        for (int c = 0; c < static_cast<int>(v[0].size()); ++c) {
            m.at<double>(r, c) = v[r][c];
        }
    }
    return m;
}

StereoPipeline::StereoPipeline() {}
StereoPipeline::~StereoPipeline() {}

bool StereoPipeline::loadCalibration(const std::string& calib_file) {
    try {
        YAML::Node calib = YAML::LoadFile(calib_file);

        // Image resolution
        calib_.image_width = calib["resolution"]["width"].as<int>();
        calib_.image_height = calib["resolution"]["height"].as<int>();
        calib_.num_valid_pairs = calib["num_valid_pairs"].as<int>();
        calib_.reprojection_error = calib["reprojection_error"]["stereo"].as<double>();
        calib_.estimated_baseline_mm = calib["estimated_baseline_mm"].as<double>();

        // Left camera
        auto cam_left = calib["camera_left"];
        calib_.left.K = toMat(cam_left["K"].as<std::vector<std::vector<double>>>());
        calib_.left.dist = toMat(cam_left["dist"].as<std::vector<std::vector<double>>>());

        // Right camera
        auto cam_right = calib["camera_right"];
        calib_.right.K = toMat(cam_right["K"].as<std::vector<std::vector<double>>>());
        calib_.right.dist = toMat(cam_right["dist"].as<std::vector<std::vector<double>>>());

        // Stereo extrinsics
        calib_.stereo.R = toMat(calib["stereo"]["R"].as<std::vector<std::vector<double>>>());
        calib_.stereo.T = toMat(calib["stereo"]["T"].as<std::vector<std::vector<double>>>());
        calib_.stereo.baseline_mm = calib_.estimated_baseline_mm;

        // Rectification
        calib_.left.R = toMat(calib["rectification"]["R1"].as<std::vector<std::vector<double>>>());
        calib_.left.P = toMat(calib["rectification"]["P1"].as<std::vector<std::vector<double>>>());
        calib_.right.R = toMat(calib["rectification"]["R2"].as<std::vector<std::vector<double>>>());
        calib_.right.P = toMat(calib["rectification"]["P2"].as<std::vector<std::vector<double>>>());
        calib_.stereo.Q = toMat(calib["rectification"]["Q"].as<std::vector<std::vector<double>>>());

        // Rectification maps are derived data - compute them from the
        // calibration matrices instead of storing them in the YAML file.
        const cv::Size img_size(calib_.image_width, calib_.image_height);
        cv::initUndistortRectifyMap(calib_.left.K, calib_.left.dist,
                                    calib_.left.R, calib_.left.P,
                                    img_size, CV_32FC1,
                                    calib_.left.map_x, calib_.left.map_y);
        cv::initUndistortRectifyMap(calib_.right.K, calib_.right.dist,
                                    calib_.right.R, calib_.right.P,
                                    img_size, CV_32FC1,
                                    calib_.right.map_x, calib_.right.map_y);

        calib_.left.width = calib_.image_width;
        calib_.left.height = calib_.image_height;
        calib_.right.width = calib_.image_width;
        calib_.right.height = calib_.image_height;

        // Set depth parameters from calibration
        depth_params_.baseline_mm = calib_.stereo.baseline_mm;
        depth_params_.focal_length_px = calib_.left.P.at<double>(0, 0);

        std::cout << "Calibration loaded: " << calib_file << std::endl;
        std::cout << "  Resolution: " << calib_.image_width << "x" << calib_.image_height << std::endl;
        std::cout << "  Stereo error: " << calib_.reprojection_error << " px" << std::endl;
        std::cout << "  Baseline: " << calib_.stereo.baseline_mm << " mm" << std::endl;
        std::cout << "  Focal length: " << depth_params_.focal_length_px << " px" << std::endl;

        initStereoMatcher();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load calibration: " << e.what() << std::endl;
        return false;
    }
}

bool StereoPipeline::loadConfig(const std::string& config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        // Disparity parameters
        auto disp = config["disparity"];
        disp_params_.min_disparity = disp["sgbm"]["min_disparity"].as<int>();
        disp_params_.num_disparities = disp["sgbm"]["num_disparities"].as<int>();
        disp_params_.block_size = disp["sgbm"]["block_size"].as<int>();
        disp_params_.P1 = disp["sgbm"]["P1"].as<int>();
        disp_params_.P2 = disp["sgbm"]["P2"].as<int>();
        disp_params_.disp12_max_diff = disp["sgbm"]["disp12_max_diff"].as<int>();
        disp_params_.uniqueness_ratio = disp["sgbm"]["uniqueness_ratio"].as<int>();
        disp_params_.speckle_window_size = disp["sgbm"]["speckle_window_size"].as<int>();
        disp_params_.speckle_range = disp["sgbm"]["speckle_range"].as<int>();
        disp_params_.pre_filter_cap = disp["sgbm"]["pre_filter_cap"].as<int>();

        // Parse mode string to enum
        std::string mode_str = disp["sgbm"]["mode"].as<std::string>();
        if (mode_str == "SGBM_3WAY") {
            disp_params_.mode = cv::StereoSGBM::MODE_SGBM_3WAY;
        } else if (mode_str == "SGBM") {
            disp_params_.mode = cv::StereoSGBM::MODE_SGBM;
        } else {
            std::cerr << "Warning: Unknown disparity mode '" << mode_str
                      << "', using SGBM_3WAY" << std::endl;
            disp_params_.mode = cv::StereoSGBM::MODE_SGBM_3WAY;
        }

        disp_params_.wls_filter = disp["wls_filter"].as<bool>();
        disp_params_.wls_lambda = disp["wls_lambda"].as<double>();
        disp_params_.wls_sigma = disp["wls_sigma"].as<double>();

        // Depth parameters
        auto depth = config["depth"];
        depth_params_.min_depth_mm = depth["min_depth_mm"].as<double>();
        depth_params_.max_depth_mm = depth["max_depth_mm"].as<double>();
        if (depth["texture_min_gradient"]) {
            depth_params_.texture_min_grad = depth["texture_min_gradient"].as<double>();
        }

        // Point cloud parameters
        auto pc = config["pointcloud"];
        pc_params_.colorize = pc["colorize"].as<bool>();
        pc_params_.max_points = pc["max_points"].as<int>();
        pc_params_.voxel_size_mm = pc["downsample_voxel_mm"].as<double>();

        std::cout << "Config loaded: " << config_file << std::endl;
        std::cout << "  Disparity params:" << std::endl;
        std::cout << "    min_disparity:    " << disp_params_.min_disparity << std::endl;
        std::cout << "    num_disparities:  " << disp_params_.num_disparities << std::endl;
        std::cout << "    block_size:       " << disp_params_.block_size << std::endl;
        std::cout << "    P1:               " << disp_params_.P1 << std::endl;
        std::cout << "    P2:               " << disp_params_.P2 << std::endl;
        std::cout << "    disp12_max_diff:  " << disp_params_.disp12_max_diff << std::endl;
        std::cout << "    uniqueness_ratio: " << disp_params_.uniqueness_ratio << std::endl;
        std::cout << "    speckle_win_size: " << disp_params_.speckle_window_size << std::endl;
        std::cout << "    speckle_range:    " << disp_params_.speckle_range << std::endl;
        std::cout << "    pre_filter_cap:   " << disp_params_.pre_filter_cap << std::endl;
        std::cout << "    mode:             " << mode_str << std::endl;
        std::cout << "    wls_filter:       " << (disp_params_.wls_filter ? "ON" : "OFF") << std::endl;
        std::cout << "  Depth params:" << std::endl;
        std::cout << "    min_depth_mm:     " << depth_params_.min_depth_mm << std::endl;
        std::cout << "    max_depth_mm:     " << depth_params_.max_depth_mm << std::endl;
        std::cout << "    texture_min_grad: " << depth_params_.texture_min_grad
                  << (depth_params_.texture_min_grad > 0.0 ? "" : " (disabled)") << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        return false;
    }
}

void StereoPipeline::initStereoMatcher() {
    // NOTE: argument order matters - create()'s signature is
    // (minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
    //  preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode)
    matcher_left_ = cv::StereoSGBM::create(
        disp_params_.min_disparity,
        disp_params_.num_disparities,
        disp_params_.block_size,
        disp_params_.P1,
        disp_params_.P2,
        disp_params_.disp12_max_diff,
        disp_params_.pre_filter_cap,
        disp_params_.uniqueness_ratio,
        disp_params_.speckle_window_size,
        disp_params_.speckle_range,
        disp_params_.mode
    );

    if (disp_params_.wls_filter) {
        wls_filter_ = cv::ximgproc::createDisparityWLSFilter(matcher_left_);
        wls_filter_->setLambda(disp_params_.wls_lambda);
        wls_filter_->setSigmaColor(disp_params_.wls_sigma);

        matcher_right_ = cv::ximgproc::createRightMatcher(matcher_left_);
    }
}

void StereoPipeline::rectify(const cv::Mat& left, const cv::Mat& right,
                              cv::Mat& rect_left, cv::Mat& rect_right) {
    cv::remap(left, rect_left, calib_.left.map_x, calib_.left.map_y, cv::INTER_LINEAR);
    cv::remap(right, rect_right, calib_.right.map_x, calib_.right.map_y, cv::INTER_LINEAR);
}

void StereoPipeline::computeDisparity(const cv::Mat& rect_left, const cv::Mat& rect_right,
                                      cv::Mat& disparity, cv::Mat& disparity_vis) {
    cv::Mat gray_left, gray_right;
    if (rect_left.channels() == 3) {
        cv::cvtColor(rect_left, gray_left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rect_right, gray_right, cv::COLOR_BGR2GRAY);
    } else {
        gray_left = rect_left;
        gray_right = rect_right;
    }
    last_left_gray_ = gray_left.clone();

    int factor = downscale_;
    cv::Mat match_left = gray_left, match_right = gray_right;
    if (factor > 1) {
        cv::Size small(gray_left.cols / factor, gray_left.rows / factor);
        cv::resize(gray_left, match_left, small, 0, 0, cv::INTER_AREA);
        cv::resize(gray_right, match_right, small, 0, 0, cv::INTER_AREA);
    }

    // Upscale a small fixed-point disparity map back to full resolution.
    auto upscale_disp = [&](const cv::Mat& small_disp) {
        cv::Mat f32;
        small_disp.convertTo(f32, CV_32F, 1.0 / 16.0);
        cv::resize(f32, f32, gray_left.size(), 0, 0, cv::INTER_LINEAR);
        f32 *= static_cast<float>(factor);
        f32.convertTo(disparity, CV_16S, 16.0);
    };

    cv::Mat disp_left, disp_right;

    if (disp_params_.wls_filter && wls_filter_ && matcher_right_) {
        // Left and right matching are independent - run them concurrently.
        std::thread right_thread([&] {
            matcher_right_->compute(match_right, match_left, disp_right);
        });
        matcher_left_->compute(match_left, match_right, disp_left);
        right_thread.join();

        cv::Mat filtered;
        wls_filter_->filter(disp_left, match_left, filtered, disp_right);

        if (factor > 1) {
            upscale_disp(filtered);
        } else {
            disparity = filtered;
        }
    } else {
        matcher_left_->compute(match_left, match_right, disp_left);
        if (factor > 1) {
            upscale_disp(disp_left);
        } else {
            disparity = disp_left;
        }
    }

    // Convert to 8-bit and apply color map for visualization (invalid pixels black)
    cv::Mat disp_gray;
    disparity.convertTo(disp_gray, CV_8UC1, 255.0 / (disp_params_.num_disparities * 16.0));
    cv::applyColorMap(disp_gray, disparity_vis, cv::COLORMAP_JET);
    disparity_vis.setTo(cv::Scalar::all(0), disparity <= 0);
}

void StereoPipeline::computeDepth(const cv::Mat& disparity, cv::Mat& depth) {
    depth = cv::Mat(disparity.size(), CV_64FC1);

    double f = depth_params_.focal_length_px;
    double B = depth_params_.baseline_mm;

    for (int y = 0; y < disparity.rows; y++) {
        for (int x = 0; x < disparity.cols; x++) {
            double disp = disparity.at<short>(y, x) / 16.0; // SGBM returns 16x fixed point

            if (disp > 0) {
                double Z = (f * B) / disp;
                if (Z >= depth_params_.min_depth_mm && Z <= depth_params_.max_depth_mm) {
                    depth.at<double>(y, x) = Z;
                } else {
                    depth.at<double>(y, x) = 0; // invalid
                }
            } else {
                depth.at<double>(y, x) = 0; // invalid
            }
        }
    }

    applyTextureMask(depth);
}

void StereoPipeline::applyTextureMask(cv::Mat& depth) {
    if (depth_params_.texture_min_grad <= 0.0 || last_left_gray_.empty()) {
        return;
    }

    // Local gradient magnitude, averaged over a window. Textureless regions
    // have near-zero mean gradient; SGBM output there is unreliable.
    cv::Mat gx, gy;
    cv::Sobel(last_left_gray_, gx, CV_32F, 1, 0, 3);
    cv::Sobel(last_left_gray_, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    cv::boxFilter(mag, mag, -1, cv::Size(15, 15));

    cv::Mat low_texture = mag < depth_params_.texture_min_grad;
    depth.setTo(0, low_texture);

    int masked = cv::countNonZero(low_texture);
    std::cout << "  Texture mask: " << masked << " px ("
              << 100.0 * masked / (mag.rows * mag.cols) << "%) marked invalid" << std::endl;
}

void StereoPipeline::generatePointCloud(const cv::Mat& left, const cv::Mat& depth,
                                         const cv::Mat& disparity,
                                         std::vector<cv::Vec3f>& points,
                                         std::vector<cv::Vec3b>& colors) {
    points.clear();
    colors.clear();

    double fx = calib_.left.P.at<double>(0, 0);
    double fy = calib_.left.P.at<double>(1, 1);
    double cx = calib_.left.P.at<double>(0, 2);
    double cy = calib_.left.P.at<double>(1, 2);

    for (int y = 0; y < depth.rows; y++) {
        for (int x = 0; x < depth.cols; x++) {
            double Z = depth.at<double>(y, x);
            if (Z <= 0) continue;

            double disp_val = disparity.at<short>(y, x) / 16.0;
            if (disp_val <= 0) continue;

            double X = (x - cx) * Z / fx;
            double Y = (y - cy) * Z / fy;

            points.push_back(cv::Vec3f(X, Y, Z));

            if (pc_params_.colorize && left.channels() == 3) {
                cv::Vec3b bgr = left.at<cv::Vec3b>(y, x);
                colors.push_back(cv::Vec3b(bgr[2], bgr[1], bgr[0])); // RGB
            } else {
                colors.push_back(cv::Vec3b(255, 255, 255));
            }
        }
    }

    // Cap point count
    if (static_cast<int>(points.size()) > pc_params_.max_points) {
        int step = (static_cast<int>(points.size()) + pc_params_.max_points - 1) / pc_params_.max_points;
        std::vector<cv::Vec3f> sampled_points;
        std::vector<cv::Vec3b> sampled_colors;
        for (size_t i = 0; i < points.size(); i += step) {
            sampled_points.push_back(points[i]);
            sampled_colors.push_back(colors[i]);
        }
        points = sampled_points;
        colors = sampled_colors;
    }
}

bool StereoPipeline::savePLY(const std::string& filename,
                              const std::vector<cv::Vec3f>& points,
                              const std::vector<cv::Vec3b>& colors) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PLY file: " << filename << std::endl;
        return false;
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    for (size_t i = 0; i < points.size(); i++) {
        file << points[i][0] << " " << points[i][1] << " " << points[i][2] << " "
             << static_cast<int>(colors[i][0]) << " "
             << static_cast<int>(colors[i][1]) << " "
             << static_cast<int>(colors[i][2]) << "\n";
    }

    file.close();
    std::cout << "Saved PLY: " << filename << " (" << points.size() << " points)" << std::endl;
    return true;
}

bool StereoPipeline::saveDepthImage(const std::string& filename, const cv::Mat& depth) {
    double min_val, max_val;
    cv::minMaxLoc(depth, &min_val, &max_val, nullptr, nullptr,
                  depth > 0);

    if (max_val <= min_val) {
        std::cerr << "No valid depth values to save" << std::endl;
        return false;
    }

    // Near = red, far = blue, invalid = black.
    double range = max_val - min_val;
    cv::Mat depth_norm;
    depth.convertTo(depth_norm, CV_8UC1,
                    -255.0 / range, max_val * 255.0 / range);
    cv::Mat depth_vis;
    cv::applyColorMap(depth_norm, depth_vis, cv::COLORMAP_JET);
    depth_vis.setTo(cv::Scalar::all(0), depth <= 0);

    cv::imwrite(filename, depth_vis);
    std::cout << "Saved depth image: " << filename << std::endl;
    return true;
}

bool StereoPipeline::saveDisparityImage(const std::string& filename, const cv::Mat& disparity_vis) {
    cv::imwrite(filename, disparity_vis);
    std::cout << "Saved disparity: " << filename << std::endl;
    return true;
}

} // namespace stereo
