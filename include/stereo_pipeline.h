#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

namespace stereo {

struct CameraParams {
    cv::Mat K;          // 3x3 camera matrix
    cv::Mat dist;       // distortion coefficients
    cv::Mat R;          // rotation matrix (for rectification)
    cv::Mat P;          // projection matrix (for rectification)
    cv::Mat map_x;      // rectification map X
    cv::Mat map_y;      // rectification map Y
    int width = 0;
    int height = 0;
};

struct StereoParams {
    cv::Mat R;          // rotation between cameras
    cv::Mat T;          // translation between cameras (mm)
    cv::Mat E;          // essential matrix
    cv::Mat F;          // fundamental matrix
    cv::Mat Q;          // disparity-to-depth mapping matrix
    double baseline_mm = 0.0;
};

struct DisparityParams {
    int algorithm = 0;  // 0=SGBM, 1=BM
    int min_disparity = 0;
    int num_disparities = 128;
    int block_size = 7;
    int P1 = 0;
    int P2 = 0;
    int disp12_max_diff = 1;
    int uniqueness_ratio = 10;
    int speckle_window_size = 100;
    int speckle_range = 32;
    int pre_filter_cap = 63;
    int mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    bool wls_filter = true;
    double wls_lambda = 8000.0;
    double wls_sigma = 1.5;
};

struct DepthParams {
    double min_depth_mm = 200.0;
    double max_depth_mm = 5000.0;
    double focal_length_px = 0.0;
    double baseline_mm = 0.0;
    double texture_min_grad = 0.0;   // 0 disables texture masking
};

struct PointCloudParams {
    double voxel_size_mm = 5.0;
    bool colorize = true;
    int max_points = 500000;
    std::string output_format = "ply";
};

struct CalibrationData {
    CameraParams left;
    CameraParams right;
    StereoParams stereo;
    int image_width = 0;
    int image_height = 0;
    int num_valid_pairs = 0;
    double reprojection_error = 0.0;
    double estimated_baseline_mm = 0.0;
};

class StereoPipeline {
public:
    StereoPipeline();
    ~StereoPipeline();

    bool loadCalibration(const std::string& calib_file);
    bool loadConfig(const std::string& config_file);

    void rectify(const cv::Mat& left, const cv::Mat& right,
                 cv::Mat& rect_left, cv::Mat& rect_right);

    void computeDisparity(const cv::Mat& rect_left, const cv::Mat& rect_right,
                          cv::Mat& disparity, cv::Mat& disparity_vis);

    void computeDepth(const cv::Mat& disparity, cv::Mat& depth);

    void generatePointCloud(const cv::Mat& left, const cv::Mat& depth,
                            const cv::Mat& disparity,
                            std::vector<cv::Vec3f>& points,
                            std::vector<cv::Vec3b>& colors);

    bool savePLY(const std::string& filename,
                 const std::vector<cv::Vec3f>& points,
                 const std::vector<cv::Vec3b>& colors);

    bool saveDepthImage(const std::string& filename, const cv::Mat& depth);

    bool saveDisparityImage(const std::string& filename, const cv::Mat& disparity_vis);

    const CalibrationData& getCalibration() const { return calib_; }
    const DisparityParams& getDisparityParams() const { return disp_params_; }
    const DepthParams& getDepthParams() const { return depth_params_; }

    void setDisparityParams(const DisparityParams& p) { disp_params_ = p; }
    void setDepthParams(const DepthParams& p) { depth_params_ = p; }
    void setDownscale(int factor) { downscale_ = factor > 1 ? factor : 1; }

private:
    void initStereoMatcher();
    void filterDisparity(const cv::Mat& disparity_left, const cv::Mat& disparity_right,
                         cv::Mat& filtered);
    void applyTextureMask(cv::Mat& depth);

    CalibrationData calib_;
    DisparityParams disp_params_;
    DepthParams depth_params_;
    PointCloudParams pc_params_;

    cv::Ptr<cv::StereoSGBM> matcher_left_;
    cv::Ptr<cv::StereoMatcher> matcher_right_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    cv::Mat last_left_gray_;   // rectified left gray, saved by computeDisparity
    int downscale_ = 1;        // disparity compute downscale factor (1 = full res)
};

} // namespace stereo
