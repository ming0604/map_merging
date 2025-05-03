#ifndef MAP_MERGING_TOOL_H
#define MAP_MERGING_TOOL_H

#include <ros/ros.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


typedef struct bbox
{
    int x_min;
    int x_max;
    int y_min;
    int y_max;
}bbox;// bounding box of the map in pixel coordinates

// Define the color values of every state
const uchar OCCUPIED_COLOR = 0; // black
const uchar FREE_COLOR = 254; // almost white
const uchar UNKNOWN_COLOR = 205; // light gray


class MapMergingTool
{
    public:
        MapMergingTool(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher, float ratio_thr);
        virtual ~MapMergingTool();

        void set_ratio_threshold(float thr); 
        float get_ratio_threshold() const;
        void detect_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        void match_features(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);
        cv::Mat draw_features(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);
        cv::Mat draw_matches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                             const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches);

        cv::Mat compute_affine_matrix(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, 
                                        const std::vector<cv::DMatch>& matches, cv::Mat& inliers, int& num_inliers);
        cv::Mat draw_inlier_matches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                                    const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches, const cv::Mat& inliers);
        cv::Mat merge_maps(const cv::Mat& reference_map, const cv::Mat& align_map, const cv::Mat& affine_align_to_ref, 
                            int& origin_shift_x_cells, int& origin_shift_y_cells, double& acceptance_index);
    private:
        cv::Ptr<cv::Feature2D> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        float ratio_threshold;
        
        bbox compute_global_map_bbox(const cv::Mat& ref_map, const cv::Mat& align_map, const cv::Mat& affine_align_to_ref);
        void compute_origin_shift(const bbox& global_bbox_on_ref, int ref_rows, int& origin_shift_x_cells, int& origin_shift_y_cells);
        void compute_global_affine(const bbox& global_bbox, const cv::Mat& affine_align_to_ref, cv::Mat& affine_ref_to_global,
                                    cv::Mat& affine_align_to_global);
        double compute_acceptance_index(const cv::Mat& ref_map_on_global, const cv::Mat& align_map_on_global);
        cv::Mat generate_merged_map(const cv::Mat& ref_map_on_global, const cv::Mat& align_map_on_global);
};  


#endif
