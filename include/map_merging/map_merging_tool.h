#ifndef MAP_MERGING_TOOL_H
#define MAP_MERGING_TOOL_H

#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


class MapMergingTool
{
    public:
        MapMergingTool(cv::Ptr<cv::Feature2D> detector, cv::Ptr<cv::DescriptorMatcher> matcher);
        virtual ~MapMergingTool();

        void detect_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        void match_features(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);
        cv::Mat draw_features(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);
        cv::Mat draw_matches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                             const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches);
        /*
        cv::Mat compute_homography(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, 
                                   const std::vector<cv::DMatch>& matches);
        cv::Mat merge_maps(const cv::Mat& map1, const cv::Mat& map2, const cv::Mat& H, const double origin1[3], double resolution, double outOrigin[3]);
        */
    private:
        cv::Ptr<cv::Feature2D> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;
};


#endif
