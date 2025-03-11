#include "map_merging/map_merging_tool.h"
#include <algorithm>

MapMergingTool::MapMergingTool(cv::Ptr<cv::Feature2D> detector, cv::Ptr<cv::DescriptorMatcher> matcher):detector(detector), matcher(matcher) 
{
    ROS_INFO("Map merging tool constructed, detector: %s, matcher: %s", detector->getDefaultName().c_str(), matcher->getDefaultName().c_str());
}

MapMergingTool::~MapMergingTool() {}

void MapMergingTool::detect_features(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

void MapMergingTool::match_features(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches)
{
    matcher->match(descriptors1, descriptors2, matches);
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {return a.distance < b.distance;});
}

cv::Mat MapMergingTool::draw_features(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints)
{
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0));
    return output;
}

cv::Mat MapMergingTool::draw_matches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                                     const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
{
    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output);
    return output;
}