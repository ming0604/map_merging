#include "map_merging/map_merging_tool.h"
#include <algorithm>

MapMergingTool::MapMergingTool(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher, float ratio_thr):detector(feature_detector), matcher(feature_matcher)
, ratio_threshold(ratio_thr) 
{
    ROS_INFO("Map merging tool constructed,and the matching ratio threshold is %f",ratio_threshold);
}

MapMergingTool::~MapMergingTool() {}

void MapMergingTool::detect_features(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

void MapMergingTool::match_features(const cv::Mat& descriptors1, const cv::Mat& descriptors2, vector<cv::DMatch>& matches)
{   
    // Normal matching
    //matcher->match(descriptors1, descriptors2, matches);

    // Perform Lowe's ratio test to filter matches 
    vector<vector<cv::DMatch>> knn_matches;
    // Find the 2 nearest neighbors (k=2) for each descriptor
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2); 
    for(size_t i=0; i<knn_matches.size(); i++)
    {
        // If the ratio of the distance to the nearest neighbor and the second nearest neighbor is less than the threshold, keep the match
        if(knn_matches[i][0].distance / knn_matches[i][1].distance < ratio_threshold)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }

    // Sort matches by distance(ascending)
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {return a.distance < b.distance;});

}

cv::Mat MapMergingTool::draw_features(const cv::Mat& image, const vector<cv::KeyPoint>& keypoints)
{
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0));
    return output;
}

cv::Mat MapMergingTool::draw_matches(const cv::Mat& image1, const vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                                     const vector<cv::KeyPoint>& keypoints2, const vector<cv::DMatch>& matches)
{
    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output);
    return output;
}