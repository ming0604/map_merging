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

void MapMergingTool::match_features(const cv::Mat& descriptors1, const cv::Mat& descriptors2, vector<cv::DMatch>& final_matches)
{   
    /*
    // Normal matching
    matcher->match(descriptors1, descriptors2, matches);
    */

    // Forward matching: from descriptors1(image1) to descriptors2(image2)
    vector<vector<cv::DMatch>> forward_knn_matches;
    vector<cv::DMatch> forward_matches;
    // Find the 2 nearest neighbors (k=2) for each descriptor
    matcher->knnMatch(descriptors1, descriptors2, forward_knn_matches, 2); 

    // Perform Lowe's ratio test to filter matches 
    for(size_t i=0; i<forward_knn_matches.size(); i++)
    {   
        if(forward_knn_matches[i].size() < 2)
        {
            continue;
        }
        // If the ratio of the distance to the nearest neighbor and the second nearest neighbor is less than the threshold, keep the match
        if(forward_knn_matches[i][0].distance / forward_knn_matches[i][1].distance < ratio_threshold)
        {
            forward_matches.push_back(forward_knn_matches[i][0]);
        }
    }
    
    // Backward matching: from descriptors2(image2) to descriptors1(image1)
    vector<vector<cv::DMatch>> backward_knn_matches;
    vector<cv::DMatch> backward_matches;
    // Find the 2 nearest neighbors (k=2) for each descriptor
    matcher->knnMatch(descriptors2, descriptors1, backward_knn_matches, 2);
    
    // Perform Lowe's ratio test to filter matches
    for(size_t i=0; i<backward_knn_matches.size(); i++)
    {   
        if(forward_knn_matches[i].size() < 2)
        {
            continue;
        }
        // If the ratio of the distance to the nearest neighbor and the second nearest neighbor is less than the threshold, keep the match
        if(backward_knn_matches[i][0].distance / backward_knn_matches[i][1].distance < ratio_threshold)
        {
            backward_matches.push_back(backward_knn_matches[i][0]);
        }
    }

    // Set the forward matches as the initial final matches
    final_matches = forward_matches;
    // For each reverse match, check if it is already in the forward matches
    // If not, add the reverse match to the final matches
    for(size_t i=0; i<backward_matches.size(); i++)
    {
        // Check if the match is in the forward matches
        bool found = false;
        for(size_t j=0; j<forward_matches.size(); j++)
        {
            if((backward_matches[i].queryIdx == forward_matches[j].trainIdx) && (backward_matches[i].trainIdx == forward_matches[j].queryIdx))
            {
                found = true;
                break;
            }

        }
        // If not found, add it to the final matches
        if(!found)
        {   
            // Flip the match's queryIdx and trainIdx for matching drawing can be done correctly in the future
            cv::DMatch flipped;
            flipped.queryIdx = backward_matches[i].trainIdx;
            flipped.trainIdx = backward_matches[i].queryIdx;
            flipped.distance = backward_matches[i].distance;
            final_matches.push_back(flipped);
        }
    }
    
    // Sort final matches by distance(ascending)
    std::sort(final_matches.begin(), final_matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {return a.distance < b.distance;});
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
/**/
cv::Mat MapMergingTool::compute_affine_matrix(const vector<cv::KeyPoint>& keypoints1,const vector<cv::KeyPoint>& keypoints2,
                                            const vector<cv::DMatch>& matches, cv::Mat& inliers)
{
    // Check if the number of matches is less than 3
    if(matches.size() < 3)
    {
        ROS_ERROR("Number of matches is less than 3, cannot compute affine matrix");
        exit(1);
    }
    // Extract the matched keypoints
    vector<cv::Point2f> matched_points1, matched_points2;
    for(size_t i=0; i<matches.size(); i++)
    {   
        // Use the queryIdx and trainIdx (index) to get the matched keypoints from the two images's keypoints
        matched_points1.push_back(keypoints1[matches[i].queryIdx].pt);
        matched_points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    // Use RANSAC to estimate the affine transformation matrix, and record the inliers

    cv::Mat affine_mat = cv::estimateAffinePartial2D(matched_points2, matched_points1, inliers, cv::RANSAC);
    // Check if the affine matrix is empty
    if(affine_mat.empty())
    {
        ROS_ERROR("Failed to compute affine matrix");
        exit(1);
    }
    else
    {
        ROS_INFO("Affine matrix computed successfully");
        return affine_mat;
    }

}       

cv::Mat MapMergingTool::draw_inlier_matches(const cv::Mat& image1, const vector<cv::KeyPoint>& keypoints1, const cv::Mat& image2,
                                            const vector<cv::KeyPoint>& keypoints2, const vector<cv::DMatch>& matches, const cv::Mat& inliers)
{
    vector<cv::DMatch> inlier_matches;
    cv::Mat output;

    // Check if the number of inliers is same as the number of matches
    if(inliers.rows != matches.size())
    {
        ROS_ERROR("Number of inliers mask vector is not equal to the number of matches");
        exit(1);
    }

    // Extract the inlier matches
    for(int i=0; i<inliers.rows; i++)
    {   
        // If the inlier flag is 1, means it is an inlier match
        if(inliers.at<uchar>(i,0) == 1)
        {   
            inlier_matches.push_back(matches[i]);
        }
    }
    
    // Draw the inlier matches
    cv::drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, output, cv::Scalar(0,0,255) , cv::Scalar(0,255,0));
    // Print the number of inliers
    int inlier_matches_size = inlier_matches.size();
    cout << "Number of inlier matches: " << inlier_matches_size << endl;
    return output;

}
