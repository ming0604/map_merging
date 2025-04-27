#include "map_merging/map_merging_tool.h"

using namespace std;

MapMergingTool::MapMergingTool(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher, float ratio_thr):detector(feature_detector), matcher(feature_matcher)
, ratio_threshold(ratio_thr) 
{
    ROS_INFO("Map merging tool constructed, feature type: %s, ratio threshold: %f", feature_detector->getDefaultName().c_str(), ratio_thr);
}

MapMergingTool::~MapMergingTool() {}


void MapMergingTool::set_ratio_threshold(float thr)
{
    ratio_threshold = thr;
}

float MapMergingTool::get_ratio_threshold() const
{
    return ratio_threshold;
}

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
        if(backward_knn_matches[i].size() < 2)
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

bbox MapMergingTool::compute_global_map_bbox(const cv::Mat& ref_map, const cv::Mat& align_map, const cv::Mat& affine_align_to_ref)
{
    // Get the corner points of the reference map (in continuous coordinates)
    vector<cv::Point2f> ref_map_corners = {
        {0.0, 0.0},
        {static_cast<float>(ref_map.cols), 0.0},
        {0.0, static_cast<float>(ref_map.rows)},
        {static_cast<float>(ref_map.cols), static_cast<float>(ref_map.rows)}
    };
    // Get the corner points of the aligned map (in continuous coordinates)
    vector<cv::Point2f> align_map_corners = {
        {0.0, 0.0},
        {static_cast<float>(align_map.cols), 0.0},
        {0.0, static_cast<float>(align_map.rows)},
        {static_cast<float>(align_map.cols), static_cast<float>(align_map.rows)}
    };
    // Transform the corner points of the aligned map to the reference map's coordinate system
    vector<cv::Point2f> transformed_align_map_corners;
    cv::transform(align_map_corners, transformed_align_map_corners, affine_align_to_ref);

    // Get the min and max x,y coordinates of all the corner points of the reference map and the transformed aligned map
    vector<cv::Point2f> all_corners;
    all_corners.insert(all_corners.end(), ref_map_corners.begin(), ref_map_corners.end());
    all_corners.insert(all_corners.end(), transformed_align_map_corners.begin(), transformed_align_map_corners.end());
    // Set the min and max x,y eqals to the first corner point as the initial value
    float x_min = all_corners[0].x;
    float x_max = all_corners[0].x;
    float y_min = all_corners[0].y;
    float y_max = all_corners[0].y;
    // Iterate through all the corner points to find the min and max x,y coordinates
    for(size_t i = 0; i < all_corners.size(); i++)
    {
        if(all_corners[i].x < x_min)
        {
            x_min = all_corners[i].x;
        }
        if(all_corners[i].x > x_max)
        {
            x_max = all_corners[i].x;
        }
        if(all_corners[i].y < y_min)
        {
            y_min = all_corners[i].y;
        }
        if(all_corners[i].y > y_max)
        {
            y_max = all_corners[i].y;
        }
    }

    // Create a global map bounding box
    bbox global_map_bbox;
    // Min needs to be floored (to the upper left)m
    global_map_bbox.x_min = static_cast<int>(std::floor(x_min));
    global_map_bbox.y_min = static_cast<int>(std::floor(y_min));
    // Max needs to be ceiled (to the lower right)
    // The max value needs to -1 because the pixel coordinates 
    global_map_bbox.x_max = static_cast<int>(std::ceil(x_max))-1;
    global_map_bbox.y_max = static_cast<int>(std::ceil(y_max))-1;

    return global_map_bbox;
}

void MapMergingTool::compute_origin_shift(const bbox& global_bbox_on_ref, int ref_rows, int& origin_shift_x_cells, int& origin_shift_y_cells)
{
    // Compute the origin shift in ROS map coordinates, which is the bottom-left cell need to be added to the map
    origin_shift_x_cells = -global_bbox_on_ref.x_min;
    origin_shift_y_cells = global_bbox_on_ref.y_max - (ref_rows - 1); // ref_rows  
}

void MapMergingTool::compute_global_affine(const bbox& global_bbox, const cv::Mat& affine_align_to_ref, cv::Mat& affine_ref_to_global,
                                            cv::Mat& affine_align_to_global)
{
    // Compute shift of the reference map to the global map
    double tx = -global_bbox.x_min;
    double ty = -global_bbox.y_min;
    // Create the affine matrix (2*3) from the reference map to the global map
    // [1 0 tx]
    // [0 1 ty]
    affine_ref_to_global = (cv::Mat_<double>(2, 3) << 1.0, 0.0, tx, 0.0, 1.0, ty);

    // Use homogeneous transformation to compute the affine matrix from the aligned map to the global map

    // Build 3×3 homogeneous transformation matrix of the reference map to the global map
    //  [1 0 tx]
    //  [0 1 ty]
    //  [0 0 1 ]
    cv::Mat homo_ref_to_global = cv::Mat::eye(3, 3, CV_64F);
    homo_ref_to_global.at<double>(0, 2) = tx;
    homo_ref_to_global.at<double>(1, 2) = ty;
    // Build 3×3 homogeneous transformation matrix of the align map to the reference map
    cv::Mat homo_align_to_ref = cv::Mat::eye(3, 3, CV_64F);
    affine_align_to_ref.copyTo(homo_align_to_ref(cv::Range(0, 2), cv::Range(0, 3)));
    // align_to_global = ref_to_global * align_to_ref
    cv::Mat homo_align_to_global = homo_ref_to_global * homo_align_to_ref;
    // Extract back to 2×3 for affine matrix
    affine_align_to_global = homo_align_to_global(cv::Range(0, 2), cv::Range(0, 3)).clone();
}

double MapMergingTool::compute_acceptance_index(const cv::Mat& ref_map_on_global, const cv::Mat& align_map_on_global)
{   

    // Check if the two maps have the same size in the global coordinate system.
    // If not, report an error and return -1.
    if(ref_map_on_global.size() != align_map_on_global.size())
    {
        ROS_ERROR("compute_acceptance_index: global size mismatch: ref=(%dx%d), ali=(%dx%d)",ref_map_on_global.rows, 
                    ref_map_on_global.cols, align_map_on_global.rows, align_map_on_global.cols);
        return -1.0;
    }

    int agr_cells = 0;
    int dis_cells = 0;
    uchar ref_value, align_value;
    for(int y=0; y<ref_map_on_global.rows; y++)
    {   
        for(int x=0; x<ref_map_on_global.cols; x++)
        {
            ref_value = ref_map_on_global.at<uchar>(y,x);
            align_value = align_map_on_global.at<uchar>(y,x);
            // Skip the unknown cells
            if(ref_value == UNKNOWN_COLOR || align_value == UNKNOWN_COLOR)
            {
                continue;
            }

            // Check the value is valid
            if((ref_value == OCCUPIED_COLOR || ref_value == FREE_COLOR)&&
               (align_value == OCCUPIED_COLOR || align_value == FREE_COLOR))
            {
                // If the two maps have the same value, count as agreement
                if(ref_value == align_value)
                {
                    agr_cells++;
                }
                // If the two maps have different values, count as disagreement
                else
                {
                    dis_cells++;
                }
            }
            else
            {
                ROS_ERROR("compute_acceptance_index: invalid value in the map at (%d,%d): reference map value=%u, aligned map value=%u",
                            y, x, static_cast<unsigned int>(ref_value), static_cast<unsigned int>(align_value));
                return -1.0;
            }
        }
    }

    // Compute the acceptance index
    double acc_index;
    if(agr_cells + dis_cells == 0)
    {
        ROS_WARN("compute_acceptance_index: (agr + dis) equals 0, return 0.0");
        acc_index = 0.0;
    }
    else
    {
        acc_index = (static_cast<double>(agr_cells) / static_cast<double>(agr_cells + dis_cells));
    }
    
    return acc_index;
}

cv::Mat MapMergingTool::generate_merged_map(const cv::Mat& ref_map_on_global, const cv::Mat& align_map_on_global)
{
    // Check if the two maps have the same size in the global coordinate system.
    // If not, report an error and return an empty map.
    if(ref_map_on_global.size() != align_map_on_global.size())
    {
        ROS_ERROR("generate_merged_map: global size mismatch: ref=(%dx%d), ali=(%dx%d)",ref_map_on_global.rows, 
                    ref_map_on_global.cols, align_map_on_global.rows, align_map_on_global.cols);
        exit(-1);
    }

    // Create a new map to store the merged map (initially set to unknown color)
    cv::Mat merged_map = cv::Mat(ref_map_on_global.size(), CV_8UC1, cv::Scalar(UNKNOWN_COLOR));
    uchar ref_value, align_value;
    for(int y=0; y<ref_map_on_global.rows; y++)
    {   
        for(int x=0; x<ref_map_on_global.cols; x++)
        {
            ref_value = ref_map_on_global.at<uchar>(y,x);
            align_value = align_map_on_global.at<uchar>(y,x);
            // If the two maps have the same value, set the merged map to that value
            if(ref_value == align_value)
            {
                merged_map.at<uchar>(y,x) = ref_value;
            }
            // If one map is occupied and the other is free, set the merged map to occupied
            else if((ref_value == OCCUPIED_COLOR && align_value == FREE_COLOR) ||
                    (ref_value == FREE_COLOR && align_value == OCCUPIED_COLOR))
            {
                merged_map.at<uchar>(y,x) = OCCUPIED_COLOR;
            }
            // If one map is unknown and the other is Known(free or occpied), set the merged map to the known value
            else if(ref_value == UNKNOWN_COLOR && (align_value == OCCUPIED_COLOR || align_value == FREE_COLOR))
            {
                merged_map.at<uchar>(y,x) = align_value;
            }
            else if(align_value == UNKNOWN_COLOR && (ref_value == OCCUPIED_COLOR || ref_value == FREE_COLOR))
            {
                merged_map.at<uchar>(y,x) = ref_value;
            }
   
            else
            {
                ROS_ERROR("generate_merged_map: invalid value in the map at (%d,%d): reference map value=%u, aligned map value=%u",
                            y, x, static_cast<unsigned int>(ref_value), static_cast<unsigned int>(align_value));
                exit(-1);
            }
        }
    }

    // Return the merged map
    return merged_map;
}

cv::Mat MapMergingTool::merge_maps(const cv::Mat& reference_map, const cv::Mat& align_map, const cv::Mat& affine_align_to_ref, 
                                    int& origin_shift_x_cells, int& origin_shift_y_cells, double& acceptance_index)
{
    // 1. Compute the global map bounding box
    bbox global_map_bbox_on_ref_frame = compute_global_map_bbox(reference_map, align_map, affine_align_to_ref);

    // 2. Compute the origin shift
    compute_origin_shift(global_map_bbox_on_ref_frame, reference_map.rows, origin_shift_x_cells, origin_shift_y_cells);

    // 3. Compute the global affine transformation matrix
    cv::Mat affine_ref_to_global, affine_align_to_global;
    compute_global_affine(global_map_bbox_on_ref_frame, affine_align_to_ref, affine_ref_to_global, affine_align_to_global);

    // 4. Warp each map into the global coordinate system
    // Compute the size of the global map (bounding box is in pixel coordinates)
    int global_map_h = global_map_bbox_on_ref_frame.y_max - global_map_bbox_on_ref_frame.y_min + 1;
    int global_map_w = global_map_bbox_on_ref_frame.x_max - global_map_bbox_on_ref_frame.x_min + 1;
    // Create canvas for the two maps in the global coordinate system, and initialize them to unknown color
    cv::Mat ref_on_global(global_map_h, global_map_w, reference_map.type(), cv::Scalar(UNKNOWN_COLOR));
    cv::Mat align_on_global(global_map_h, global_map_w, align_map.type(), cv::Scalar(UNKNOWN_COLOR));
    // Warp the two maps to the global coordinate system
    cv::warpAffine(reference_map, ref_on_global, affine_ref_to_global, ref_on_global.size(), 
                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(UNKNOWN_COLOR));
    cv::warpAffine(align_map, align_on_global, affine_align_to_global, align_on_global.size(),
                    cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(UNKNOWN_COLOR));
    // 5. Merge the two maps into the final map
    cv::Mat final_merged_map = generate_merged_map(ref_on_global, align_on_global);
    // 6. Compute the acceptance index
    acceptance_index = compute_acceptance_index(ref_on_global, align_on_global);

    // Return the final merged map
    return final_merged_map;
}