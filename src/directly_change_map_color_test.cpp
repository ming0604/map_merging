#include <ros/ros.h>
#include "map_merging/map_merging_tool.h"  
#include <opencv2/opencv.hpp>

#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

#include <nav_msgs/OccupancyGrid.h>
#include "nav_msgs/GetMap.h"


using namespace std;

// Funtion to find the file name without extension
string getFileName(const string& path)
{
    size_t start = path.find_last_of("/\\");
    size_t end = path.find_last_of(".");
    return path.substr(start + 1, end - start - 1);
}

// Funtion to convert the map image color
cv::Mat convert_map_image_color(const cv::Mat& raw_map, uchar free_color, uchar occ_color, uchar unknown_color)
{   
    // Ensure the input map is in grayscale
    CV_Assert(raw_map.type() == CV_8UC1);

    cv::Mat coverted_map = raw_map.clone();
    uchar pixel_value;
    // Iterate through each pixel and set the expected color
    // original color of raw map image: 0(black): occupied space, 254(almost white): free space, 205(light gray): unknown space 
    // convert the original color to the expected color
    for(int i=0; i<coverted_map.rows; i++)
    {
        for(int j=0; j<coverted_map.cols; j++)
        {
            pixel_value = coverted_map.at<uchar>(i, j);
            if(pixel_value == 0) // occupied space
            {
                coverted_map.at<uchar>(i, j) = occ_color;
            }
            else if(pixel_value == 254) // free space
            {
                coverted_map.at<uchar>(i, j) = free_color;
            }
            else if(pixel_value == 205) // unknown space
            {
                coverted_map.at<uchar>(i, j) = unknown_color;
            }
            else //error
            {
                cerr << "Error: Invalid pixel value in map image: " << (int)pixel_value << endl;
                // terminate the program
                exit(-1);
            }
        }
    }
    return coverted_map;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "directly_change_map_color_test");
    // Create a node handle
    ros::NodeHandle nh("~");

    // Get the parameters
    string map1Path;
    string map2Path; 
    string save_dir;
    string feature_type;
    string matcher_type;
    string map1_file_name;
    string map2_file_name;
    bool use_stitching_module;
    nh.param<string>("map1_image_path", map1Path, "/home/mingzhun/lab_localization/src/navigation/amcl/amcl_data/maps/20241218_vive_mapping_test3.pgm");
    nh.param<string>("map2_image_path", map2Path, "/home/mingzhun/lab_localization/src/navigation/amcl/amcl_data/maps/20250216_vive_mapping_20m_test3.pgm");
    nh.param<string>("output_folder_path", save_dir, "/home/mingzhun/Pictures/map_merging_output_image/");
    nh.param<string>("feature_type", feature_type, "AKAZE");
    nh.param<string>("matcher_type", matcher_type, "BF");
    nh.param("use_stitching_module", use_stitching_module, false);


    // Read the two map images
    cv::Mat raw_map1 = cv::imread(map1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat raw_map2 = cv::imread(map2Path, cv::IMREAD_GRAYSCALE);
    // Print the images  area
    //cout << "map2 mat area:" << map2(cv::Rect(300, 500, 100, 100)) << endl; 
    if (raw_map1.empty() || raw_map2.empty())
    {
        cerr << "Error: Cannot read image " << map1Path << " or " << map2Path << endl;
        return -1;
    }
    // Print the names of the input maps
    map1_file_name = getFileName(map1Path);
    map2_file_name = getFileName(map2Path);
    cout << "Map1: " << map1_file_name << endl;
    cout << "Map2: " << map2_file_name << endl;
    
    // Convert the raw map color to the expected color
    // free space: black(0), occupied space: dark gray(100), unknown space: white(255), like the method of directly using map topic
    
    uchar free_color = 0; 
    uchar occ_color = 100; 
    uchar unknown_color = 255; 
    /*
    // free space: white(255), occupied space: black(0), unknown space: gray(127) 
    uchar free_color = 255; 
    uchar occ_color = 0; 
    uchar unknown_color = 127; 
    */
    cv::Mat map1 = convert_map_image_color(raw_map1, free_color, occ_color, unknown_color);
    cv::Mat map2 = convert_map_image_color(raw_map2, free_color, occ_color, unknown_color);

    // display the map after color conversion
    cv::namedWindow("Converted Map1", cv::WINDOW_NORMAL);
    cv::namedWindow("Converted Map2", cv::WINDOW_NORMAL);
    cv::imshow("Converted Map1", map1);
    cv::imshow("Converted Map2", map2);

    if(use_stitching_module)
    {   
        cv::Ptr<cv::Feature2D> finder;
        cv::Ptr<cv::detail::AffineBestOf2NearestMatcher> matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();
        cv::Ptr<cv::detail::AffineBasedEstimator> estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
        cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
    
        // Use the stitching module to detect features 
        // 1) Detect features using a FeaturesFinder from stitching/detail
        if (feature_type == "SIFT")
        {
            finder = cv::SIFT::create();
        }
        else if (feature_type == "ORB")
        {
            finder = cv::ORB::create(5000);
        }
        else if (feature_type == "AKAZE")
        {
            finder = cv::AKAZE::create();
        }
        else
        {
            cerr << "Error: Invalid feature type. Please choose from SIFT, AKAZE, or ORB." << endl;
            return -1;
        }
        // Print the feature type
        ROS_INFO("Feature type: %s", finder->getDefaultName().c_str());
        // Detect features
        cv::detail::ImageFeatures features1, features2;
        cv::detail::computeImageFeatures(finder, map1, features1);
        features1.img_idx = 0;
        cv::detail::computeImageFeatures(finder, map2, features2);
        features2.img_idx = 1;
        // Print the number of features detected
        cout << "Number of features in map1: " << features1.keypoints.size() << endl;
        cout << "Number of features in map2: " << features2.keypoints.size() << endl;
        // Draw the detected features on each image
        cv::Mat map1_with_features;
        cv::Mat map2_with_features;
        cv::drawKeypoints(map1, features1.keypoints, map1_with_features, cv::Scalar(0, 255, 0));
        cv::drawKeypoints(map2, features2.keypoints, map2_with_features, cv::Scalar(0, 255, 0));
        cv::imshow("Map1 Features", map1_with_features);
        cv::imshow("Map2 Features", map2_with_features);

        // 2. Match features using AffineBestOf2NearestMatcher
        std::vector<cv::detail::ImageFeatures> features_vec;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        features_vec.push_back(features1);
        features_vec.push_back(features2);
        (*matcher)(features_vec, pairwise_matches);
        
        // Print the number of matches
        for(size_t i = 0; i < pairwise_matches.size(); i++)
        {
            cout << "Number of matches in pairwise_matches[" << i << "]: " << pairwise_matches[i].matches.size() << endl;
        }
        // Print the number of inliers
        for(size_t i = 0; i < pairwise_matches.size(); i++)
        {
            cout << "Number of inliers in pairwise_matches[" << i << "]: " << pairwise_matches[i].num_inliers << endl;
        }
        cv::detail::leaveBiggestComponent(features_vec, pairwise_matches, 0.3f);
        /*
        // 3. Run the affine estimator
        vector<cv::detail::CameraParams> cameras;
        if(!(*estimator)(features_vec, pairwise_matches, cameras))
        {
            ROS_ERROR("Estimator failed");
            return -1;
        }
        // Set the R matrix to in camera params to CV_32F type, because the adjuster requires it
        for (size_t i = 0; i < cameras.size(); i++)
        {
            cameras[i].R.convertTo(cameras[i].R, CV_32F);
        }

        // 4. Run the bundle adjuster
        //set the adjuster's confidence threshold
        adjuster->setConfThresh(0.3f);
        if(!(*adjuster)(features_vec, pairwise_matches, cameras))
        {
            ROS_ERROR("Bundle adjuster failed");
            return -1;
        }
        // Print the final affine transformation matrix
        // Affine transformation matrix is a 2x3 matrix
        cv::Mat map1_affine = cameras[0].R(cv::Rect(0, 0, 2, 3));
        cv::Mat map2_affine = cameras[1].R(cv::Rect(0, 0, 2, 3));
        cout << "map1's affine matrix: " << map1_affine << endl;
        cout << "map2's affine matrix: " << map2_affine << endl;
        */
    }
    else
    {
        // Create the feature detector and BFMatcher
        cv::Ptr<cv::Feature2D> detector;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if(feature_type == "SIFT")
        {   
            // Create SIFT detector
            detector = cv::SIFT::create();
            // Create matcher
            if(matcher_type == "BF")
            {   
                // For SIFT use NORM_L2 
                //matcher = cv::BFMatcher::create(cv::NORM_L2, true); // crossCheck enabled
                matcher = cv::BFMatcher::create(cv::NORM_L2, false); // crossCheck disabled
                ROS_INFO("Using BF matcher with NORM_L2");
            }
            else if(matcher_type == "FLANN")
            {
                // FLANN based matcher
                matcher = cv::FlannBasedMatcher::create();
                ROS_INFO("Using FLANN-based matcher");
            }
            else
            {
                ROS_ERROR("Error: Invalid matcher type. Please choose from BF or FLANN.");
                return -1;
            }
                
        }
        else if(feature_type == "AKAZE")
        {   
            // Create AKAZE detector
            detector = cv::AKAZE::create();
            // Create matcher
            if(matcher_type == "BF")
            {   
                // For AKAZE or ORB,use NORM_HAMMING:
                //matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // crossCheck enabled
                matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck disabled
                ROS_INFO("Using BF matcher with NORM_HAMMING");
            }
            else if(matcher_type == "FLANN")
            {
                // FLANN based matcher
                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
                indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
                ROS_INFO("Using FLANN-based matcher");
            }
            else
            {
                ROS_ERROR("Error: Invalid matcher type. Please choose from BF or FLANN.");
                return -1;
            }
        }
        else if(feature_type == "ORB")
        {   
            // Create ORB detector
            detector = cv::ORB::create(5000);
            // Create matcher
            if(matcher_type == "BF")
            {   
                // For AKAZE or ORB,use NORM_HAMMING:
                //matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // crossCheck enabled
                matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck disabled
                ROS_INFO("Using BF matcher with NORM_HAMMING");
            }
            else if(matcher_type == "FLANN")
            {
                //FLANN based matcher
                cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
                cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
                indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
                matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
                ROS_INFO("Using FLANN-based matcher");
            }
            else
            {
                ROS_ERROR("Error: Invalid matcher type. Please choose from BF or FLANN.");
                return -1;
            }
        }
        else
        {
            ROS_ERROR("Error: Invalid feature type. Please choose from SIFT, AKAZE, or ORB.");
            return -1;
        }
        // Print the feature type
        ROS_INFO("Feature type: %s", detector->getDefaultName().c_str());
        
        // Create a MapMergingTool object, passing in the detector and matcher
        float ratio_threshold = 0.7;
        //float ratio_threshold = 0.8;    
        MapMergingTool tool(detector, matcher, ratio_threshold);

        // Perform feature detection
        vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        tool.detect_features(map1, kp1, desc1);
        tool.detect_features(map2, kp2, desc2);

        // Draw the detected features on each image
        cv::Mat map1_with_features = tool.draw_features(map1, kp1);
        cv::Mat map2_with_features = tool.draw_features(map2, kp2);
        // Print the number of detected features
        cout << "Number of features in map1: " << kp1.size() << endl;
        cout << "Number of features in map2: " << kp2.size() << endl;
        // Display the images with features
        cv::imshow("Map1 Features", map1_with_features);
        cv::imshow("Map2 Features", map2_with_features);

        /**/
        // Save the images with features

        string map1_with_features_path = save_dir + map1_file_name + "_" + feature_type + "_features_test.png";
        string map2_with_features_path = save_dir + map2_file_name + "_" + feature_type + "_features_test.png";
        cv::imwrite(map1_with_features_path, map1_with_features);
        cv::imwrite(map2_with_features_path, map2_with_features);
        cout << "Saved images with features to " << save_dir << endl;
        
        
        // Perform feature matching
        vector<cv::DMatch> matches;
        tool.match_features(desc1, desc2, matches);
        cout << "Number of matches: " << matches.size() << endl;

        // Draw the matching results (here all matches are drawn; you can modify to show only the top N matches if needed)
        cv::Mat match_img = tool.draw_matches(map1, kp1, map2, kp2, matches);
        cv::imshow("Match Result", match_img);

        /*
        // Save the matching result image
        //string match_img_path = save_dir + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_matches_crosscheck.png";
        //string match_img_path = save_dir + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_matches_no_crosscheck.png";
        string match_img_path = save_dir + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_matches_ratio_test.png";
        cv::imwrite(match_img_path, match_img);
        cout << "Saved matching result image to " << save_dir << endl;
        */

        // Compute the affine transformation matrix
        cv::Mat inliers;
        int num_inliers;
        cv::Mat affine = tool.compute_affine_matrix(kp1, kp2, matches, inliers, num_inliers);
        cout << "Number of inliers: " << num_inliers << endl;
        // Draw the inlier matches
        cv::Mat inlier_img = tool.draw_inlier_matches(map1, kp1, map2, kp2, matches, inliers);
        cv::imshow("Inlier Matches", inlier_img);

        /*
        // Save the inlier matches image
        string inlier_img_path = save_dir + "directly_change_map_color" + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_inlier_matches_.png";
        cv::imwrite(inlier_img_path, inlier_img);
        */

        // Draw the warped raw map2 using the computed affine matrix
        cv::Mat warped_map2;
        cv::warpAffine(raw_map2, warped_map2, affine, map1.size(),cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(255));
        cv::imshow("Warped Map2", warped_map2);
        // Use map1 as the base map, see how well map2 is overlaid on map1
        cv::Mat merge_test;
        cv::cvtColor(raw_map1, merge_test, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < merge_test .rows; y++) {
            for (int x = 0; x < merge_test .cols; x++) {
                uchar warped_map2_val = warped_map2.at<uchar>(y, x);
                uchar map1_val = raw_map1.at<uchar>(y, x);
                // If the warped map2 pixel is black and map1 pixel is not black, set the pixel to red (bad overlap)
                if (warped_map2_val == 0 && map1_val != 0) {
                    // Set the pixel to red
                    merge_test.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                }
            }
        }
        cv::imshow("Merging test Map", merge_test);

        /*
        // Save the merging test image
        string merge_test_path = save_dir + "directly_change_map_color_" + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_merge_test.png";
        cv::imwrite(merge_test_path, merge_test);
        */

    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}