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


// funtion to transform the map data to cv::Mat
cv::Mat RosMap2cvMat(const nav_msgs::OccupancyGrid& map)
{
    int width = map.info.width;
    int height = map.info.height;
    // directly set the data to cv::Mat as unsigned char
    cv::Mat map_image(height, width, CV_8UC1, const_cast<int8_t*>(map.data.data()));
    // flip the map image to make it in the same orientation as the ros type grid map
    cv::Mat map_image_flipped;
    cv::flip(map_image, map_image_flipped, 0);
    return map_image_flipped;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "directly_use_map_topic_test");
    // create a node handle
    ros::NodeHandle nh("~");

    // get the parameters
    string map1Path;
    string map2Path; 
    string save_dir;
    string feature_type;
    string map1_file_name;
    string map2_file_name;
    bool use_stitching_module;
    nh.param<std::string>("map1_image_path", map1Path, "/home/mingzhun/lab_localization/src/navigation/amcl/amcl_data/maps/20241218_vive_mapping_test3.pgm");
    nh.param<std::string>("map2_image_path", map2Path, "/home/mingzhun/lab_localization/src/navigation/amcl/amcl_data/maps/20250216_vive_mapping_20m_test3.pgm");
    nh.param<std::string>("output_folder_path", save_dir, "/home/mingzhun/Pictures/map_merging_output_image/");
    nh.param<std::string>("feature_type", feature_type, "AKAZE");
    nh.param("use_stitching_module", use_stitching_module, false);
    // Print the names of the input maps
    map1_file_name = getFileName(map1Path);
    map2_file_name = getFileName(map2Path);
    cout << "Map1: " << map1_file_name << endl;
    cout << "Map2: " << map2_file_name << endl;

    // create map client 
    ros::ServiceClient map1_client = nh.serviceClient<nav_msgs::GetMap>("/robot1/static_map");
    ros::ServiceClient map2_client = nh.serviceClient<nav_msgs::GetMap>("/robot2/static_map");
    // create map service
    nav_msgs::GetMap map1_srv;
    nav_msgs::GetMap map2_srv;
    // create map variable
    nav_msgs::OccupancyGrid map1;
    nav_msgs::OccupancyGrid map2;

    // call the map service to get the map
    if (map1_client.call(map1_srv))
    {   
        map1 = map1_srv.response.map;
        // Print map information
        ROS_INFO("Received map1: width=%d, height=%d, resolution=%.3f\n", map1.info.width, map1.info.height, map1.info.resolution);
    }
    else
    {
        ROS_ERROR("Failed to call service /robot1/static_map");
        return 1;
    }
    if (map2_client.call(map2_srv))
    {   
        map2 = map2_srv.response.map;
        // Print map information
        ROS_INFO("Received map2: width=%d, height=%d, resolution=%.3f\n", map2.info.width, map2.info.height, map2.info.resolution);
    }
    else
    {
        ROS_ERROR("Failed to call service /robot2/static_map");
        return 1;
    }

    // Convert the map to cv::Mat
    cv::Mat map1_image_mat = RosMap2cvMat(map1);
    cv::Mat map2_image_mat = RosMap2cvMat(map2);
    // display the map after conversion to cv::Mat
    cv::imshow("Map1", map1_image_mat);
    cv::imshow("Map2", map2_image_mat);

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
        cv::detail::computeImageFeatures(finder, map1_image_mat, features1);
        features1.img_idx = 0;
        cv::detail::computeImageFeatures(finder, map2_image_mat, features2);
        features2.img_idx = 1;
        // Print the number of features detected
        cout << "Number of features in map1: " << features1.keypoints.size() << endl;
        cout << "Number of features in map2: " << features2.keypoints.size() << endl;
        // Draw the detected features on each image
        cv::Mat map1_with_features;
        cv::Mat map2_with_features;
        cv::drawKeypoints(map1_image_mat, features1.keypoints, map1_with_features, cv::Scalar(0, 255, 0));
        cv::drawKeypoints(map2_image_mat, features2.keypoints, map2_with_features, cv::Scalar(0, 255, 0));
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
            // For SIFT use NORM_L2 
            detector = cv::SIFT::create();
            //matcher = cv::BFMatcher::create(cv::NORM_L2, true); // crossCheck enabled
            matcher = cv::BFMatcher::create(cv::NORM_L2, false); // crossCheck disabled
        }
        else if(feature_type == "AKAZE")
        {   
            // For AKAZE or ORB,use NORM_HAMMING:
            detector = cv::AKAZE::create();
            //matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // crossCheck enabled
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck disabled
        }
        else if(feature_type == "ORB")
        {   
            // For AKAZE or ORB,use NORM_HAMMING:
            detector = cv::ORB::create(5000);
            //matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // crossCheck enabled
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck disabled
        }
        else
        {
            cerr << "Error: Invalid feature type. Please choose from SIFT, AKAZE, or ORB." << endl;
            return -1;
        }
        // Print the feature type
        ROS_INFO("Feature type: %s", detector->getDefaultName().c_str());

        // Create a MapMergingTool object, passing in the detector and matcher
        float ratio_threshold = 0.7;    
        MapMergingTool tool(detector, matcher, ratio_threshold);

        // Perform feature detection
        vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        tool.detect_features(map1_image_mat, kp1, desc1);
        tool.detect_features(map2_image_mat, kp2, desc2);

        // Draw the detected features on each image
        cv::Mat map1_with_features = tool.draw_features(map1_image_mat, kp1);
        cv::Mat map2_with_features = tool.draw_features(map2_image_mat, kp2);
        // Print the number of detected features
        cout << "Number of features in map1: " << kp1.size() << endl;
        cout << "Number of features in map2: " << kp2.size() << endl;
        // Display the images with features
        cv::imshow("Map1 Features", map1_with_features);
        cv::imshow("Map2 Features", map2_with_features);

        /*
        // Save the images with features
        string map1_with_features_path = save_dir + "directly_use_map_topic_" + map1_file_name + "_" + feature_type + "_features.png";
        string map2_with_features_path = save_dir + "directly_use_map_topic_" + map2_file_name + "_" + feature_type + "_features.png";
        cv::imwrite(map1_with_features_path, map1_with_features);
        cv::imwrite(map2_with_features_path, map2_with_features);
        cout << "Saved images with features to " << save_dir << endl;
        */
        
        // Perform feature matching
        vector<cv::DMatch> matches;
        tool.match_features(desc1, desc2, matches);
        cout << "Number of matches: " << matches.size() << endl;

        // Draw the matching results (here all matches are drawn; you can modify to show only the top N matches if needed)
        cv::Mat match_img = tool.draw_matches(map1_image_mat, kp1, map2_image_mat, kp2, matches);
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
        cv::Mat inlier_img = tool.draw_inlier_matches(map1_image_mat, kp1, map2_image_mat, kp2, matches, inliers);
        cv::imshow("Inlier Matches", inlier_img);

        /*
        // Save the inlier matches image
        string inlier_img_path = save_dir + map1_file_name + "_" + map2_file_name + "_" + feature_type + "_inlier_matches_.png";
        cv::imwrite(inlier_img_path, inlier_img);
        */

    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}