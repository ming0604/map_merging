#include <ros/ros.h>
#include "map_merging/map_merging_tool.h"  // Adjust path according to your package structure
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;

// Funtion to find the file name without extension
string getFileName(const string& path)
{
    size_t start = path.find_last_of("/\\");
    size_t end = path.find_last_of(".");
    return path.substr(start + 1, end - start - 1);
}

int main(int argc, char** argv)
{
    // Initialize ROS node 
    ros::init(argc, argv, "feature_matching_test");
    ros::NodeHandle nh("~");

    // Parameter check: two image file paths are required as input
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <map1_image_path> <map2_image_path> <output_folder_path> <feature_type> " << endl;
        return -1;
    }

    string map1Path = argv[1];
    string map2Path = argv[2];
    // Set the output directory
    string save_dir = argv[3];
    string feature_type = argv[4];
    string map1_file_name, map2_file_name;

    // Read the two map images
    cv::Mat map1 = cv::imread(map1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat map2 = cv::imread(map2Path, cv::IMREAD_GRAYSCALE);
    if (map1.empty() || map2.empty())
    {
        cerr << "Error: Cannot read image " << map1Path << " or " << map2Path << endl;
        return -1;
    }
    // Extract map filenames (without extension) for output naming
    map1_file_name = getFileName(map1Path);
    map2_file_name = getFileName(map2Path);
    // Print the names of the input maps
    cout << "Map1: " << map1_file_name << endl;
    cout << "Map2: " << map2_file_name << endl;
 

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
        detector = cv::ORB::create(3000);
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

    /*
    // Save the images with features
    string map1_with_features_path = save_dir + map1_file_name + "_" + feature_type + "_features.png";
    string map2_with_features_path = save_dir + map2_file_name + "_" + feature_type + "_features.png";
    cv::imwrite(map1_with_features_path, map1_with_features);
    cv::imwrite(map2_with_features_path, map2_with_features);
    cout << "Saved images with features to " << save_dir << endl;
    */
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
    cv::waitKey(0);
    cv::destroyAllWindows();

    /*
    //test opencv version
    std::cout << cv::getBuildInformation() << std::endl;
    
    */

    return 0;
}



