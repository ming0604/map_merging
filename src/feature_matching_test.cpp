#include <ros/ros.h>
#include "map_merging/map_merging_tool.h"  // Adjust path according to your package structure
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    // Initialize ROS node 
    ros::init(argc, argv, "feature_matching_test");
    ros::NodeHandle nh("~");

    // Parameter check: two image file paths are required as input
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <map1_image> <map2_image>" << endl;
        return -1;
    }

    string map1Path = argv[1];
    string map2Path = argv[2];

    // Read the two map images
    cv::Mat map1 = cv::imread(map1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat map2 = cv::imread(map2Path, cv::IMREAD_GRAYSCALE);
    if (map1.empty() || map2.empty())
    {
        cerr << "Error: Cannot read image " << map1Path << " or " << map2Path << endl;
        return -1;
    }

    // Create a feature detector and BFMatcher
    // For SIFT use NORM_L2 with crossCheck enabled
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    // Alternatively, to use AKAZE or ORB, uncomment the following line:
    // cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create();
    // cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    ROS_INFO("Feature detector type: %s", detector->getDefaultName().c_str());

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
    // For AKAZE or ORB,use NORM_HAMMING:
    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    
    // Create a MapMergingTool object, passing in the detector and matcher
    MapMergingTool tool(detector, matcher);

    // Perform feature detection
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    tool.detect_features(map1, kp1, desc1);
    tool.detect_features(map2, kp2, desc2);

    // Draw the detected features on each image
    cv::Mat map1Features = tool.draw_features(map1, kp1);
    cv::Mat map2Features = tool.draw_features(map2, kp2);

    cv::imshow("Map1 Features", map1Features);
    cv::imshow("Map2 Features", map2Features);

    // Perform feature matching
    vector<cv::DMatch> matches;
    tool.match_features(desc1, desc2, matches);
    cout << "[INFO] Number of matches: " << matches.size() << endl;

    // Draw the matching results (here all matches are drawn; you can modify to show only the top N matches if needed)
    cv::Mat matchImg = tool.draw_matches(map1, kp1, map2, kp2, matches);
    cv::imshow("Match Result", matchImg);

    cv::waitKey(0);
    cv::destroyAllWindows();

    /*
    //test opencv version
    std::cout << cv::getBuildInformation() << std::endl;
    
    */

    return 0;
}



