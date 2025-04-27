#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "map_merging/map_merger.h"


using namespace std;

int main(int argc, char** argv)
{
    // Initialize ROS node
    ros::init(argc, argv, "map_merging");
    ros::NodeHandle nh("~");  // private node handle for loading parameters from the launch file
    // Load parameters
    string feature_type;
    string maps_folder_path;
    string ref_map_name;
    string output_folder_path;
    string merged_map_name;
    float ratio_thr;
    int free_color_int, occ_color_int, unknown_color_int;
    uchar free_color, occ_color, unknown_color; // expected color that used to convert the original map color

    nh.param<std::string>("feature_type", feature_type, "AKAZE");
    nh.param<std::string>("maps_folder_path", maps_folder_path, "/home/mingzhun/lab_localization/src/map_merging/maps/");
    nh.param<std::string>("reference_map_name", ref_map_name, "map1");
    nh.param<std::string>("output_folder_path", output_folder_path, "/home/mingzhun/lab_localization/src/map_merging/merging_result/");
    nh.param<std::string>("merged_map_name", merged_map_name, "merged_map");
    nh.param("ratio_test_threshold", ratio_thr, 0.7f);
    // ROS can not directly load uchar type parameters, so we need to load them as int and then cast them to uchar
    nh.param("free_color_expected", free_color_int, 255);
    nh.param("occ_color_expected", occ_color_int, 0);
    nh.param("unknown_color_expected", unknown_color_int, 127);
    free_color = static_cast<uchar>(free_color_int);
    occ_color = static_cast<uchar>(occ_color_int);
    unknown_color = static_cast<uchar>(unknown_color_int);

    // Create the feature detector and matcher
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

            /*
            //FLANN based matcher
            cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
            cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
            indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
            */
        }
        else if(feature_type == "ORB")
        {   
            // For AKAZE or ORB,use NORM_HAMMING:
            detector = cv::ORB::create(5000);
            //matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true); // crossCheck enabled
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); // crossCheck disabled

            /*
            //FLANN based matcher
            cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
            cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
            indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
            */
        }
        else
        {
            ROS_ERROR("Error: Invalid feature type. Please choose from SIFT, AKAZE, or ORB.");
            return -1;
        }

        // Create a MapMerger object
        MapMerger map_merger(detector, matcher, 0.7);

        // 1. Load maps from the specified folder
        map_merger.load_maps(maps_folder_path, ref_map_name);
        // 2. Merge the maps and save the merged map
        map_merger.merge_maps(output_folder_path, merged_map_name, free_color, occ_color, unknown_color);
        ROS_INFO("Map merging completed successfully");
        return 0;
}