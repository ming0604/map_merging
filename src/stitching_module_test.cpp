#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <iostream>
#include <string>

using namespace std;

// Function to extract file name without extension
string getFileName(const string& path)
{
    size_t start = path.find_last_of("/\\");
    size_t end = path.find_last_of(".");
    return path.substr(start + 1, end - start - 1);
}

int main(int argc, char** argv)
{   
    // Initialize ROS node
    ros::init(argc, argv, "stitching_module_test");
    ros::NodeHandle nh("~");

    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <map1_image_path> <map2_image_path> <output_folder_path> <feature_type> "  << endl;
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
    
    cv::Ptr<cv::Feature2D> finder;
    cv::Ptr<cv::detail::AffineBestOf2NearestMatcher> matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();
    cv::Ptr<cv::detail::AffineBasedEstimator> estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();

    // 1) Detect features using a FeaturesFinder from stitching/detail
    if (feature_type == "SIFT")
    {
        finder = cv::SIFT::create();
    }
    else if (feature_type == "ORB")
    {
        finder = cv::ORB::create();
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

    return 0;
}
