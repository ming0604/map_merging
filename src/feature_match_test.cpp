#include <ros/ros.h>
#include "map_merging/map_merging_tool.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    // 初始化 ROS 節點（可選，如果僅測試 OpenCV 也可以不使用 ROS）
    ros::init(argc, argv, "feature_matching_test");
    ros::NodeHandle nh("~");

    // 參數檢查：需要兩個影像檔案路徑作為輸入
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <map1_image> <map2_image>" << std::endl;
        return -1;
    }

    std::string map1Path = argv[1];
    std::string map2Path = argv[2];

    // 讀取兩張地圖影像 (必須是灰階)
    cv::Mat map1 = cv::imread(map1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat map2 = cv::imread(map2Path, cv::IMREAD_GRAYSCALE);
    if (map1.empty() || map2.empty())
    {
        std::cerr << "Error: 無法讀取影像 " << map1Path << " 或 " << map2Path << std::endl;
        return -1;
    }

    // 建立 feature 檢測器與 BFMatcher (SIFT 使用 NORM_L2, crossCheck=true)
    //cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create();
    //cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    // 建立 MapMergingTool 物件，並傳入檢測器與匹配器
    MapMergingTool tool(detector, matcher);

    // 執行特徵偵測
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    tool.detect_features(map1, kp1, desc1);
    tool.detect_features(map2, kp2, desc2);

    // 使用 draw_features 顯示特徵點圖
    cv::Mat map1Features = tool.draw_features(map1, kp1);
    cv::Mat map2Features = tool.draw_features(map2, kp2);

    cv::imshow("Map1 特徵點", map1Features);
    cv::imshow("Map2 特徵點", map2Features);

    // 執行特徵匹配
    std::vector<cv::DMatch> matches;
    tool.match_features(desc1, desc2, matches);
    std::cout << "[INFO] 匹配點數量: " << matches.size() << std::endl;

    // 繪製匹配結果 (此處畫出全部匹配，可依需要改成僅顯示前 N 個)
    cv::Mat matchImg = tool.draw_matches(map1, kp1, map2, kp2, matches);
    cv::imshow("匹配結果", matchImg);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}