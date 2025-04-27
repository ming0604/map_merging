#ifndef MAP_MERGER_H
#define MAP_MERGER_H

#include <ros/ros.h>
#include "map_merging_tool.h"
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>


typedef struct MapData
{   
    std::string map_name; // name of the map
    std::string map_img_path;
    cv::Mat raw_image;
    cv::Mat converted_image;  //map image after color conversion
    std::vector<double> origin;
    double resolution;

    std::vector<cv::KeyPoint> features;
    cv::Mat descriptors;

    // Define the =operator overload to copy the map data
    MapData& operator=(const MapData& other)
    {
        if(this != &other) // Avoid self-assignment
        {
            map_name = other.map_name;
            map_img_path = other.map_img_path;
            raw_image = other.raw_image.clone();
            converted_image = other.converted_image.clone();
            origin = other.origin;
            resolution = other.resolution;
            features = other.features;
            descriptors = other.descriptors.clone();
        }
        return *this;
    }
}MapData; 

class MapMerger
{
    public:
        MapMerger(cv::Ptr<cv::Feature2D> detector, cv::Ptr<cv::DescriptorMatcher> matcher, float ratio_thr);
        virtual ~MapMerger();
        void load_maps(const std::string& maps_folder_path, const std::string& ref_map_name);
        void convert_map_color(const cv::Mat& raw_image, cv::Mat& converted_image, const uchar free_color, const uchar occ_color, const uchar unknown_color);
        void merge_maps(const std::string& output_folder_path, const std::string& merged_map_name, const uchar free_color, const uchar occ_color, const uchar unknown_color);
        std::string getFileName(const std::string& path);
        
    private:
        MapMergingTool merge_tool;
        std::vector<MapData> maps; // list of maps to be merged

        void load_map_data(const std::string& yaml_path, MapData& map_data);
};

#endif