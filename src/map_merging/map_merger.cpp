#include "map_merging/map_merger.h"

using namespace std;

MapMerger::MapMerger(cv::Ptr<cv::Feature2D> feature_detector, cv::Ptr<cv::DescriptorMatcher> feature_matcher, float ratio_thr):
merge_tool(feature_detector, feature_matcher, ratio_thr)
{
    ROS_INFO("Map merger constructed");

}

MapMerger::~MapMerger() {}


string MapMerger::getFileName(const string& path)
{
    size_t start = path.find_last_of("/\\");
    size_t end = path.find_last_of(".");
    return path.substr(start + 1, end - start - 1);
}

void MapMerger::load_map_data(const string& yaml_path, MapData& map_data)
{
    // Load the yaml file
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if(!fs.isOpened()) 
    {
        ROS_ERROR("Failed to open yaml file: %s", yaml_path.c_str());
        exit(-1);
    }
    // Read the map image path
    fs["image"] >> map_data.map_img_path;
    // Find the map name
    map_data.map_name = getFileName(map_data.map_img_path);
    // Read the map resolution
    fs["resolution"] >> map_data.resolution;
    // Read the map origin
    fs["origin"] >> map_data.origin;
    // Close the file
    fs.release();

    // Load the map image
    map_data.raw_image = cv::imread(map_data.map_img_path, cv::IMREAD_GRAYSCALE);
    if(map_data.raw_image.empty()) 
    {
        ROS_ERROR("Failed to load map image: %s", map_data.map_img_path.c_str());
        exit(-1);
    }
}


// Load all the map yamls from the specified folder
void MapMerger::load_maps(const string& maps_folder_path, const string& ref_map_name)
{   

    DIR* dir = opendir(maps_folder_path.c_str());
    // Check if the directory exists
    if(dir == nullptr) 
    {
        ROS_ERROR("Directory %s does not exist", maps_folder_path.c_str());
        exit(-1);
    }
    // Read all the .yaml files in the directory
    struct dirent* entry;
    string file_name;

    entry = readdir(dir);   // read the first entry 
    while(entry != nullptr) 
    {   
        // Get the file name string
        file_name = entry->d_name;
        // Check if the file is a normal file
        if(entry->d_type != DT_REG) 
        {
            entry = readdir(dir);
            continue;
        }
        // Check if the file is a .yaml file
        if(file_name.find(".yaml") != string::npos) 
        {
            // Construct the full path to the yaml file
            string yaml_path = maps_folder_path + file_name;
            // Load the map data from the yaml file
            MapData map_data;
            load_map_data(yaml_path, map_data);
            // Add the map data to the vector of maps
            maps.push_back(map_data);
            ROS_INFO("Loaded map: %s", map_data.map_name.c_str());
        }
        entry = readdir(dir);
    }
    closedir(dir);
    // Print the number of maps loaded
    ROS_INFO("Loaded %zu maps from %s", maps.size(), maps_folder_path.c_str());

    // Find the reference map and set it as the first map
    bool found_ref_map = false;
    for(size_t i=0; i<maps.size(); i++)
    {
        if(maps[i].map_name == ref_map_name)
        {   
            found_ref_map = true;
            // Swap the reference map with the first map
            swap(maps[0], maps[i]);
            ROS_INFO("Reference map of first merging: %s", maps[0].map_name.c_str());
            break;
        }
    }

    // Print maps index
    if(!found_ref_map)
    {
        ROS_ERROR("Reference map %s not found in the folder %s", ref_map_name.c_str(), maps_folder_path.c_str());
        exit(-1);
    }
    else
    {
        ROS_INFO("Load maps successfully");
        for(size_t i=0; i<maps.size(); i++)
        {
            ROS_INFO("Map %zu: %s", i, maps[i].map_name.c_str());
        }
    }

}

void MapMerger::convert_map_color(const cv::Mat& raw_image, cv::Mat& converted_image, const uchar free_color, const uchar occ_color, const uchar unknown_color)
{
    // Ensure the raw map image is in grayscale
    CV_Assert(raw_image.type() == CV_8UC1);
    // Convert the map image color
    converted_image = raw_image.clone();
    uchar pixel_value;
    // Iterate through each pixel and set the expected color
    // original color of raw map image: 0(black): occupied space, 254(almost white): free space, 205(light gray): unknown space 
    // convert the original color to the expected color
    for(int y=0; y<converted_image.rows; y++)
    {
        for(int x=0; x<converted_image.cols; x++)
        {
            pixel_value = converted_image.at<uchar>(y, x);
            if(pixel_value == 0) // occupied space
            {
                converted_image.at<uchar>(y, x) = occ_color;
            }
            else if(pixel_value == 254) // free space
            {
                converted_image.at<uchar>(y, x) = free_color;
            }
            else if(pixel_value == 205) // unknown space
            {
                converted_image.at<uchar>(y, x) = unknown_color;
            }
            else //error
            {
                ROS_ERROR("Error: Invalid pixel value when converting color, pixel value: %u"
                    , static_cast<unsigned int>(pixel_value));
                exit(-1);
            }
        }
    }
    
}  

void MapMerger::merge_maps(const string& output_folder_path, const string& merged_map_name, const uchar free_color, const uchar occ_color, const uchar unknown_color)
{   
    // Check if there are at least two maps to merge
    if(maps.size() < 2) 
    {
        ROS_ERROR("Not enough maps to merge. At least two maps are required.");
        exit(-1);
    }

    // Convert the map color to the expected color
    for(size_t i=0; i<maps.size(); i++)
    {
        convert_map_color(maps[i].raw_image, maps[i].converted_image, free_color, occ_color, unknown_color);
    }

    // Set the reference map as the first map at the beginning
    merge_tool.detect_features(maps[0].converted_image, maps[0].features, maps[0].descriptors);
    MapData ref_map = maps[0];
    ref_map.map_name = "map1"; // give the reference map a name
    int origin_shift_x_cells_total = 0;
    int origin_shift_y_cells_total = 0;
    int origin_shift_x_cells = 0;
    int origin_shift_y_cells = 0;
    double acceptance_index = 0.0;
    for(size_t i=1; i<maps.size(); i++)
    {   
        // Detect features for the current map
        merge_tool.detect_features(maps[i].converted_image, maps[i].features, maps[i].descriptors);
        // Match features between the reference map and the current map
        vector<cv::DMatch> matches;
        merge_tool.match_features(ref_map.descriptors, maps[i].descriptors, matches);
        ROS_INFO("Number of matches between reference map %s and map %zu: %zu",ref_map.map_name.c_str(), i+1, matches.size());
        // Draw the matching results
        cv::Mat match_img = merge_tool.draw_matches(ref_map.converted_image, ref_map.features, 
                                                    maps[i].converted_image, maps[i].features, matches);
        // Save the matching result image
        string match_img_path = output_folder_path + ref_map.map_name + "_" + "map"+ to_string(i) + "_matches.png";
        cv::imwrite(match_img_path, match_img);

        // Compute the affine transformation matrix
        cv::Mat inliers;
        cv::Mat affine = merge_tool.compute_affine_matrix(ref_map.features, maps[i].features, matches, inliers);
        // Draw the inlier matches
        cv::Mat inlier_img = merge_tool.draw_inlier_matches(ref_map.converted_image, ref_map.features,
                                                            maps[i].converted_image, maps[i].features,
                                                            matches, inliers);
        // Save the inlier matches image
        string inlier_img_path = output_folder_path + ref_map.map_name + "_" + "map"+ to_string(i) + "_inlier_matches.png";
        cv::imwrite(inlier_img_path, inlier_img);
        
        // Merge the two maps using the computed affine transformation matrix
        cv::Mat merged_map = merge_tool.merge_maps(ref_map.raw_image, maps[i].raw_image, affine, origin_shift_x_cells,
                                                    origin_shift_y_cells, acceptance_index);
        // Print the acceptance index
        ROS_INFO("Acceptance index between reference map %s and map %zu: %f", ref_map.map_name.c_str(), i+1, acceptance_index);                                    
        // Update the total origin shift cells
        origin_shift_x_cells_total += origin_shift_x_cells;
        origin_shift_y_cells_total += origin_shift_y_cells;
        // Set the merged map as the reference map for the next iteration
        ref_map.raw_image = merged_map.clone();
        convert_map_color(ref_map.raw_image, ref_map.converted_image, free_color, occ_color, unknown_color);
        merge_tool.detect_features(ref_map.converted_image, ref_map.features, ref_map.descriptors);
        // give the merged map a name
        ref_map.map_name = ref_map.map_name + "+" + to_string(i+1);    
    }
    ROS_INFO("All maps merged successfully");
    MapData final_merged_map = ref_map;
    final_merged_map.map_name = merged_map_name;
    // Comepute the new origin of the merged map in world coordinates
    double reso = final_merged_map.resolution;
    double new_origin_x = maps[0].origin[0] - (origin_shift_x_cells_total * reso);
    double new_origin_y = maps[0].origin[1] - (origin_shift_y_cells_total * reso);
    final_merged_map.origin[0] = new_origin_x;
    final_merged_map.origin[1] = new_origin_y;
    ROS_INFO("New origin of the merged map in ROS type: (%f, %f)", final_merged_map.origin[0], final_merged_map.origin[1]);

    // Save the final merged map
    string merged_map_path = output_folder_path + merged_map_name + ".pgm";
    cv::imwrite(merged_map_path, final_merged_map.raw_image);

    // Save the final merged map yaml file
    string merged_map_yaml_path = output_folder_path + merged_map_name + ".yaml";
    cv::FileStorage fs(merged_map_yaml_path, cv::FileStorage::WRITE);
    if(!fs.isOpened()) 
    {
        ROS_ERROR("Failed to open yaml file: %s", merged_map_yaml_path.c_str());
        exit(-1);
    }
    // Write the map image path
    fs << "image" << merged_map_path;
    // Write the map resolution
    fs << "resolution" << final_merged_map.resolution;
    // Write the map origin 
    fs << "origin" << final_merged_map.origin;
    // Write other information
    fs << "negate" << 0;
    fs << "occupied_thresh" << 0.65;
    fs << "free_thresh" << 0.196;
    // Close the file
    fs.release();

    ROS_INFO("Merged map saved to %s", output_folder_path.c_str());

    // Create a new txt file to save the all maps name and the merged map name
    string merged_map_txt_path = output_folder_path + "maps_info" + ".txt";
    ofstream maps_info_file;
    maps_info_file.open(merged_map_txt_path);
    // Check if the file is opened successfully
    if(!maps_info_file.is_open()) 
    {
        ROS_ERROR("Failed to open file: %s", merged_map_txt_path.c_str());
        exit(-1);
    }
    // Write the maps name to the file
    for(size_t i=0; i<maps.size(); i++)
    {
        maps_info_file << "map" << (i+1) << ": " << maps[i].map_name << endl;
    }
    // Write the merged map name to the file
    maps_info_file << "merged_map: " << merged_map_name << endl;
    // Close the file
    maps_info_file.close();

    ROS_INFO("Info of all maps is saved to %s", merged_map_txt_path.c_str());
}