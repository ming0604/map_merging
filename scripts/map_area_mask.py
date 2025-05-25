#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import sys

# global variables
rectangles = []
map_rect_area_img = None
rect_start = None
drawing = False

def generate_masked_map(map_raw_img, map_yaml_path, output_map_img_path, output_map_yaml_path):
    global rectangles
    # Mask the selected rectangles areas to 205 (means unknown area in the ROS map)
    masked_map = map_raw_img.copy()
    for rect in rectangles:
        (x_min, y_min), (x_max, y_max) = rect
        # row is y, column is x in the image, tail +1 to include the last pixel
        masked_map[y_min:y_max+1, x_min:x_max+1] = 205
    cv2.imwrite(output_map_img_path, masked_map)
    print(f"Masked map image is saved to {output_map_img_path}")
    print(f"Size of the masked map: h:{masked_map.shape[0]}, w:{masked_map.shape[1]}")    

    # Save the masked map yaml file
    # Check if the yaml file exists
    if not os.path.exists(map_yaml_path):
        print(f"Error: The map's yaml file does not exist, please ensure the name of the yaml file is same as the map image file.")
        sys.exit(1)
    # Read the yaml file of the original map
    with open(map_yaml_path, 'r') as f:
        original_yaml_lines = f.readlines()
    #　Create a new yaml file for the masked map
    with open(output_map_yaml_path, 'w') as f:
        for line in original_yaml_lines:
            # Replace the image file path with the new masked map image file path 
            # Check if the line starts with "image:"
            if line.strip().startswith("image:"):      
                f.write(f"image: {output_map_img_path}\n")
            # Copy other lines as they are
            else:
                f.write(line)
    print(f"Masked map yaml file is saved to {output_map_yaml_path}")

    return masked_map

def draw_rectangles(event, x, y, flags, param):
    global rectangles, map_rect_area_img, rect_start, drawing
    # Left mouse button down to start drawing a rectangle area
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the starting point of the rectangle
        rect_start = (x, y)
        # Set the drawing flag to True
        drawing = True
    # Mouse move to choose the rectangle area for masking
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = map_rect_area_img.copy()
        # Draw the rectangle you are currently selecting
        cv2.rectangle(temp_img, rect_start, (x, y), (0, 0, 255), 2)
        cv2.imshow("Map for masking area selection", temp_img)
    # Left mouse button up to finish drawing this rectangle area
    elif event == cv2.EVENT_LBUTTONUP:
        # Set the drawing flag to False
        drawing = False
        # Store the rectangle into the list of rectangles
        x_start, y_start = rect_start
        # Find the minimum and maximum coordinates of the rectangle
        x_min = min(x_start, x)
        x_max = max(x_start, x)
        y_min = min(y_start, y) 
        y_max = max(y_start, y)

        # Ensure the rectangle coordinates are within the image bounds
        h, w = map_rect_area_img.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(h - 1, y_max)
        rectangles.append(((x_min, y_min), (x_max, y_max)))
        # Print the added rectangle coordinates
        print(f"Added selected area: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
        # Draw the rectangle on the map image
        # Update the  map rect area image with the new rectangle
        cv2.rectangle(map_rect_area_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.imshow("Map for masking area selection", map_rect_area_img)
def main():
    global rectangles, map_rect_area_img
    rospy.init_node("map_area_mask", anonymous=True)
    map_folder_path = rospy.get_param("~map_folder_path", "/home/mingzhun/lab_localization/src/map_merging/maps/area_mask_maps/")
    map_name = rospy.get_param("~map_name", "original_map")
    map_img_path = map_folder_path + map_name + ".pgm"
    map_yaml_path = map_folder_path + map_name + ".yaml"
    output_map_img_name = map_name + "_masked.pgm"
    output_map_img_path = map_folder_path + output_map_img_name
    output_map_yaml_name = map_name + "_masked.yaml"
    output_map_yaml_path = map_folder_path + output_map_yaml_name

    # Load the map image
    map_raw_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded map image from {map_img_path}")
    print(f"Size of the raw map: h:{map_raw_img.shape[0]}, w:{map_raw_img.shape[1]}")
    map_rect_area_img = map_raw_img.copy()
    # Convert the map_rect_area_img to BGR for displaying the selected rectangles in color
    map_rect_area_img = cv2.cvtColor(map_rect_area_img, cv2.COLOR_GRAY2BGR)
    
    # Create a window to display the map
    cv2.namedWindow("Map for masking area selection", cv2.WINDOW_NORMAL)
    # Set mouse callback function to handle rectangle area marking
    cv2.setMouseCallback("Map for masking area selection", draw_rectangles,)
    # Display the map image
    cv2.imshow("Map for masking area selection", map_rect_area_img)
    print("Click and drag to select rectangle areas you want to mask.")
    print("Press 's' to save the masked map.")

    while True:
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        # If 'q' is pressed, exit the loop
        if key == ord('s'):
            cv2.destroyWindow("Map for masking area selection")
            # Generate the masked map image and save it
            masked_map_img = generate_masked_map(map_raw_img, map_yaml_path, output_map_img_path, output_map_yaml_path)
            # Display the final masked map image
            cv2.namedWindow("Masked Map", cv2.WINDOW_NORMAL)
            cv2.imshow("Masked Map", masked_map_img)
            # Wait user to press any key to quit
            print("Masked map image is displayed, press any key to quit.")
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()