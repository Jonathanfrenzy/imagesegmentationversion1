import cv2
import numpy as np


def mean_shift_segmentation(image):
    # Convert image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Apply mean shift filtering
    shifted_image = cv2.pyrMeanShiftFiltering(lab_image, 20, 40)

    return shifted_image


def region_growing_segmentation(image, seed_point, threshold):

    # Create a mask to store the segmented region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Set up the region growing parameters
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    height, width = image.shape[:2]
    visited = np.zeros((height, width), dtype=np.uint8)

    # Set the seed point and initial color
    seed_color = image[seed_point[0], seed_point[1]]
    queue = [seed_point]

    # Perform region growing
    while queue:
        current_point = queue.pop(0)
        current_color = image[current_point[0], current_point[1]]

        # Check if the current point is within the threshold range
        if np.sum(np.abs(current_color - seed_color)) <= threshold:
            # Add the point to the mask and mark it as visited
            mask[current_point[0], current_point[1]] = 255
            visited[current_point[0], current_point[1]] = 1

            # Explore the neighbors of the current point
            for neighbor in neighbors:
                neighbor_point = (current_point[0] + neighbor[0], current_point[1] + neighbor[1])

                # Check if the neighbor is within the image bounds and not visited
                if (0 <= neighbor_point[0] < height and 0 <= neighbor_point[1] < width
                        and visited[neighbor_point[0], neighbor_point[1]] == 0):
                    queue.append(neighbor_point)
                    visited[neighbor_point[0], neighbor_point[1]] = 1

    return mask

def combine_ms_and_rg(image, ms, rg):
    regions = ms(image)
    new_regions = rg(regions)
    return new_regions

image = cv2.imread('image2.jpg')

# Perform Mean Shift segmentation
shifted_image = mean_shift_segmentation(image)

# Perform Region Growing segmentation
seed_point = (100, 100)  # Define the seed point for region growing
threshold = 20  # Define the threshold for region growing
segmented_region = region_growing_segmentation(shifted_image, seed_point, threshold)


#perform joint operation
combine_ms_and_rg(image, shifted_image, region_growing_segmentation)

# Display the segmented image and region
cv2.imshow('Mean Shift Segmentation', shifted_image)
cv2.imshow('Region Growing Segmentation', segmented_region)
cv2.imshow('joint image segmentationxregion growing', combine_ms_and_rg)
cv2.waitKey(0)
cv2.destroyAllWindows()