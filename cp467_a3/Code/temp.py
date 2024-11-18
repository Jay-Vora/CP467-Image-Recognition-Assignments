import cv2
import numpy as np

# Load the uploaded image
image_path = 'Input_images/iris1.tif'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert grayscale to BGR to allow colored circle overlay
color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray_image, (7, 7), 1.5)
cv2.imshow("blurred img", blurred)
# Edge Detection with Canny
edges = cv2.Canny(blurred, 50, 150)

# Save edge map for verification
edge_map_path = 'Edge_Maps/edge_map_iris1.tif'
cv2.imshow("edge_map", edges)
#cv2.imwrite(edge_map_path, edges)

# Circle Detection with Hough Transform
# Detect circles for the pupil (smaller circle)
pupil_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                 param1=100, param2=30, minRadius=20, maxRadius=60)

# Detect circles for the iris (larger circle)
iris_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                param1=100, param2=30, minRadius=70, maxRadius=120)

# Overlay Detected Circles
if pupil_circles is not None:
    pupil_circles = pupil_circles[0, :].astype("int")
    for (x, y, r) in pupil_circles:
        cv2.circle(color_image, (x, y), r, (0, 255, 0), 2)  # Green for pupil

if iris_circles is not None:
    iris_circles = iris_circles[0, :].astype("int")
    for (x, y, r) in iris_circles:
        cv2.circle(color_image, (x, y), r, (255, 0, 0), 2)  # Blue for iris

# Save the segmented image with circles overlay for verification
segmented_image_path = 'Output_images/segmented_iris1.tif'
cv2.imwrite(segmented_image_path, color_image)

edge_map_path, segmented_image_path
