import matplotlib.pyplot as plt
import cv2

# Load the images
img1 = cv2.imread('images/t1a.tif', cv2.IMREAD_GRAYSCALE)  # First image
img2 = cv2.imread('images/t2a.tif', cv2.IMREAD_GRAYSCALE)  # Second image
img3 = cv2.imread('images/t1b.tif', cv2.IMREAD_GRAYSCALE)  # Third image   
img4 = cv2.imread('images/t2b.tif', cv2.IMREAD_GRAYSCALE)  # Fourth image
img5 = cv2.imread('images/t1c.tif', cv2.IMREAD_GRAYSCALE)  # Fifth image
img6 = cv2.imread('images/t2c.tif', cv2.IMREAD_GRAYSCALE)  # Sixth image

# Set up the plot layout to accommodate 6 images (3 rows, 2 columns)
plt.figure(figsize=(10, 15))

# Display the first image
plt.subplot(3, 2, 1)  # (rows, columns, index)
plt.imshow(img1, cmap='gray')
plt.title('My own averaging filter implementation')
plt.axis('off')  # Hide the axis

# Display the second image
plt.subplot(3, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title('OpenCV averaging filter implementation')
plt.axis('off')

# Display the third image
plt.subplot(3, 2, 3)
plt.imshow(img3, cmap='gray')
plt.title('My own gaussian filter implementation')
plt.axis('off')

# Display the fourth image
plt.subplot(3, 2, 4)
plt.imshow(img4, cmap='gray')
plt.title('OpenCV gaussian filter implementation')
plt.axis('off')

# Display the fifth image
plt.subplot(3, 2, 5)
plt.imshow(img5, cmap='gray')
plt.title('My own sobel filter implementation')
plt.axis('off')

# Display the sixth image
plt.subplot(3, 2, 6)
plt.imshow(img6, cmap='gray')
plt.title('OpenCV sobel filter implementation')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
