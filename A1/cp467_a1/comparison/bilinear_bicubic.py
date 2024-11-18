import cv2
import matplotlib.pyplot as plt

# Load the images
bilinear_img = cv2.imread('lena_bilinear_cv.tif', cv2.IMREAD_GRAYSCALE)
bicubic_img = cv2.imread('lena_bicubic_cv.tif', cv2.IMREAD_GRAYSCALE)

# Plot the images side by side for comparison
plt.figure(figsize=(10, 5))

# Bilinear image
plt.subplot(1, 2, 1)
plt.imshow(bilinear_img, cmap='gray')
plt.title('Bilinear Interpolation')
plt.axis('off')

# Bicubic image
plt.subplot(1, 2, 2)
plt.imshow(bicubic_img, cmap='gray')
plt.title('Bicubic Interpolation')
plt.axis('off')

plt.tight_layout()
plt.show()