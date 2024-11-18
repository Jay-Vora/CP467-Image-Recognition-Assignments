import cv2
import numpy as np


def histogram_calculation(input_img):

    height = len(input_img)
    width = len(input_img[0])

    histogram_array = [0] * 256

    for y in range(height):
        for x in range(width):
            
            intensity = input_img[y][x]
            histogram_array[intensity] += 1

    return histogram_array

def calculate_cdf(histogram_array):

    #need to convert histogram array to numpy array for cdf
    hist_np = np.array(histogram_array)

    #calculate the cdf
    cdf = hist_np.cumsum()
    #print(cdf, cdf.min, cdf.max)

    #normalize for getting the pixel intensities in the range of 0 to 255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    #we want int values at the end, so converting float into int
    cdf_normalized = cdf_normalized.astype(np.uint8)

    return cdf_normalized

def histogram_equalization(input_img):

    #calculate histogram
    histogram = histogram_calculation(input_img)

    #calcuating cdf and normalizing it
    cdf_normalized = calculate_cdf(histogram)

    #applying the cdf normalized pixel values to the input image

    height = len(input_img)
    width = len(input_img[0])

    equalized_img = np.zeros([height, width], dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            equalized_img[y][x] = cdf_normalized[input_img[y][x]]

    return equalized_img

# Load the grayscale image
input_img = cv2.imread('images/einstein.tif', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
einstein_equalized = histogram_equalization(input_img)

# Save the result
cv2.imwrite('einstein_equalized.tif', einstein_equalized)
cv2.imshow("einstein equalized", einstein_equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()