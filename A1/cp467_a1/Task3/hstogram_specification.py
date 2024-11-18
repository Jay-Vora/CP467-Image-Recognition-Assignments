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

    #normalize for getting the pixel intensities in the range of 0 to 1
    cdf_normalized = cdf /cdf.max()


    return cdf_normalized

def create_mapping(input_cdf, target_cdf):

    #mapping array to track the matching intensity level
    mapping = np.zeros(256, dtype=np.uint8)

    for input in range(256):
        closest_target = np.argmin(np.abs(input_cdf[input] - target_cdf))
        mapping[input] = closest_target

    return mapping

def apply_mapping(input_img, mapping):

    #apply the mapping to the input_img
    height, width = input_img.shape

    output_img = np.zeros([height, width], dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):

            output_img[y][x] = mapping[input_img[y][x]]

    return output_img

input_img = cv2.imread('images/chest_x-ray1.jpeg', cv2.IMREAD_GRAYSCALE)
target_img = cv2.imread('images/chest_x-ray2.jpeg', cv2.IMREAD_GRAYSCALE)

input_histogram = histogram_calculation(input_img)
target_histogram = histogram_calculation(target_img)

input_cdf = calculate_cdf(input_histogram)
target_cdf = calculate_cdf(target_histogram)

mapping = create_mapping(input_cdf, target_cdf)

output_img = apply_mapping(input_img, mapping)

cv2.imwrite('chest_x-ray3.jpeg', output_img)
cv2.imshow('chest_x-ray3', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()