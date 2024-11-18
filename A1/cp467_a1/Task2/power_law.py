import cv2
import numpy as np

#constant
MAX_SIZE = 255.0
C = 1 # c is set to 1 for simplicity
#normalize pixel values of the image for power-law trnasformation range for [0,1]

def normalize(input_img):

    height = len(input_img)
    width = len(input_img[0])

    normalized_img = np.zeros([height, width], dtype=np.float32)
    for y in range(height):
        for x in range(width):
            
            normalized_img[y][x] = input_img[y][x] / MAX_SIZE 
            
    return normalized_img

def power_law_transformation(input_img, gamma):

    height = len(input_img)
    width = len(input_img[0])

    transformed_img = np.zeros([height, width], dtype=np.float32)

    for y in range(height):
        for x in range(width):

            transformed_img[y][x] = C * (input_img[y][x])**gamma

    return transformed_img


def denormalize(input_img):

    height = len(input_img)
    width = len(input_img[0])

    denormalized_img = np.zeros([height, width], dtype=np.uint8)
    for y in range(height):
        for x in range(width):

            denormalized_img[y][x] = input_img[y][x] * MAX_SIZE

            #if out of bounds from the max size set it back to max size and to zero if lower than 0
            if denormalized_img[y][x] > MAX_SIZE:
                denormalized_img[y][x] = MAX_SIZE

            if denormalized_img[y][x] < 0:
                denormalized_img[y][x] = 0
    
    #print(denormalized_img)
    return denormalized_img

cameraman_img = cv2.imread("images/cameraman.tif", cv2.IMREAD_GRAYSCALE)

cameraman_power_normalized = normalize(cameraman_img)

cameraman_transformed = power_law_transformation(cameraman_power_normalized, 1.2)

cameraman_power = denormalize(cameraman_transformed)

#displaying the scaled down image
cv2.imshow("Power Transformed Image", cameraman_power)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the negative image
cv2.imwrite("cameraman_power.tif", cameraman_power)
