#Find the negative of the image and store the output as “cameraman_negative”.
import cv2
import numpy as np


def negative_implementation(input_img):

    height = len(input_img)
    width = len(input_img[0])

    negative_img = np.zeros([height, width], dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            new_value = 255 - input_img[y][x]
            negative_img[y][x] = new_value 


    return negative_img



#to read the image, it takes two paras - the path and the flag value
cameraman_img = cv2.imread("images/cameraman.tif", cv2.IMREAD_GRAYSCALE)

cameraman_negative = negative_implementation(cameraman_img)

#displaying the scaled down image
cv2.imshow("Negative Image", cameraman_negative)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the negative image
cv2.imwrite("cameraman_negative.tif", cameraman_negative)
