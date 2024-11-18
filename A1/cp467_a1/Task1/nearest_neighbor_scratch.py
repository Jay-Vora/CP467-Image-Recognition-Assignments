import cv2
import numpy as np


def nearest_neighbor_interpolation_scratch(input_img, scale_factor_x, scale_factor_y):

    #height, width of orignal image
    height = len(input_img)
    width = len(input_img[0])

    #new height and width based on the scale factor for both dimensions
    new_width = int(width * scale_factor_x)
    new_height = int(height * scale_factor_y)

    #empty image array for the scaled image
    scaled_image = np.zeros([new_height, new_width], dtype=np.uint8)

    #main algorithm where the action happens 
    for y in range(new_height):
        for x in range(new_width):

            #calculate the coordinates
            new_x_cord = int(x / scale_factor_x + 0.5)
            new_y_cord = int(y / scale_factor_y + 0.5)

            #ensuring that coordinates do not go out of bound
            if new_x_cord >= width:
                new_x_cord = width -1 #because starts from 0 to 255

            if new_y_cord >= height:
                new_y_cord = height - 1

            scaled_image[y,x] = input_img[new_y_cord, new_x_cord]

    
    return scaled_image

    
#first task is to scale down the image to half from both dimensions.

#to read the image, it takes two paras - the path and the flag value
lena_img = cv2.imread("images/lena.tif", cv2.IMREAD_GRAYSCALE)

#setting the scale factors
scale_down_x = 0.5
scale_down_y = 0.5

lena_nearest_scratch_down = nearest_neighbor_interpolation_scratch(lena_img, scale_down_x, scale_down_y)

#displaying the scaled down image
cv2.imshow("Scaled down Image", lena_nearest_scratch_down)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_nearest_scratch_down.tif", lena_nearest_scratch_down)

#dispalying the scaled up image
scale_up_x = 2.0
scale_up_y = 2.0

lena_nearest_scratch = nearest_neighbor_interpolation_scratch(lena_nearest_scratch_down, scale_up_x, scale_up_y)

#displaying the scaled up image
cv2.imshow("Scaled Up Image", lena_nearest_scratch)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_nearest_scratch.tif", lena_nearest_scratch)