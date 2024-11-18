import cv2
import numpy as np


def bilinear_interpolation_scratch(input_img, scale_factor_x, scale_factor_y):

    #height, width of orignal image
    height = len(input_img)
    width = len(input_img[0])

    #new height and width based on the scale factor for both dimensions
    new_width = int(width * scale_factor_x)
    new_height = int(height * scale_factor_y)

    #empty image array for the scaled image
    scaled_image = np.zeros([new_height, new_width], dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):

            new_x_cord = x / scale_factor_x 
            new_y_cord = y / scale_factor_y 

            # identifying nearing 4 pixels
            x1 = int(new_x_cord)            
            x2 = min(x1 + 1, width - 1)  # Make sure x2 stays within bounds
            y1 = int(new_y_cord)
            y2 = min(y1 + 1, height - 1)  # Make sure y2 stays within bounds
        
            #calculating the fractional distance 
            dx = new_x_cord - x1
            dy = new_y_cord - y1


            #x-axis top row interpolation and bottom interpolation
            top_interpolation = (1 - dx) * input_img[y1, x1] + dx * input_img[y1, x2]
            bottom_interpolation = (1 - dx) * input_img[y2, x1] + dx * input_img[y2, x2]

            #y-axis interpolation of the above two
            final_value = (1 - dy) * top_interpolation + dy * bottom_interpolation
            final_value = np.clip(final_value,0,255)

            scaled_image[y, x] = int(final_value)


    return scaled_image

#to read the image, it takes two paras - the path and the flag value
lena_img = cv2.imread("images/lena.tif", cv2.IMREAD_GRAYSCALE)

height, width = lena_img.shape

#scale factors for horizontal and vertical axis
scale_x = 0.5
scale_y = 0.5

lena_bilinear_scratch_down = bilinear_interpolation_scratch(lena_img, scale_x, scale_y)

#displaying the scaled down image
cv2.imshow("Scaled down Image", lena_bilinear_scratch_down)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_bilinear_scratch_down.tif", lena_bilinear_scratch_down)

#dispalying the scaled up image
scale_up_x = 2.0
scale_up_y = 2.0

lena_bilinear_scratch = bilinear_interpolation_scratch(lena_bilinear_scratch_down, scale_up_x, scale_up_y)

#displaying the scaled up image
cv2.imshow("Scaled Up Image", lena_bilinear_scratch)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_bilinear_scratch.tif", lena_bilinear_scratch)