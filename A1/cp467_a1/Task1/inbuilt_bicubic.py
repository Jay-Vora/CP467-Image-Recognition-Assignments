import cv2

#to read the image, it takes two paras - the path and the flag value
lena_img = cv2.imread("images/lena.tif", cv2.IMREAD_GRAYSCALE)

height, width = lena_img.shape

#scale factors for horizontal and vertical axis
scale_x = 0.5
scale_y = 0.5

new_width = int(width * scale_x) #since the horizontal plane refers to x-axis due to top-down approach
new_height = int(height * scale_y) # vertical plane refers to y-axis

#in-built function for nearest neighbor interpolation
lena_bicubic_cv_down = cv2.resize(lena_img, (new_width, new_height),fx=scale_x, fy=scale_y ,interpolation=cv2.INTER_CUBIC)

#displaying the scaled down image
cv2.imshow("Scaled down Image", lena_bicubic_cv_down)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_bicubic_cv_down.tif", lena_bicubic_cv_down)

#dispalying the scaled up image
scale_up_x = 2.0
scale_up_y = 2.0

scaled_up_width = int(new_width * scale_up_x)
scaled_up_height = int(new_height * scale_up_y)

lena_bicubic_cv = cv2.resize(lena_bicubic_cv_down, (scaled_up_width, scaled_up_height),fx=scale_up_x, fy=scale_up_y ,interpolation=cv2.INTER_CUBIC)

#displaying the scaled up image
cv2.imshow("Scaled Up Image", lena_bicubic_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

#save the scaled down image
cv2.imwrite("lena_bicubic_cv.tif", lena_bicubic_cv)
