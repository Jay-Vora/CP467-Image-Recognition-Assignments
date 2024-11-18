import cv2
import numpy as np

def main():
    #read the image
    img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply opencv sobel filter
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    #calculating the magnitude of the gradient using inbuilt function
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    #normalizing the magnitude to 0-255 range
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    #converting to uint8
    magnitude = np.uint8(magnitude)

    #save the image
    cv2.imwrite('images/t2c.tif', magnitude)

    #show the image
    cv2.imshow('lena_image', img)
    cv2.imshow('t2c', magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


