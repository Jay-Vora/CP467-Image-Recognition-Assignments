import cv2
import numpy as np

def main():
    #read the image
    lena_img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply canny edge detector to the image
    edges = cv2.Canny(lena_img, 50, 100)

    #save the image
    cv2.imwrite('images/t3b.tif', edges)

    #display the image
    cv2.imshow('Original Image', lena_img)
    cv2.imshow('Canny Edge Detector', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()  
    
