import cv2
import numpy as np

def marr_hildreth_edge_detector(img, sigma, mask_size, threshold):
    # applying gaussian smoothing filter
    blurred_img = cv2.GaussianBlur(img, (mask_size, mask_size), sigma)

    # applying Laplacian operator to smoothed image
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)

    # height and width of the image
    height, width = laplacian.shape

    # detecting zero-crossings (edges) with value 255       
    zero_crossings = np.zeros((height, width), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            #  Check for zero crossing
            if (laplacian[y-1, x] < 0 and laplacian[y+1, x] > 0) or \
               (laplacian[y, x-1] < 0 and laplacian[y, x+1] > 0):
                # Compute strength of the zero-crossing


                strength = abs(laplacian[y-1, x] + laplacian[y+1, x])
                # Mark zero-crossing if above threshold
                if strength > threshold:
                    zero_crossings[y, x] = 255
        
            
    
    return zero_crossings

def main():
    #read the image
    lena_img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #applying Marr-Hildreth edge detection
    edges = marr_hildreth_edge_detector(lena_img, 1.0, 5,10)

    #save the result
    cv2.imwrite('images/t3a.tif', edges)

    #display the result
    cv2.imshow('Original Image', lena_img)
    cv2.imshow('Marr-Hildreth Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
