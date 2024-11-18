import numpy as np
import cv2

#convolve the image with the mask
def convolve(img, mask):
    #height and width of the image
    height, width = img.shape

    #height and width of the mask
    mask_height, _ = mask.shape

    #getting the padding value for out of bounds 
    padding = mask_height // 2

    #padding the image
    img = np.pad(img, ((padding, padding), (padding, padding)), 'constant')

    #define the new image
    new_img = np.zeros((height, width))

    #convolve the image with the mask
    for y in range(padding, height+padding):
        for x in range(padding, width+padding):
            #get the neighborhood
            neighborhood = img[y-padding:y+padding+1, x-padding:x+padding+1]

            #convolve the neighborhood with the mask
            new_img[y-padding, x-padding] = np.sum(neighborhood * mask)
    
    return new_img

def sobel_filter(img):
    #defining the sobel masks since we are already given the size so just hardcoding the values
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #convolving the image with the sobel masks
    sobel_x_img = convolve(img, sobel_x)
    sobel_y_img = convolve(img, sobel_y)

    #calculating the magnitude of the gradient
    magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)
    #print(np.max(magnitude)*255)

    #normalizing the magnitude and scaling back to 0-255 range
    magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
    # print(magnitude)
    return magnitude

def main(): 
    #read the image
    lena_img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply the sobel filter
    sobel_img = sobel_filter(lena_img)

    #save the image
    cv2.imwrite('images/t1c.tif', sobel_img)

    #show the image
    cv2.imshow("lena_image", lena_img)
    cv2.imshow('t1c', sobel_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    



    
