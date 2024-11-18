import numpy as np
import cv2

def averaging_filter(img, mask_size):

    #height and width of the image
    height, width = img.shape

    #define the mask as ones
    mask = np.ones((mask_size, mask_size))

    #add zero padding to the image to get the edges of the image
    #constant default value is 0
    img = np.pad(img, ((1, 1), (1, 1)), 'constant')

    #create a new image with the same size as the original image
    new_img = np.zeros((height, width))

    #iterate through the image
    for y in range(1, height+1):
        for x in range(1, width+1):

            #get the neighborhood of the pixel
            neighborhood = img[y-1:y+2, x-1:x+2]

            #first sum the neighborhood and then divide by the number of elements in the mask
            new_img[y-1, x-1] = np.sum(neighborhood * mask) / (mask_size**2)

    #convert the image to uint8
    new_img = new_img.astype(np.uint8)

    return new_img

def main():

    #read image
    lena_img = cv2.imread("images/lena.tif", cv2.IMREAD_GRAYSCALE)

    #apply the averaging filter
    t1a = averaging_filter(lena_img, 3)

    #save the image
    cv2.imwrite("images/t1a.tif", t1a)

    #show the image
    cv2.imshow("lena_image", lena_img)
    cv2.imshow("t1a", t1a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


