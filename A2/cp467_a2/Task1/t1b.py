import numpy as np
import cv2

def gaussian_mask(mask_size=7, sigma=1):

    #calculate the center of the mask
    center = mask_size // 2

    #creating a grid for mask's (x,y) coordinates
    grid = np.arange(-center, center+1)
    #print(grid)
    
    #making the grid 2D from getting the limit
    x, y = np.meshgrid(grid, grid)

    #calculating the gaussian filter, also the constant part is generally ignored
    filtered_mask = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    #normalizing the gaussian filter
    filtered_mask = filtered_mask / np.sum(filtered_mask)

    return filtered_mask
    

def gaussian_filter(img, mask_size=7, sigma=1):

    #height and width of the image
    height, width = img.shape

    #define the mask as ones
    mask = np.ones((mask_size, mask_size))

    #getting the padding value for out of bounds 
    padding = mask_size // 2

    #add zero padding to the image to get the edges of the image
    #constant default value is 0    
    img = np.pad(img, ((padding, padding), (padding, padding)), 'constant')

    #create a new image with the same size as the original image
    new_img = np.zeros((height, width))

    #getting the gaussian mask
    mask = gaussian_mask(mask_size, sigma)

    #iterate through the image
    for y in range(padding, height+padding):
        for x in range(padding, width+padding):

            #get the neighborhood of the pixel
            neighborhood = img[y-padding:y+padding+1, x-padding:x+padding+1]

            #applying alreay computed gaussian mask
            new_img[y-padding, x-padding] = np.sum(neighborhood * mask)

    #convert the image to uint8
    new_img = new_img.astype(np.uint8)

    return new_img  

def main():

    #read the image
    lena_img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply the gaussian filter
    gaussian_img = gaussian_filter(lena_img)

    #save the image
    cv2.imwrite('images/t1b.tif', gaussian_img)

    #show the image
    cv2.imshow("lena_image", lena_img)
    cv2.imshow('t1b', gaussian_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






    


