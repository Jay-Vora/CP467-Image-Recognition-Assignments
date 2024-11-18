import cv2

def main():
    #read the image
    img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply opencv gaussian filter with sigma = 1 and mask size = 7
    gaussian_img = cv2.GaussianBlur(img, (7, 7), 1)

    #save the image
    cv2.imwrite('images/t2b.tif', gaussian_img)

    #show the image
    cv2.imshow('lena_image', img)
    cv2.imshow('t2b', gaussian_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
