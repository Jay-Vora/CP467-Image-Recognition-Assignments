import cv2

def main():
    #read the image
    img = cv2.imread('images/lena.tif', cv2.IMREAD_GRAYSCALE)

    #apply opencv averaging filter
    avg_img = cv2.blur(img, (3, 3))

    #save the image
    cv2.imwrite('images/t2a.tif', avg_img)

    #show the image
    cv2.imshow('lena_image', img)
    cv2.imshow('t2a', avg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
