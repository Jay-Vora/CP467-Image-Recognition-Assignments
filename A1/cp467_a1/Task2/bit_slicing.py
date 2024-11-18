import cv2
import numpy as np


def bit_plane_slicing(input_img, bit_plane):
        

    height = len(input_img)
    width = len(input_img[0])

    bit_plane_img = np.zeros([height, width], dtype=np.uint8)

    for y in range(height):
        for x in range(width):

            bit_value = (input_img[y][x] >> bit_plane) & 1 # to extract the specified bit for the plane

            bit_plane_img[y][x] = bit_value * 255

    return bit_plane_img


cameraman_img = cv2.imread("images/cameraman.tif", cv2.IMREAD_GRAYSCALE)

cameraman_b1 = bit_plane_slicing(cameraman_img, 0)
cameraman_b2 = bit_plane_slicing(cameraman_img, 1)
cameraman_b3 = bit_plane_slicing(cameraman_img, 2)
cameraman_b4 = bit_plane_slicing(cameraman_img, 3)
cameraman_b5 = bit_plane_slicing(cameraman_img, 4)
cameraman_b6 = bit_plane_slicing(cameraman_img, 5)
cameraman_b7 = bit_plane_slicing(cameraman_img, 6)
cameraman_b8 = bit_plane_slicing(cameraman_img, 7)

cv2.imwrite('cameraman_b1.tif', cameraman_b1)
cv2.imshow("Bit Plane 1", cameraman_b1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b2.tif', cameraman_b2)
cv2.imshow("Bit Plane 2", cameraman_b2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b3.tif', cameraman_b3)
cv2.imshow("Bit Plane 3", cameraman_b3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b4.tif', cameraman_b4)
cv2.imshow("Bit Plane 4", cameraman_b4)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b5.tif', cameraman_b5)
cv2.imshow("Bit Plane 5", cameraman_b5)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b6.tif', cameraman_b6)
cv2.imshow("Bit Plane 6", cameraman_b6)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('cameraman_b7.tif', cameraman_b7)
cv2.imshow("Bit Plane 7", cameraman_b7)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save and display the bit plane
cv2.imwrite('cameraman_b8.tif', cameraman_b8)
cv2.imshow("Bit Plane 8", cameraman_b8)
cv2.waitKey(0)
cv2.destroyAllWindows()
