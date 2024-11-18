import cv2


def detect_circles(image, kernal_size, sigma, upper_limit, lower_limit):

    # firstly, blur the image 
    gaussian_img = cv2.GaussianBlur(image, (kernal_size, kernal_size), sigma)

    cv2.imshow("bllurred img", gaussian_img)
    # apply canny to identify edges
    edge_map = cv2.Canny(gaussian_img, upper_limit, lower_limit)

    cv2.imshow("edge_map", edge_map)
    #cv2.imwrite("Edge_Maps/edge_map", edge_map)

    #detecting smaller circle
    pupil_circle = cv2.HoughCircles(gaussian_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=20,maxRadius=60)

     # Detect circles for the iris (larger circle)
    iris_circle = cv2.HoughCircles(gaussian_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=70, maxRadius=120)

      # Overlay Detected Circles
    if pupil_circle is not None:
        pupil_circle = pupil_circle[0, :].astype("int")
        for (x, y, r) in pupil_circle:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Green for pupil

    if iris_circle is not None:
        iris_circle = iris_circle[0, :].astype("int")
        for (x, y, r) in iris_circle:
            cv2.circle(image, (x, y), r, (255, 0, 0), 2)  # Blue for iris

    # Save the segmented image with circles overlay
    #cv2.imwrite("Output_images/segmented_img.tif", image)



def main():

    eye_image = cv2.imread('Input_Images/iris1.tif', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('image', eye_image)

    detect_circles(eye_image,7,1.5, 20, 60)

    color_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2BGR)
    #show the image
    cv2.imshow("segmenterd_img", eye_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()