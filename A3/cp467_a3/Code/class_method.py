# Numpy is needed because OpenCV images in python are actually numpy arrays.
import numpy as np
import cv2

class IrisPupilDetection():

    def __init__(self, img_path):
        """
        Initializes the iris and pupil detection class with the path to
        the image and setting up the variables
        """
        self._img_path = img_path # The path for the input image
        self._img = None # The image itself
        self._original_img = None # For drawing the circle at the end for pupil and iris 
        self._pupil = None 
        self._iris = None
        self._is_blurred = False # Tracking the status of the image's blurriness 

    def load_image(self):
        """
        Loading the initial images from the input images
        also detect if image is not fed
        """
        self._img = cv2.imread(self._img_path, cv2.IMREAD_GRAYSCALE)
        self._original_img = cv2.imread(self._img_path)

        if self._img is None:
            raise ValueError("Image not found or unable to load")

    def show(self, img):

        cv2.imshow("function", img)


    def blur_image(self):
        """
        Need to reduce the noise from the image so using
        GaussianBlur to blur the image for more accurate edge detection.
        """
        if self._img is not None:
            self._img = cv2.GaussianBlur(self._img, (9,9), 0) # probably need to change values (7,7) , 1.5
            self._is_blurred = True # Update the status once the image's been blurred
            self.show(self._img)
        
        else:
            raise ValueError("Image not loaded. Call load_image() in order to blur the image!")
    
    def detect_pupil(self):
        """
        Canny Edge detection and Hough Circle Tranform can be used 
        to detect edges and then detecting the inner circle cv2.Canny 
        and cv2.HoughCircles.
        """
        if self._img is None:
             raise ValueError("Image not loaded. Call load_image() in order to proceed!")

        # Need to check whether the image is blurred or not
        if not self._is_blurred:
            print("Blurring the image before applying the Canny Edge detection!")
            self.blur_image()
        
        # Apply Canny edge detection
        self._img = cv2.Canny(self._img, 20, 40, apertureSize=3) #(40,60), (15, 50)
        self.show(self._img)

        # Apply Hough Circle Transform 
        circles = cv2.HoughCircles(self._img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=20, maxRadius=60)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self._pupil = circles[0]  # Take the first detected circle (x, y, radius)
            return self._pupil
        else:
            raise ValueError("No pupil detected.")

    def draw_pupil(self):
        """
        
        """
        if self._img is not None:
            output_img = self._original_img.copy()
            x, y, radius = self._pupil

            #draw the circle around the detected  pupil green
            cv2.circle(output_img, (x,y), radius, (0, 255, 0), 2)


            cv2.imshow("Pupil Detection", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            raise ValueError("Pupil not detected. Call detect_pupil() before drawing.")
    
    def detect_iris(self):


        
detector = IrisPupilDetection("Input_Images/iris1.tif")
detector.load_image()
detector.detect_pupil()
detector.draw_pupil()