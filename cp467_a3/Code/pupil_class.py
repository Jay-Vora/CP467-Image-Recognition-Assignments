import numpy as np
import cv2

class PupilDetection():

    def __init__(self, img_path):
        """
        Initializes the iris and pupil detection class with the path to
        the image and setting up the variables
        """
        self._img_path = img_path # The path for the input image
        self._img = None # The image itself
        self._original_img = None # For drawing the circle at the end for pupil and iris 
        self._pupil = None  # coordinates of the pupil in the format [x, y, radius]

    def load_image(self):
        """
        Loading the initial images from the input images
        also detect if image is not fed
        """
        self._img = cv2.imread(self._img_path, cv2.IMREAD_GRAYSCALE)
        self._original_img = cv2.imread(self._img_path)

        if self._img is None:
            raise ValueError("Image not found or unable to load")


    def blur_image(self):
        """
        Need to reduce the noise from the image so using
        GaussianBlur to blur the image for more accurate edge detection.
        """
        if self._img is not None:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(self._img)
            
            # Apply median blur to reduce noise while preserving edges
            self._img = cv2.medianBlur(clahe_image, 7)
        
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
        
        # Apply Canny edge detection
        self._img = cv2.Canny(self._img, 100, 180, apertureSize=3) #(40,60), (15, 50)

        # Apply Hough Circle Transform 
        circles = cv2.HoughCircles(self._img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=30, minRadius=20, maxRadius=60)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self._pupil = circles[0]  # Take the first detected circle (x, y, radius)
            return self._pupil
        else:
            raise ValueError("No pupil detected.")
        
    def detect(self):
        """
        detects the pupil in the image by loading the image, blurring it.
        """
        if (self._img_path is not None):
            self.load_image()
            self.blur_image()
            return self.detect_pupil()
    
        else:
            raise ValueError("Image file could not be loaded.")