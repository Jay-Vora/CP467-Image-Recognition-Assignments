import cv2
import numpy as np
from pupil_class import PupilDetection

class IrisDetection(PupilDetection):
    def __init__(self, img_path, pupil):
        """
        Initializes the iris detection with the path to the image and
        pupil information for guided iris detection.
        """
        super().__init__(img_path)
        self.load_image() # intially load the image from pupil class
        self._pupil = pupil  # Pupil data in the format [x, y, radius]
        self._iris = None  # Iris detection result

    def detect_iris(self):
        """
        Detects the iris circle using the pupil location and radius as a reference.
        Applies Hough Circle Transform with adjusted parameters for iris.
        """
        if self._img is None:
            raise ValueError("Image not loaded. Call load_image() before detecting the iris!")
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(self._img)
            
        # Apply median blur to reduce noise while preserving edges
        self._img = cv2.medianBlur(clahe_image, 7)


        # Use Canny for edge detection, just as in pupil detection
        edges = cv2.Canny(self._img, 40, 60)
        cv2.imwrite(f"Edge_Maps/{self._img_path[17]}_edge.tif", edges)
        
        # Set the search range for the iris based on the pupil radius
        min_radius = int(self._pupil[2] * 1.5)  # Iris is typically larger than pupil
        max_radius = int(self._pupil[2] * 3.0)
        
        # Hough Circle Transform for iris detection
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=100, param2=30,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self._iris = circles[0]  # Take the first detected circle for the iris
            return self._iris
        else:
            raise ValueError("No iris detected.")

    def draw_iris(self):
        """
        Draws both the detected pupil and iris circles on the original image.
        """
        if self._img is not None:
            output_img = self._original_img.copy()
            
            # Draw the pupil circle in green
            px, py, pradius = self._pupil
            cv2.circle(output_img, (px, py), pradius, (0, 255, 0), 2)
            
            # Draw the iris circle in blue
            if self._iris is not None:
                ix, iy, iradius = self._iris
                cv2.circle(output_img, (ix, iy), iradius, (255, 0, 0), 2)
            
            cv2.imwrite(f"Output_Images/{self._img_path[17]}_output.tif", output_img)
            cv2.imshow("Pupil and Iris Detection", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            raise ValueError("No iris detected. Call detect_iris() before drawing.")



def main():
    image_paths = [
        "Input_Images/iris1.tif",
        "Input_Images/iris2.tif",
        "Input_Images/iris3.tif",
        "Input_Images/iris4.tif",
        "Input_Images/iris5.tif"
    ]

    for img_path in image_paths:
        # First, detect the pupil
        pupil_detector = PupilDetection(img_path)
        pupil = pupil_detector.detect()  
        # Use the detected pupil to initialize and detect the iris
        iris_detector = IrisDetection(img_path, pupil)
        iris_detector.detect_iris()
        iris_detector.draw_iris()


if __name__ == "__main__":
    main()   