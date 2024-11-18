
import cv2
import numpy as np

def label_adjacent_pixels(image):
    height, width = image.shape
    labels = np.zeros((height, width), dtype=int)  # Array to store labels
    cur_label = 1  # Start labeling from 1 as per the algorithm
    
    # defining possible movements (8 neighbors)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # check if it's an edge pixel and has not been labeled yet
            if image[y, x] == 255 and labels[y, x] == 0:
                # start BFS for this pixel
                labels[y, x] = cur_label  # label it with the current label
                stack = [(y, x)]  # use a list (stack) to manage BFS

                while stack:
                    cur_y, cur_x = stack.pop()  # getting the current pixel
                    
                    # exploring all 8 neighbors
                    for dy, dx in directions:
                        ny, nx = cur_y + dy, cur_x + dx

                        # ensuring the neighbor is within bounds and is an unlabeled edge pixel
                        if 0 <= ny < height and 0 <= nx < width and image[ny, nx] == 255 and labels[ny, nx] == 0:
                            labels[ny, nx] = cur_label  # labeling the neighbor
                            stack.append((ny, nx))  # adding it to the stack to explore its neighbors

                # increment the label after exploring this component
                cur_label += 1

    return labels, cur_label - 1  # return the labels and the number of connected components becuase started from 1

# function to assign random colors to different connected regions
def color_components(labels, num_labels):
    # generate random colors for each component (1 to num_labels)
    colors = np.random.randint(0, 255, (num_labels + 1, 3), dtype=int)
    
    # create an empty color image
    color_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    
    # color each label with a different color
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] > 0:
                color_image[y, x] = colors[labels[y, x]]  # apply the corresponding color

    return color_image

def main():
    # read the grayscale image
    canny_edges = cv2.imread('images/t3b.tif', cv2.IMREAD_GRAYSCALE)
    MR_image = cv2.imread('images/t3a.tif', cv2.IMREAD_GRAYSCALE)

    # label the connected components
    labels_canny, num_labels_canny = label_adjacent_pixels(canny_edges)
    labels_MR, num_labels_MR = label_adjacent_pixels(MR_image)

    # color each connected component
    colored_components_canny = color_components(labels_canny, num_labels_canny)
    colored_components_MR = color_components(labels_MR, num_labels_MR)

    # save the result image
    cv2.imwrite('images/t4_canny.tif', colored_components_canny)
    cv2.imwrite('images/t4_MR.tif', colored_components_MR)

    # display the colored connected components
    cv2.imshow('Colored Connected Components Canny', colored_components_canny)
    cv2.imshow('Colored Connected Components MR', colored_components_MR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
