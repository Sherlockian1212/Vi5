import cv2
import numpy as np
from rembg import remove

def order_points(pts):
    """Rearrange coordinates to order:
       top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()

class imagePreprocessing:
    def __init__(self, image):
        self.image = image

    def imageProcessing(selt):
        img = selt.image
        # Resize image to workable size
        dim_limit = 1080
        max_dim = max(img.shape)
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

        # Create a copy of resized original image for later use
        orig_img = img.copy()
        # cv2.imshow("original_resized", orig_img)

        # Repeated Closing operation to remove text from the document.
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

        output = remove(img)

        gray_image = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        segmented_image = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        canny = cv2.Canny(segmented_image, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Keeping only the largest detected contour.
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Detecting Edges through Contour approximation
        if len(page) == 0:
            return orig_img
        # loop over the contours
        for c in page:
            # approximate the contour
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            # if our approximated contour has four points
            if len(corners) == 4:
                break
        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())
        # For 4 corner points being detected.
        # Rearranging the order of the corner points.
        corners = order_points(corners)

        # Finding Destination Co-ordinates
        w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
        w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
        # Finding the maximum width.
        w = max(int(w1), int(w2))

        h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
        h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
        # Finding the maximum height.
        h = max(int(h1), int(h2))

        # Final destination co-ordinates.
        destination_corners = order_points(np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]))

        h, w = orig_img.shape[:2]
        # Getting the homography.
        homography, mask = cv2.findHomography(np.float32(corners), np.float32(destination_corners), method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
        # Perspective transform using homography.
        un_warped = cv2.warpPerspective(orig_img, np.float32(homography), (w, h), flags=cv2.INTER_LINEAR)
        # Crop
        final = un_warped[:destination_corners[2][1], :destination_corners[2][0]]
        return final

