import cv2
import numpy as np
import imutils
from skimage.filters import threshold_local

def order_points(pts):
   # Chuyển đổi danh sách pts thành mảng NumPy
   pts = np.array(pts, dtype="float32")

   # Khởi tạo danh sách tọa độ đã sắp xếp
   rect = np.zeros((4, 2), dtype="float32")

   # Tính tổng các tọa độ theo trục 1
   s = pts.sum(axis=1)

   # Tìm điểm top-left có tổng nhỏ nhất
   rect[0] = pts[np.argmin(s)]

   # Tìm điểm bottom-right có tổng lớn nhất
   rect[2] = pts[np.argmax(s)]

   # Tính sự khác biệt giữa các tọa độ theo trục 1
   diff = np.diff(pts, axis=1)

   # Tìm điểm top-right có sự khác biệt nhỏ nhất
   rect[1] = pts[np.argmin(diff)]

   # Tìm điểm bottom-left có sự khác biệt lớn nhất
   rect[3] = pts[np.argmax(diff)]

   # Trả về danh sách tọa độ đã sắp xếp
   return rect


def perspective_transform(image, pts):
   # unpack the ordered coordinates individually
   rect = order_points(pts)
   (tl, tr, br, bl) = rect

   '''compute the width of the new image, which will be the
   maximum distance between bottom-right and bottom-left
   x-coordinates or the top-right and top-left x-coordinates'''
   widthA = np.sqrt((np.abs(br[0] - bl[0]) * 2) + (np.abs(br[1] - bl[1]) * 2))
   widthB = np.sqrt((np.abs(tr[0] - tl[0]) * 2) + (np.abs(tr[1] - tl[1]) * 2))
   maxWidth = max(int(widthA), int(widthB))

   '''compute the height of the new image, which will be the
   maximum distance between the top-left and bottom-left y-coordinates'''
   heightA = np.sqrt((np.abs(tr[0] - br[0]) * 2) + (np.abs(tr[1] - br[1]) * 2))
   heightB = np.sqrt((np.abs(tl[0] - bl[0]) * 2) + (np.abs(tl[1] - bl[1]) * 2))
   maxHeight = max(int(heightA), int(heightB))

   '''construct the set of destination points to obtain an overhead shot'''
   dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

   # compute the perspective transform matrix
   transform_matrix = cv2.getPerspectiveTransform(rect, dst)

   # Apply the transform matrix
   warped = cv2.warpPerspective(image, transform_matrix, (int(maxWidth), int(maxHeight)))

   # return the warped image
   return warped

class EdgeDetection:
    def __init__(self, image):
        self.image = image

    def edgeDetection(self):
        original_image = self.image
        copy = original_image.copy()
        ratio = original_image.shape[0] / 500.0
        img_resize = imutils.resize(original_image, height=500)
        gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edged_img = cv2.Canny(blurred_image, 75, 200)

        cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                doc = approx
                break
        p = []

        for d in doc:
            tuple_point = tuple(d[0])
            p.append([tuple_point[0]*ratio, tuple_point[1]*ratio])
            cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)

        warped_image = perspective_transform(copy, p)

        cv2.imwrite('./uploads/' + 'scan' + '.png', imutils.resize(warped_image, height=650))
        return imutils.resize(warped_image, height=650)

