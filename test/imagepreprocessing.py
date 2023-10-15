from models.ImageProcessing import edgedetection
import cv2
import os

# Đường dẫn tới thư mục mặc định bạn muốn thiết lập
new_default_directory = os.getcwd()
# Sử dụng os.chdir() để thiết lập thư mục mặc định
os.chdir(new_default_directory)
image = cv2.imread('../uploads/test.png')

# preprocessimg = edgedetection.EdgeDetection(image).edgeDetectionGrabCut()
# print(preprocessimg)
cv2.imshow('preprocessed image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


