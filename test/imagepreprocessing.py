from models.ImageProcessing import edgedetection
import cv2
import os

# Đường dẫn tới thư mục mặc định bạn muốn thiết lập
new_default_directory = 'D:/STUDY/DHSP/NCKH-2023-With my idol/Vi6/'

# Sử dụng os.chdir() để thiết lập thư mục mặc định
os.chdir(new_default_directory)
image = cv2.imread('uploads/SGK07.png')

preprocessimg = edgedetection.EdgeDetection(image).edgeDetection()
# print(preprocessimg)
cv2.imshow('preprocessed image', preprocessimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


