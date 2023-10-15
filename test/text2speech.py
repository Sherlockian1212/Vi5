import cv2
import os
from models.Text2Speech import text2speech

new_default_directory = os.getcwd()
# Sử dụng os.chdir() để thiết lập thư mục mặc định
os.chdir(new_default_directory)
image = cv2.imread('../uploads/SGK07.png')

text2speech.Text2Speech(image).text2Speech()
