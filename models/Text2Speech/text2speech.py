import os
from PIL import Image
import cv2
import pytesseract

class Text2Speech:
    def __init__(self, image):
        self.image = image
    def text2speech(self):
        image = self.image
        # Chuyển ảnh sang độ xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Xử lý ảnh (tuỳ chọn)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Ghi tạm ảnh xuống ổ cứng để sau đó apply OCR
        temp_filename = "temp.png"
        cv2.imwrite(temp_filename, gray)

        # Load ảnh và apply nhận dạng bằng Tesseract OCR
        text = pytesseract.image_to_string(Image.open(temp_filename), lang='vie')

        # Xóa ảnh tạm sau khi nhận dạng
        os.remove(temp_filename)

        with open('./output/temp.txt', 'w', encoding='utf-8') as file:
            file.write(text)
