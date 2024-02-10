import os
from PIL import Image
import cv2
import pytesseract
from gtts import gTTS

class Text2Speech:
    def __init__(self, image):
        self.image = image
    def image2Text(self):
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

        return text

    def text2Speech(self):
        text = self.image2Text()
        output = gTTS(text, lang="vi", slow=False)

        new_default_directory = os.getcwd()
        # Sử dụng os.chdir() để thiết lập thư mục mặc định
        os.chdir(new_default_directory)
        output.save('../output/output.mp3')

# path = r"D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\models\runs\segment\predict\crops\Text\image0.jpg"
# img = cv2.imread(path)
# print(Text2Speech(img).image2Text())
#
# text = "không ghi nhận được"
# output = gTTS(text, lang="vi", slow=False)
# new_default_directory = os.getcwd()
# # Sử dụng os.chdir() để thiết lập thư mục mặc định
# os.chdir(new_default_directory)
# output.save('../output/no_record.mp3')