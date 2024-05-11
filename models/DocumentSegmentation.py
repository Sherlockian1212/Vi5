import cv2
from ultralytics import YOLO
import os
import shutil
from FormulaCaptioning import FormulaCaption
from ImageCaptioning import ImageCaption
from Text2Speech import Text2Speech
from PIL import Image
import re
from gtts import gTTS

class DocumentSegmentation:
    def __init__(self, image):
        self.model = YOLO(r'D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\weight\best.pt')
        self.image = image
        current_directory = os.getcwd()
        path=os.path.join(current_directory, 'runs', 'segment', 'predict')
        shutil.rmtree(path)
        self.yolo_data = self.predict()

    def predict(self):
        source = self.image
        results = self.model.predict(source, save=True, save_crop=True, conf=0.5)
        output_list = []

        for result in results:
            # iterate results
            boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for box in boxes:  # iterate boxes
                r = box.xyxy[0].astype(int)
                output_list.append((r[0], r[1], r[2], r[3], int(box.cls[0])))
        print(output_list)
        sorted_data = sorted(output_list, key=lambda x: (x[1], x[0]), reverse=False)
        # In ra kết quả
        print(sorted_data)
        return sorted_data

    def result_text(self):
        img = self.image
        list = self.yolo_data
        re = []
        for l in list:
            cr_img = img[l[1]:l[3], l[0]:l[2], :]
            if l[4] == 4:
                content = Text2Speech(cr_img).image2Text()
            elif l[4] == 0:
                 content = 'Mô tả của công thức như sau ' + FormulaCaption(cr_img).formulaCaption()[0]
            elif l[4] == 2:
                content = 'Mô tả của hình ảnh như sau ' + ImageCaption(cr_img).imageCaption()[0]
            re.append(content)
        print(re)
        return re
img = cv2.imread(r'D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\uploads\SGK13.png')
res = DocumentSegmentation(img)
result = res.result_text()
result = [''.join(sublist) for sublist in result]
result = ''.join(result)
print(result)
processing_result = re.sub(r'[^\w\s]', '', result)
processing_result = processing_result.replace('\n', ' ')
text = processing_result
print(text)
output = gTTS(text, lang="vi", slow=False)
output.save(r'D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\output\output.mp3')