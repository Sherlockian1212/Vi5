import cv2
from ultralytics import YOLO
import json
import os
import shutil
from PIL import Image
from FormulaCaptioning import FormulaCaption
from ImageCaptioning import ImageCaption
from Text2Speech import Text2Speech

def crop_image(image, x_center, y_center, width, height):
    img = image

    # Tính toán tọa độ của hình chữ nhật cần cắt
    left = int((x_center - width / 2) * img.width)
    top = int((y_center - height / 2) * img.height)
    right = int((x_center + width / 2) * img.width)
    bottom = int((y_center + height / 2) * img.height)

    # Cắt ảnh
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img


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
        results = self.model.predict(source, save=True, save_txt=True, conf=0.5)

    def convert_yolo_to_json(self):
        current_directory = os.getcwd()
        path = os.path.join(current_directory, 'runs', 'segment', 'predict', 'labels')
        yolo_data_path = os.path.join(path, 'image0.txt')

        with open(yolo_data_path, 'r') as file:
            yolo_data = file.read()

        # Ensure yolo_data is a string
        yolo_data_str = '\n'.join(map(str, yolo_data))

        objects_list = []

        for line in yolo_data.split('\n'):
            if line.strip() != '':
                values = line.split()
                cr_img = crop_image(self.image, float(values[2]), float(values[3]), float(values[4]), float(values[5]))
                content="thành phần khác"
                if (int(values[0])) == 4:
                    content = Text2Speech(cr_img).image2Text()
                elif (int(values[0])) == 0:
                    content = FormulaCaption(cr_img).formulaCaption()
                elif (int(values[0])) == 2:
                    content = ImageCaption(cr_img).imageCaption()
                obj = {
                    "label": int(values[0]),
                    "confidence": float(values[1]),
                    "bounding_box": {
                        "x_center": float(values[2]),
                        "y_center": float(values[3]),
                        "width": float(values[4]),
                        "height": float(values[5])
                    },
                    "content": content
                }
                objects_list.append(obj)

        # Sắp xếp theo tọa độ y_center giảm dần
        sorted_objects = sorted(objects_list, key=lambda x: x["bounding_box"]["y_center"], reverse=True)

        result = {"objects": sorted_objects}
        with open(os.path.join(path, 'image0.json'), 'w') as json_file:
            json.dump(result, json_file, indent=2)

img = cv2.imread(r'D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\uploads\SGK07.png')
re = DocumentSegmentation(img).convert_yolo_to_json()
print(re)