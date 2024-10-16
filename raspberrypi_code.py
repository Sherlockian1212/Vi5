from picamera import PiCamera
from time import sleep
from gpiozero import Button
import cv2
from gtts import gTTS
import pytesseract
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import pygame
from PIL import Image
from pydub import AudioSegment

def get_audio_len(path):
    audio = AudioSegment.from_file(path)
    return len(audio)/1000

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

class K_means:
    def __init__(self, image):
        self.image = image
    def cluster(self):
        img = self.image

        # Resize image
        dim_limit = 1080
        max_dim = max(img.shape)
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

        # Create a copy of resized original image for later use
        orig_img = img.copy()
        # cv2.imshow('orig_img', orig_img)
        # cv2.waitKey(0)

        # dir = 'D:\\STUDY\\DHSP\\Year3\\HK1\\DigitalImageProcessing-ThayVietDzeThuong\\Final-Project\\Document2Braille\\resources\\K-means\\'

        # Repeated Closing operation
        kernel = np.ones((9, 9), np.uint8)
        morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        # cv2.imwrite(dir + f"morph.png", morph)
        # cv2.imshow('morphology', morph)
        # cv2.waitKey(0)

        (h,w,c) = morph.shape
        img2D = morph.reshape(h*w,c)

        kmeans_model = KMeans(n_clusters=7, n_init = 'auto')
        cluster_labels = kmeans_model.fit_predict(img2D)

        rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)
        labels_count = Counter(cluster_labels)
        # print(labels_count)
        # print(kmeans_model.cluster_centers_)

        clustered_img = np.reshape(rgb_cols[cluster_labels],(h,w,c)).astype(np.uint8)
        #cv2.imwrite(dir + f"clustered_img.png", clustered_img)
        # cv2.imshow('cluster',clustered_img)
        # cv2.waitKey(0)

        # Find the label of the largest cluster
        largest_cluster_label = max(labels_count, key=labels_count.get)

        # Dilate the largest cluster
        largest_cluster_mask = (cluster_labels == largest_cluster_label).reshape(h, w)
        cluster_image = (largest_cluster_mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(cluster_image, kernel, iterations=4)
        #cv2.imwrite(dir + f"dilate.png", dilate)
        # cv2.imshow('dilate', dilate)
        # cv2.waitKey(0)

        # Find convex hull
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = cnt[0]
        largest_cluster = cv2.drawContours(morph.copy(), [largest_contour], -1, (0, 255, 0), 2)
        all_points = np.concatenate(largest_contour)
        hull = cv2.convexHull(all_points)
        hull_img = orig_img.copy()
        cv2.drawContours(hull_img, [hull], -1, (0, 255, 0), 10)
        # cv2.imwrite(dir + f"hull_img.png", hull_img)
        # cv2.imshow('hull_img', hull_img)
        # cv2.waitKey(0)

        # approximate the contour
        epsilon = 0.02 * cv2.arcLength(hull, True)
        corners = cv2.approxPolyDP(hull, epsilon, True)
        corners = sorted(np.concatenate(corners).tolist())
        corners = order_points(corners)
        corners_img = orig_img.copy()
        for corner in corners:
            cv2.circle(corners_img, tuple(corner), 20, (0, 255, 0), -1)
        # cv2.imwrite(dir + f"corners_img.png", corners_img)
        # cv2.imshow('corners_img', corners_img)
        # cv2.waitKey(0)

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

        if (final.shape[0] * final.shape[1]) < (orig_img.shape[0] * orig_img.shape[1] / 10):
            final = orig_img

        # dir = 'D:\\STUDY\\DHSP\\Year3\\HK1\\DigitalImageProcessing-ThayVietDzeThuong\\Final-Project\\Document2Braille\\resources\\K-means_Result'
        # cv2.imwrite(dir + f"final.png", final)

        return final
    

class Text2Speech:
    def __init__(self, image):
        self.image = image
    def image2Text(self):
        image = self.image
        # Chuyển ảnh sang độ xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Xử lý ảnh (tuỳ chọn)
        #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Load ảnh và apply nhận dạng bằng Tesseract OCR
        text = pytesseract.image_to_string(gray, lang='vie')

        return text

    def text2Speech(self):
        text = self.image2Text()
        if text.strip() == "":
            text = 'Không ghi nhận được, bạn thử lại nhé, yêu bạn'
        print(text)
        output = gTTS(text, lang="vi", slow=False)
        output.save('/home/nckh/Documents/audio/output.mp3')

def playsound(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def imageCapture(camera, button):
    camera.start_preview()
    button.wait_for_press()
    camera.capture('/home/nckh/Documents/image/sample.jpg')
    camera.stop_preview()
    
#che do chup xa   
def option_1():
    playsound('/home/nckh/Documents/audio/option1.mp3')
    sleep(2)
    image = cv2.imread('/home/nckh/Documents/image/sample.jpg')
    playsound('/home/nckh/Documents/audio/image_processing.mp3')
    sleep(2)
    k = K_means(image)
    preprocessed_image = k.cluster()
    cv2.imwrite('/home/nckh/Documents/image/preprocessed_image.jpg', preprocessed_image)

    tts = Text2Speech(preprocessed_image)
    tts.text2Speech()
    
    playsound('/home/nckh/Documents/audio/output.mp3')
    sleep(get_audio_len('/home/nckh/Documents/audio/output.mp3'))
#che do chup gan    
def option_2():
    playsound('/home/nckh/Documents/audio/option2.mp3')
    sleep(2)
    image = cv2.imread('/home/nckh/Documents/image/sample.jpg')
    playsound('/home/nckh/Documents/audio/image_processing.mp3')
    sleep(2)
    tts = Text2Speech(image)
    tts.text2Speech()

    playsound('/home/nckh/Documents/audio/output.mp3')
    sleep(get_audio_len('/home/nckh/Documents/audio/output.mp3'))
    
def chooseOption_2(button):
    result = False
    time_count = 0
    playsound('/home/nckh/Documents/audio/choose_option.mp3')
    sleep(7)
    while True:
        time_count += 1
        print(time_count)
        if (time_count >= 4000):
            break
        if button.is_pressed:
            result = True
            break
    return result
    
def main():
    pygame.mixer.init()
    
    button = Button(17)
    camera = PiCamera()
    
    while True:
        playsound('/home/nckh/Documents/audio/init.mp3')
        sleep(2)
        imageCapture(camera, button)
        
        option = chooseOption_2(button)
        if option:
            option_2()
        else:
            option_1()
      
if __name__== "__main__":
    main()