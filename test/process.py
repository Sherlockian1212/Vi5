from models.ImageProcessing import imageprocessing
from models.Text2Speech import text2speech
from models.ImageCaption import predictimagecaption

''' TODO:
function createImage:
create new image with location is output of segmentation function is location of top-left and bottom-right pixel
'''
def createImage(image, location):
    return image
class Process:
    def __init__(self, image):
        self.image = image

    def imageProcessing(self):
        outputText = ''
        detectEdgeImage = imageprocessing.ImageProcessing(self.image).detectEdge()
        lists = imageprocessing.ImageProcessing(detectEdgeImage).segmentation()

        for list in lists:
            subImage = createImage(detectEdgeImage, list)
            if list[0] == 1:
                outputText += text2speech.Text2Speech(subImage).text2speech()
            elif list[0] == 2:
                outputText += predictimagecaption.PredictImageCaptioning(subImage).predict()
            elif list[0] == 3:
                outputText += 'formula'

        with open('./output/temp.txt', 'w', encoding='utf-8') as file:
            file.write(outputText)
        return 0
