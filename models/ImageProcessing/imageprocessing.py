import cv2

class ImageProcessing:
    def __init__(self, image):
        self.image = image
    '''
    function detectEdge:
    input: a image of a book page + background
    output: detect edge, delete background, rotate, cut --> image
    '''
    def detectEdge(self):

        output = self.image.copy()

        return output

    '''
    function segmentation:
    input: a detected edge image
    output: a array with format: [(type,x1,y1,x2,y2)]
    type: a number (1: text, 2: image, 3: formula)
    x1, y1, x2, y2: location of top-left and bottom-right
    '''
    def segmentation(self):
        return 0
