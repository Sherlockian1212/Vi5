from pix2tex.cli import LatexOCR
import latex2mathml.converter
from javascript import require
from googletrans import Translator
from PIL import Image
import cv2

class FormulaCaption:
    def __init__(self, image):
        self.image = image

    def formulaCaption(self):
        img_pil = Image.fromarray(self.image)
        model = LatexOCR()
        latexCode = model(img_pil)

        latex_expr = [latexCode]

        # print(latex_expr)

        mml_expr = []
        for expr in latex_expr:
            mml_expr.append(latex2mathml.converter.convert(expr))

        # print(mml_expr)

        sre = require('speech-rule-engine')

        sre.setupEngine({'domain': 'clearspeak'})
        sre.engineReady()

        en_text = []
        for expr in mml_expr:
            en_text.append(sre.toSpeech(expr))

        # print(en_text)

        translator = Translator()
        vi_text = []
        for text in en_text:
            vi_text.append(translator.translate(text, dest='vi').text)

        return vi_text

# path = r"D:\STUDY\DHSP\NCKH-2023-With my idol\Vi6\uploads\test20.png"
# img = cv2.imread(path)
# print(FormulaCaption(img).formulaCaption())