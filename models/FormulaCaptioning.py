from pix2tex.cli import LatexOCR
import latex2mathml.converter
from javascript import require
from googletrans import Translator
from PIL import Image

class FormulaCaption:
    def __init__(self, image):
        self.image = image

    def formulaCaption(self):
        model = LatexOCR()
        latexCode = model(self.image)

        latex_expr = [latexCode]
        mml_expr = []
        for expr in latex_expr:
            mml_expr.append(latex2mathml.converter.convert(expr))

        sre = require('speech-rule-engine')

        sre.setupEngine({'domain': 'clearspeak'})
        sre.engineReady()

        en_text = []
        for expr in mml_expr:
            en_text.append(sre.toSpeech(expr))

        translator = Translator()
        vi_text = []
        for text in en_text:
            vi_text.append(translator.translate(text, dest='vi').text)

        return vi_text
