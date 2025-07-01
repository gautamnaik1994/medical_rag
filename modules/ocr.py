import pytesseract
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
import pymupdf
import logging

logger = logging.getLogger("ocr")


class OCRPipeline:
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.text_data = {}

    def preprocess_ocr_image(self, img):
        image_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        image_cv = cv2.GaussianBlur(image_cv, (3, 3), 0)
        blur_for_sharpen = cv2.GaussianBlur(image_cv, (9, 9), 10.0)
        image_cv = cv2.addWeighted(image_cv, 1.5, blur_for_sharpen, -1, 0)
        _, image_cv = cv2.threshold(
            image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_cv = cv2.morphologyEx(image_cv, cv2.MORPH_CLOSE, kernel)
        return image_cv

    def ocr_page(self, page):
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        preprocessed_img = self.preprocess_ocr_image(img)

        data = pytesseract.image_to_data(
            preprocessed_img, output_type=pytesseract.Output.DICT)
        lines = defaultdict(list)

        for i in range(len(data['text'])):
            if data['text'][i].strip():
                key = (data['block_num'][i], data['par_num']
                       [i], data['line_num'][i])
                lines[key].append((data['left'][i], data['text'][i].strip()))

        sorted_lines = []
        for line_id in sorted(lines):
            words = sorted(lines[line_id], key=lambda x: x[0])
            line_text = " ".join([w[1] for w in words])
            sorted_lines.append(line_text)

        return " ".join(sorted_lines)

    def run(self):
        try:
            doc = pymupdf.open(self.doc_path)

            for i, page in enumerate(doc):
                page_number = i + 1
                text = page.get_text().strip()

                if text:
                    self.text_data[page_number] = text
                else:
                    logger.info(
                        f"Page {page_number} has no text, running OCR.")
                    self.text_data[page_number] = self.ocr_page(page)

            return self.text_data

        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return None
