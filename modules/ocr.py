
import pytesseract
import numpy as np
import cv2
from collections import defaultdict
import pymupdf


class OCRPipeline():
    def __init__(self, doc):
        self.doc = doc
        self.images = None
        self.text_data = {}

    def read_document(self):
        self.images = []
        doc = pymupdf.open(self.doc)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            self.images.append(np.frombuffer(
                pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n))

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

    def extract_text(self):
        if self.images is None:
            raise ValueError("No images found")

        for i, img in enumerate(self.images):
            page_number = i + 1
            preprocessed_img = self.preprocess_ocr_image(np.asarray(img))
            data = pytesseract.image_to_data(
                preprocessed_img, output_type=pytesseract.Output.DICT)
            lines = defaultdict(list)

            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    key = (data['block_num'][i], data['par_num']
                           [i], data['line_num'][i])
                    lines[key].append(
                        (data['left'][i], data['text'][i].strip()))

            sorted_lines = []
            for line_id in sorted(lines):
                words = sorted(lines[line_id], key=lambda x: x[0])
                line_text = " ".join([w[1] for w in words])
                sorted_lines.append(line_text)

            self.text_data[page_number] = " ".join(sorted_lines)

    def visualize_text_boxes(self):
        if self.images is None:
            raise ValueError("No images found")

        for img in self.images:
            preprocessed_img = self.preprocess_ocr_image(np.asarray(img))
            data = pytesseract.image_to_data(
                preprocessed_img, output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    (x, y, w, h) = (data['left'][i], data['top']
                                    [i], data['width'][i], data['height'][i])
                    text = data['text'][i].strip()
                    if text:
                        cv2.rectangle(preprocessed_img, (x, y),
                                      (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(preprocessed_img, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

            cv2.imwrite("ocr_debug_boxes.png", preprocessed_img)

    def run(self):
        try:
            self.read_document()
            self.extract_text()
            return self.text_data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
