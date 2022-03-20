import re

import cv2
import numpy as np
import pytesseract

prog = re.compile('[\\w]{4,} - [\\w]{4,}')


class Filter:
    def __init__(self, hmin: int, smin: int, vmin: int, hmax: int, smax: int, vmax: int):
        self.min = [hmin, smin, vmin]
        self.max = [hmax, smax, vmax]


class Area:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def fromTo(self, wf: int, hf: int, wt: int, ht):
        xfactor = wt / wf
        yfactor = ht / hf
        return Area(int(self.x1 * xfactor), int(self.y1 * yfactor), int(self.x2 * xfactor), int(self.y2 * yfactor))

    def crop(self, img):
        return img[self.y1:self.y2, self.x1:self.x2]


class RectangleDetector:
    def __init__(self, area: Area, f: Filter, t: int):
        self.area = area
        self.filter = f
        self.threshold = t

    def find(self, img) -> bool:
        img = self.area.crop(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(self.filter.min)
        upper = np.array(self.filter.max)
        img_result = get_rectangle(img, cv2.inRange(img_hsv, lower, upper))
        return found_rectanlge(img_result, self.threshold)


def display_img():
    img = cv2.imread("resources/frameTest2.png")

    left_area = Area(y1=290, y2=790, x1=0, x2=700)
    left_filter = Filter(hmin=0, smin=245, vmin=72, hmax=179, smax=255, vmax=136)
    left_detector = RectangleDetector(left_area, left_filter, 1000)
    left_detector.find(img)

    right_area = Area(y1=290, y2=790, x1=1230, x2=1800)
    right_filter = Filter(hmin=25, smin=155, vmin=110, hmax=360, smax=255, vmax=255)
    right_detector = RectangleDetector(right_area, right_filter, 1000)
    right_detector.find(img)

    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)

    print(f"right : {right_detector.find(img)}")
    print(f"left : {left_detector.find(img)}")
    print(f"number_of_white_pix : {number_of_white_pix}")
    print(f"number_of_black_pix : {number_of_black_pix}")

    cat_area = Area(x1=775, x2=1150, y1=870, y2=940)
    cat_filter = Filter(hmin=0, smin=106, vmin=136, hmax=24, smax=187, vmax=255)
    cat_detector = RectangleDetector(cat_area, cat_filter, 10)
    cat_detector.find(img)
    cv2.waitKey(0)


def get_left_rectangle(img):
    img = img[290:790, 0:700]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 245, 72])
    upper = np.array([179, 255, 136])
    img_result = get_rectangle(img, cv2.inRange(img_hsv, lower, upper))

    return img_result


def get_right_rectangle(img):
    img = img[290:790, 1230:1800]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 155, 110])
    upper = np.array([360, 255, 255])
    img_result = get_rectangle(img, cv2.inRange(img_hsv, lower, upper))
    return img_result


def get_rectangle(img, mask):
    img_result = cv2.bitwise_and(img, img, mask=mask)  # All the white in the mask get is kept
    img_blur = cv2.GaussianBlur(img_result, (7, 7), 1)
    img_canny = cv2.Canny(img_blur, 10, 10)
    return img_canny


def found_rectanlge(img, threshold):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold:
            cv2.drawContours(img, cnt, -1, (255, 0, 255), 3)  # Draw the countour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            return obj_cor == 4


class TextDetector:
    def __init__(self, area: Area):
        self.area = area

    def detect_text(self, img):
        pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Quentin\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
        img = self.area.crop(img)
        return pytesseract.image_to_string(img)


if __name__ == '__main__':
    print("Package imported")
    display_img()
