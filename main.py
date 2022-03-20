import re
import traceback

import cv2
import numpy as np

from detectors import RectangleDetector, Filter, Area, TextDetector


def get_fighter(text):
    text = text.strip()
    try:
        if '\n' in text:
            name, club = filter(None, text.split("\n"))
            name = re.sub('[^0-9a-zA-Z ]+', '', name.strip()).strip()
            team = re.sub('[^0-9a-zA-Z ]+', '', club.strip()).strip()
            return Fighter(name=name, team=team)
        else:
            return Fighter(text, None)
    except ValueError as e:
        print(e)
    return Fighter(text, None)


class Fighter:
    def __init__(self, team: str, name: str):
        self.team = team
        self.name = name

    def __str__(self):
        return self.name if self.team is None else f"{self.name} - {self.team}"


class Info:
    def __init__(self, base_url: str, timestamp, text1: str, text2: str):
        self.base_url = base_url
        self.timestamp = int(timestamp / 1000)
        self.left_info = get_fighter(text1)
        self.right_info = get_fighter(text2)
        self.category = ""

    def __str__(self):
        return f"{self.base_url}&t={self.timestamp}s : {self.category} : {self.left_info} vs {self.right_info}\n"


def convert_millis(millis):
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    hours = int(hours)
    return seconds, minutes, hours


def formated_timestamp(millis):
    seconds, minutes, hours = convert_millis(millis)
    return f"{get_two_digits(hours)}:{get_two_digits(minutes)}:{get_two_digits(seconds)}"


def read_video():
    cap = cv2.VideoCapture("resources/day1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    f = open("resources/old/timestamp_day1.txt", "w")
    f.close()
    # get total number of frames and generate a list with each fps th frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x = [i for i in range(1, total_frames) if divmod(i, int(fps))[1] == 0]
    buffer = 0
    left_area = Area(x1=84, y1=190, x2=450, y2=530)
    left_filter = Filter(hmin=0, smin=245, vmin=72, hmax=179, smax=255, vmax=136)
    left_detector = RectangleDetector(left_area, left_filter, 1000)

    right_area = Area(x1=834, y1=192, x2=1200, y2=530)
    right_filter = Filter(hmin=25, smin=155, vmin=110, hmax=360, smax=255, vmax=255)
    right_detector = RectangleDetector(right_area, right_filter, 1000)

    first_fighter_info = TextDetector(Area(x1=84, y1=112, x2=450, y2=184))
    second_fighter_info = TextDetector(Area(x1=834, y1=112, x2=1200, y2=184))

    text_detector = TextDetector(Area(x1=526, y1=589, x2=761, y2=624))

    first_fighter_second_info = TextDetector(Area(x1=225, y1=650, x2=550, y2=685))
    second_fighter_second_info = TextDetector(Area(x1=736, y1=650, x2=1035, y2=685))

    search_category = False
    info = Info("https://www.youtube.com/watch?v=5xIAeX4LlGI", 0, "name1 \n club1", "name2 \n club2")
    buffer = 9 * 60
    for myFrameNumber in x:
        if buffer == 0 and search_category is False:
            # set which frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
            # read frame
            ret, frame = cap.read()
            # display frame
            try:
                left_detector.find(frame)
                found_left = left_detector.find(frame)
                found_right = right_detector.find(frame)
                number_of_white_pix = np.sum(frame == 255)
                number_of_black_pix = np.sum(frame == 0)

                if (found_left and found_right) or (
                        (found_left or found_right) and number_of_black_pix > 35_000 and number_of_white_pix < 9_000):
                    buffer = 15
                    search_category = True
                    # Get the time
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    info1 = first_fighter_info.detect_text(frame)
                    info2 = second_fighter_info.detect_text(frame)
                    info = Info("https://www.youtube.com/watch?v=5xIAeX4LlGI", timestamp, info1, info2)
                    print(info)
            except:
                traceback.print_exc()
        elif buffer == 0 and search_category is True:
            buffer = 10
            # set which frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
            # read frame
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (1920, 1080))

            cat_text = text_detector.detect_text(
                frame)

            if cat_text and ("PEE" in cat_text or "ARME" in cat_text or "HOM" in cat_text or "BOCLE" in cat_text):
                search_category = False
                cat_text = cat_text[0:cat_text.rfind("S") + 1]
                # Testing
                info.category = cat_text.strip().replace("’'", " ").replace("’/", " ")
                info.left_info.name = first_fighter_second_info.detect_text(frame).strip()
                info.right_info.name = second_fighter_second_info.detect_text(frame).strip()

                print(info)
                f = open("resources/timestamp_day1.txt", "a")
                f.write(info.__str__())
                f.close()
                buffer = 60
        else:
            buffer -= 1


def get_two_digits(val):
    return f"0{val}"[len(str(f"0{val}")) - 2:len(str(f"0{val}"))]


if __name__ == '__main__':
    read_video()
