#!/usr/bin/env python3

import cv2
from letter_isolator import LetterIsolator


def main():
    feature_img = cv2.imread('paired_plates/plate_0.png')
    isolator = LetterIsolator(testing=True)

    isolator.get_chars(feature_img)

    img_0 = cv2.imread('paired_plates/plate_1.png')
    img_1 = cv2.imread('paired_plates/plate_2.png')
    img_2 = cv2.imread('paired_plates/plate_3.png')

    isolator.get_chars(img_0)
    isolator.get_chars(img_1)
    isolator.get_chars(img_2)

    print("end of main")


if __name__ == "__main__":
    main()
