#!/usr/bin/env python3

import cv2
from letter_isolator import LetterIsolator


def main():
    feature_img = cv2.imread('paired_plates/plate_0.png')
    isolator = LetterIsolator(testing=True)

    isolator.get_chars(feature_img)

    print("end of main")


if __name__ == "__main__":
    main()
