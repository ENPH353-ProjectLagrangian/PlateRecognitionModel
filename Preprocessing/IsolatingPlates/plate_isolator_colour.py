#!/usr/bin/env python3

import cv2
import numpy as np


class PlateIsolatorColour:
    """
    The goal of this module is to pick out parking
    1. will pull out cars by colour

    Input: a clean image of the plates
           Random, likely terrible images picked up from Anki camera

    Resources:
    https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    """

    def __init__(self, colour_bounds=None):
        """
        Sets up our sift recognition based off of our pattern
        """
        # in order HSB, green, blue, yellow
        if colour_bounds is None:
            self.colour_bounds = [
                ([50, 0, 0], [80, 240, 240]),
                ([90, 0, 0], [130, 240, 240]),
                ([0, 0, 0], [60, 255, 255])
            ]
        else:
            self.colour_bounds = colour_bounds

    def detectCar(self, img, duration=1000):
        bound_num = 0

        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for (lower, upper) in self.colour_bounds:
            # create numpy arrays from colour boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find colours in the range, apply mask
            if (bound_num == 0):
                title = "green"
            elif (bound_num == 1):
                title = "blue"
            else:
                title = "yellow"
            
            bound_num += 1

            mask = cv2.inRange(hsb, lower, upper)
            print(mask)
            output = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow(title, np.hstack([img, output]))
            cv2.waitKey(duration)

    def undistort(self, img):
        mtx = np.load("mtx.npy")
        dist = np.load("dist.npy")

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
