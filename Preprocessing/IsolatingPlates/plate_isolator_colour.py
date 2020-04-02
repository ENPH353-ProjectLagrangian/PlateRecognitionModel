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

    def __init__(self, colour_bounds=None, testing=False):
        """
        Sets up our sift recognition based off of our pattern
        """
        # in order HSB, green, blue, yellow
        if colour_bounds is None:
            self.colour_bounds = [
                ([50, 0, 0], [80, 255, 255]),
                ([100, 130, 100], [120, 255, 170]),
                ([30, 0, 0], [40, 255, 255])
            ]
        else:
            self.colour_bounds = colour_bounds

        self.testing = testing

    def extract_plates(self, img, duration=3000):
        """
        Returns plates in order: parking, license, or None if no plates found
        """
        contour = self.get_car_contour(img)
        if contour is None:
            print("no contour")
            return None
        cv2.drawContours(img, [contour], -1, 255, -1)
        if self.testing:
            cv2.imshow("contour", img)
            cv2.waitKey(duration)

    def get_car_contour(self, img, duration=1000):
        bound_num = 0

        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = [None, None, None]

        for (lower, upper) in self.colour_bounds:
            # create numpy arrays from colour boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find colours in the range, apply mask
            if self.testing:
                if (bound_num == 0):
                    title = "green"
                elif (bound_num == 1):
                    title = "blue"
                else:
                    title = "yellow"

            mask[bound_num] = cv2.inRange(hsb, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask[bound_num] = cv2.morphologyEx(mask[bound_num],
                                               cv2.MORPH_CLOSE, kernel)
            mask[bound_num] = cv2.morphologyEx(mask[bound_num],
                                               cv2.MORPH_OPEN, kernel)
            if self.testing:
                output = cv2.bitwise_and(img, img, mask=mask[bound_num])
                cv2.imshow(title, np.hstack([img, output]))
                cv2.waitKey(duration)
            
            bound_num += 1

        car_contour = self.car_contour(mask)

        if car_contour is not None:
            return car_contour
        return None

    def car_contour(self, mask):
        _, contours0, _ = cv2.findContours(mask[0], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        _, contours1, _ = cv2.findContours(mask[1], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        _, contours2, _ = cv2.findContours(mask[2], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        """
        guestimate area: experimentally determined
        """
        MIN_AREA = (int)(0.75 * mask[0].shape[1] / 6 * mask[0].shape[0] / 4)
        good_contours = [c for c in (contours0 + contours1 + contours2)
                         if cv2.contourArea(c) > MIN_AREA]
        list.sort(good_contours, key=cv2.contourArea)
        if (len(good_contours) == 0):
            return None
        if self.testing:
            print(good_contours[0])
        return good_contours[0]
