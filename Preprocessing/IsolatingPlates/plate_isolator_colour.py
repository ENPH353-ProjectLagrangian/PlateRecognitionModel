#!/usr/bin/env python3

import cv2
import numpy as np


def _contour_area_tuple(c):
    return cv2.contourArea(c[0])


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
                ([100, 130, 50], [120, 255, 170]),
                ([30, 0, 0], [40, 255, 255])
            ]
        else:
            self.colour_bounds = colour_bounds

        self.testing = testing

    def extract_plates(self, img, duration=3000):
        """
        Returns plates in order: parking, license, or None if no plates found
        """
        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        car_mask, car_colour = self.get_car_mask(hsb)
        if car_mask is None:
            print("no plate_found")
            return None
        parking, license = self.get_plate_contours(hsb, car_mask, car_colour)

    def get_car_mask(self, img, duration=1000):
        bound_num = 0

        mask = [None, None, None]

        # mask by each possible car colour
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

            mask[bound_num] = cv2.inRange(img, lower, upper)

            # if self.testing:
            #     output = cv2.bitwise_and(img, img, mask=mask[bound_num])
            #     cv2.imshow(title, np.hstack([img, output]))
            #     cv2.waitKey(duration)
            
            bound_num += 1
        # Get the mask that had the largest contour (largest car), and
        # the contour of said car
        used_mask, car_contour = self.car_contour(mask)

        if car_contour is None:
            return None
        car_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.drawContours(car_mask, [car_contour], -1, (255), -1)
        if self.testing:
            cv2.imshow("car mask", car_mask)
        return car_mask, used_mask

    def get_plate_contours(self, img, mask, colour):
        """
        Returns contour of the plates (with perspective still)
        Parking plate first, then license plate
        @param img - img in which we search for plate
        @param mask - the mask of the car
        @param colour - the colour of the car (that will also be part of
                        contour and thus should be filtered)
        """
        print(img.shape)
        print(mask.shape)
        img = cv2.bitwise_and(img, img, mask=mask)

        (lower, upper) = self.colour_bounds[colour]
        # create numpy arrays from colour boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        colour_mask = cv2.inRange(img, lower, upper)
        colour_mask = cv2.bitwise_not(colour_mask)
        img = cv2.bitwise_and(img, img, mask=colour_mask)

        if self.testing:            
            cv2.imshow("masked image", img)
            cv2.waitKey(2000)

        return 1, 2

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
        
        cont_0 = [(c, 0) for c in contours0 if cv2.contourArea(c) > MIN_AREA]
        cont_1 = [(c, 1) for c in contours1 if cv2.contourArea(c) > MIN_AREA]
        cont_2 = [(c, 2) for c in contours2 if cv2.contourArea(c) > MIN_AREA]

        good_contours = [c for c in (cont_0 + cont_1 + cont_2)
                         if cv2.contourArea(c[0]) > MIN_AREA]

        list.sort(good_contours, key=_contour_area_tuple)
        if (len(good_contours) == 0):
            return None, None

        if self.testing:
            print(good_contours[0])

        return good_contours[0][1], good_contours[0][0]
