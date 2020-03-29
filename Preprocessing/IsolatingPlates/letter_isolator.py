#!/usr/bin/env python3

import cv2
import numpy as np

class LetterIsolator:
    """
    Given greyscale image of the side of a car (both plates
    and grey in between), picks out each individual letter

    Resources:
    https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
    https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
    """

    def __init__(self, bin_thresh_plate=65, img_width=150,
                 testing=False):
        """
        @param binarisation_threshold: threshold which distinguishes between
                                       white and black in binary image
        @param img_width: width that input image gets scaled to
        """
        self.bin_thresh_plate = bin_thresh_plate
        self.IMG_WIDTH = img_width
        self.is_testing = testing

    def get_chars(self, img):
        """
        Given an image of a parking and license plate,
        returns each character as a separate image.
        @param img: greyscale image of the side of the car (with both plates)
        @return the cutout and binarised images for:
                P (for training purposes), parking spot number (parking plate)
                Letter0, Letter1, Num0, Num1 (license plate)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.rescale_img(img)
        self.display_test(img)
        bin_plate = self.binarise_plate(img)
        bin_text = self.binarise_text(img)
        self.display_test(bin_plate)
        out_parking, out_license = self.get_plates(bin_text, bin_plate)
        license_plate = self.clean_license_img(out_license)
        parking_plate = self.clean_parking_img(out_parking)
        self.display_test(parking_plate)
        self.display_test(license_plate)


# -------------------- Helpers -----------------------------

    def get_plates(self, img, bin_plates):

        """
        Given an image of the side of the car (both plates),
        returns the parking and license plates

        @param img: the image to process
        @return parking_plate, license_plate (cv2 images)

        Resource:
        https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
        """
        _, contours, _ = cv2.findContours(bin_plates, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        contour_parking, contour_license = self.get_contours(contours)
        # print("gottem")

        # contour_parking = contours[0] \
        #     if contours[0][0, 0, 1] < contours[1][0, 0, 1] else contours[1]
        # contour_license = contours[1] \
        #     if contours[0][0, 0, 1] < contours[1][0, 0, 1] else contours[0]

        mask_parking = np.zeros_like(img)
        mask_license = np.zeros_like(img)

        cv2.drawContours(mask_parking, [contour_parking], -1, 255, -1)
        cv2.drawContours(mask_license, [contour_license], -1, 255, -1)

        out_parking = np.zeros_like(img)
        out_license = np.zeros_like(img)

        out_parking[mask_parking == 255] = img[mask_parking == 255]
        out_license[mask_license == 255] = img[mask_license == 255]

        # crop
        (y_p, x_p) = np.where(mask_parking == 255)
        (topy_p, topx_p) = (np.min(y_p), np.min(x_p))
        (bottomy_p, bottomx_p) = (np.max(y_p), np.max(x_p))
        out_parking = out_parking[topy_p:bottomy_p + 1, topx_p:bottomx_p + 1]

        (y_l, x_l) = np.where(mask_license == 255)
        (topy_l, topx_l) = (np.min(y_l), np.min(x_l))
        (bottomy_l, bottomx_l) = (np.max(y_l), np.max(x_l))
        out_license = out_license[topy_l:bottomy_l + 1, topx_l:bottomx_l + 1]

        return out_parking, out_license

    def clean_license_img(self, img):
        delta = img.shape[1] // 55
        img[0:delta, :] = 255
        img[img.shape[0] - delta:img.shape[0], :] = 255
        img[:, 0:delta] = 255
        img[:, img.shape[1] - delta:img.shape[1]] = 255
        return img

    def clean_parking_img(self, img):
        delta = img.shape[1] // 15
        img[0:delta, :] = 255
        img[img.shape[0] - 2 * delta:img.shape[0], :] = 255
        img[:, 0:delta] = 255
        img[:, img.shape[1] - delta:img.shape[1]] = 255
        return img

    def binarise_plate(self, img):
        """
        Binarises the image to isolate plates
        @param: input image
        @return: the binarised image (plate black, rest white)
        """
        img = cv2.threshold(img, self.bin_thresh_plate, 255,
                            cv2.THRESH_BINARY)[1]
        kernel = np.ones((21, 21), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    def binarise_plate_adaptive(self, img):
        """
        Binarises the image to isolate plates, adaptively
        @param: input image
        @return: the binarised image (plate black, rest white)
        """
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 13, 2)
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def binarise_text(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, self.bin_thresh_plate, 255,
                            cv2.THRESH_BINARY)[1]
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def rescale_img(self, img):
        width = img.shape[1]
        height = img.shape[0]
        scale_factor = self.IMG_WIDTH / width
        dim = (self.IMG_WIDTH, int(height * scale_factor))
        return cv2.resize(img, dim)

    def display_test(self, img, duration=3000):
        if (self.is_testing):
            cv2.imshow("testing", img)
            cv2.waitKey(duration)

    def get_contours(self, contours):
        """
        Returns the two largest contours,
        insofar that they are large enough to be the plates
        Returns parking first, then license
        """
        # guestimate area: w = image.width * 0.8, h = 2/3*w
        # add factor of safety of 0.75
        MIN_AREA = (int)(0.75 * self.IMG_WIDTH * 0.8 * (2 + 4 / 3))

        good_contours = [c for c in contours
                         if (self.get_contour_area(c) > MIN_AREA)]

        assert len(good_contours) == 2

        contour_parking = good_contours[0] \
            if good_contours[0][0, 0, 1] < good_contours[1][0, 0, 1] \
            else good_contours[1]
        contour_license = good_contours[1] \
            if good_contours[0][0, 0, 1] < good_contours[1][0, 0, 1] \
            else good_contours[0]
        
        return contour_parking, contour_license

    def get_contour_area(self, contour):
        (x_ul, y_ul) = (np.min(contour[:, 0, 0]), np.min(contour[:, 0, 0]))
        (x_lr, y_lr) = (np.max(contour[:, 0, 0]), np.max(contour[:, 0, 0]))
        # print(str((x_lr - x_ul)*(y_lr - y_ul)))
        return (x_lr - x_ul) * (y_lr - y_ul)
