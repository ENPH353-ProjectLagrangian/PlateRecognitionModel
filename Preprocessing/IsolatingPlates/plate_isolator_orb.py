#!/usr/bin/env python3

import cv2
import numpy as np


class PlateIsolatorOrb:
    """
    The goal of this module is to pick out parking and license plates

    Input: a clean image of the plates
           Random, likely terrible images picked up from Anki camera

    Resources:
    https://pysource.com/2018/06/05/object-tracking-using-homography-opencv-3-4-with-python-3-tutorial-34/
    https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    https://docs.google.com/document/d/1trqdpvf9x_Ft62-yL35qbelQnErPKc49posviM9nw4U/edit

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/
    """

    def __init__(self, feature_img):
        """
        Sets up our sift recognition based off of our pattern
        """
        self.MAX_IMG_WIDTH = 200

        self.feature_img = self.rescale_img(feature_img)
        self.feature_img = self.preprocess_img(self.feature_img, 5, undistort=False)

        self.orb = cv2.ORB_create()
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.feature_img, None)
        self.brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def show_ref(self, duration_ms=5000):
        """
        Use for debugging purposes: show our reference image

        TODO: add functionality that displays key points for visual ref
        """
        cv2.imshow("Features image", self.feature_img)
        cv2.waitKey(duration_ms)
    
    def show_ref_and_keypoints(self, duration_ms=5000):
        """
        For debugging purposes: figure out what these keypoints look like!
        """
        # img = cv2.drawKeypoints(self.feature_img, self.keypoints,
        #                         color=(0, 255, 0), flags=0)
        # cv2.imshow("Features image with keypoints", img)
        # cv2.waitKey(duration_ms)
        pass

    def rescale_img(self, img):
        width = img.shape[1]
        height = img.shape[0]
        if (width > self.MAX_IMG_WIDTH):
            scale_factor = self.MAX_IMG_WIDTH / width
            dim = (self.MAX_IMG_WIDTH, int(height * scale_factor))
            return cv2.resize(img, dim)
        return img

    def preprocess_img(self, img, kernel_size=5, undistort=True):
        kernel = (kernel_size, kernel_size)
        img = cv2.GaussianBlur(img, kernel, 0)
        if (undistort):
            img = self.undistort(img)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detectFeature(self, ref_img, greyframe, duration=1000):
        greyframe = self.preprocess_img(greyframe, kernel_size=1)
        ref_img = self.preprocess_img(ref_img)
        kp, des = self.orb.detectAndCompute(greyframe, None)
        matches = self.brute_force.match(self.descriptors, des)
        matches = sorted(matches, key=lambda x:x.distance)
        matching_result = cv2.drawMatches(self.feature_img, self.keypoints,
                                          ref_img, kp, matches[:50],
                                          None, flags=2)
        cv2.imshow("Img1", self.feature_img)
        cv2.imshow("Img2", ref_img)
        cv2.imshow("Matching result", matching_result)
        cv2.waitKey(duration)
        return matches

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
