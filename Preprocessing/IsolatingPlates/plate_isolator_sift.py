#!/usr/bin/env python3

import cv2
import numpy as np
import random as rand
from scipy.spatial import distance as dist

rand.seed()


class PlateIsolatorSIFT:
    """
    The goal of this module is to pick out parking and license plates

    Input: a clean image of the plates
           Random, likely terrible images picked up from Anki camera

    Resources:
    https://pysource.com/2018/06/05/object-tracking-using-homography-opencv-3-4-with-python-3-tutorial-34/
    https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    https://docs.google.com/document/d/1trqdpvf9x_Ft62-yL35qbelQnErPKc49posviM9nw4U/edit

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

    For perspective transform:
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """

    def __init__(self, feature_img):
        """
        Sets up our sift recognition based off of our pattern
        """
        self.MAX_IMG_WIDTH = 200

        self.feature_img = self.rescale_img(feature_img)
        self.feature_img = self.preprocess_img(self.feature_img, 5)

        self.sift = cv2.xfeatures2d.SIFT_create()  # makes SIFT object

        # finds keypoints and gets descriptors in one method!
        self.key_points, self.descriptor = \
            self.sift.detectAndCompute(self.feature_img, None)

        # feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params,
                                           self.search_params)

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
        img = cv2.drawKeypoints(self.feature_img, self.key_points,
                                self.feature_img,
                                flags=cv2.
                                DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Features image with keypoints", img)
        cv2.waitKey(duration_ms)

    def rescale_img(self, img):
        width = img.shape[1]
        height = img.shape[0]
        if (width > self.MAX_IMG_WIDTH):
            scale_factor = self.MAX_IMG_WIDTH / width
            dim = (self.MAX_IMG_WIDTH, int(height * scale_factor))
            return cv2.resize(img, dim)
        return img

    def preprocess_img(self, img, kernel_size=5):
        # img = self.rescale_img(img)
        kernel = (kernel_size, kernel_size)
        img = cv2.GaussianBlur(img, kernel, 0)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detectFeature(self, ref_img, greyframe, duration=1000, testing=False):
        """
        Method: courtesy of homeography lab
        """
        greyframe = self.preprocess_img(greyframe, 3)
        if (testing):
            cv2.imshow("processed", greyframe)
            cv2.waitKey(duration)

        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(greyframe,
                                                                  None)
        matches = self.flann.knnMatch(self.descriptor, desc_grayframe, k=2)

        good_points = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        query_pts = np.float32([self.key_points[m.queryIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)

        if len(query_pts) == 0 or len(train_pts) == 0:
            if (testing):
                print("no query or training points")
                cv2.waitKey(duration)
            return None

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                          5.0)

        if matrix is None:
            if (testing):
                print("no homeography matrix")
                cv2.waitKey(duration)
            return None

        # perspective transform
        h, w = self.feature_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        # display result to screen if testing
        if (testing):
            homography = cv2.polylines(ref_img, [np.int32(dst)], True,
                                       (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
            cv2.waitKey(duration)

        return self.cropped_image(greyframe, np.int32(dst)[:, 0, :])

    def cropped_image(self, img, corners):
        ordered_corners = self.order_corners(corners)
        return self.four_point_transform(img, ordered_corners)

    def four_point_transform(self, img, ordered_corners):
        # obtain a consistent order of the points and unpack them
        # individually
        (tl, tr, br, bl) = ordered_corners
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    def order_corners(self, corners):
        """
        Helper function to generate our 4 points for perspective transform
        Important: points are generated in a consistent order! I'll do:
        1. top-left
        2. top-right
        3. bottom-right
        4. bottom-left

        From this blog post:
        https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
        """

        # sort points by X
        xSorted = corners[np.argsort(corners[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # grab top left and bottom left
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")
