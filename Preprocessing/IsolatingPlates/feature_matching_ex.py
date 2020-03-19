import cv2
import numpy as np

"""
Code from here! All credit to original author! 
I just want to check out what it does properly

Here: https://pysource.com/2018/03/23/feature-matching-brute-force-opencv-3-4-with-python-3-tutorial-26/
"""

img1 = cv2.imread("dev_images/the_book_thief.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("dev_images/me_holding_book.jpg", cv2.IMREAD_GRAYSCALE)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
