#!/usr/bin/env python3
import cv2

from plate_isolator_sift import PlateIsolatorSIFT as PlateIsolator
from letter_isolator import LetterIsolator


input_path = 'plate_images'
output_path = 'letters_unlabelled'
sift_failure_out = 'sift_fails'
letter_isolation_fails = 'letter_isolation_fails'
counter = 0
input_set_size = 48

feature_img = cv2.imread('new_plate_template.png')
plate_isolator = PlateIsolator(feature_img)
letter_isolator = LetterIsolator()

for i in range(0, input_set_size):
    img = cv2.imread('{}/image{}.png'.format(input_path, str(i)))
    cropped_img = plate_isolator.detectFeature(img, img, testing=False)

    if (cropped_img is not None):
        try:
            p, spot_num, l0, l1, n0, n1 = \
                letter_isolator.get_chars(cropped_img)
            cv2.imwrite("{}/p_{}.png".format(output_path, counter), p)
            counter += 1
            cv2.imwrite("{}/n_{}.png".format(output_path, counter), spot_num)
            counter += 1
            cv2.imwrite("{}/n_{}.png".format(output_path, counter), n0)
            counter += 1
            cv2.imwrite("{}/n_{}.png".format(output_path, counter), n1)
            counter += 1
            cv2.imwrite("{}/l_{}.png".format(output_path, counter), l0)
            counter += 1
            cv2.imwrite("{}/l_{}.png".format(output_path, counter), l1)
            counter += 1
            print("success")
        except AssertionError:
            print("could not recover letters from image {}".format(str(i)))
            cv2.imwrite("{}/img_{}.png".format(letter_isolation_fails, str(i)),
                        cropped_img)
    else:
        print("No plate found in image{}.png".format(str(i)))
        cv2.imwrite("{}/img_{}.png".format(sift_failure_out, str(i)), img)
