#!/usr/bin/env python3
import cv2

from plate_isolator_colour import PlateIsolatorColour as PlateIsolator
from letter_isolator_beta import LetterIsolatorBeta as LetterIsolator


input_path = 'plate_images'
output_path = 'letters_unlabelled'
plate_isolation_failure = 'plates_unfound'
letter_isolation_failure = 'letter_isolation_failures'
counter = 500
input_set_size = 48

plate_isolator = PlateIsolator(testing=False)
letter_isolator = LetterIsolator(testing=False)

for i in range(0, input_set_size):
    img = cv2.imread('{}/image{}.png'.format(input_path, str(i)))
    parking_plate, license_plate = plate_isolator.extract_plates(img)

    if (parking_plate is not None):
        try:
            p, spot_num, l0, l1, n0, n1 = \
                letter_isolator.get_chars(parking_plate, license_plate)
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
        except AssertionError as e:
            print(e)
            print("could not recover letters from image {}".format(str(i)))
            cv2.imwrite("{}/parking_{}.png".format(letter_isolation_failure, 
                        str(i)), parking_plate)
            cv2.imwrite("{}/license_{}.png".format(letter_isolation_failure,
                        str(i)), license_plate)
    else:
        print("No plates found in image{}.png".format(str(i)))
        cv2.imwrite("{}/img_{}.png".format(plate_isolation_failure, str(i)), img)
