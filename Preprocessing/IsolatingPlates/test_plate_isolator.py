#!/usr/bin/env python3

import cv2
from plate_isolator_sift import PlateIsolatorSIFT as PlateIsolator


def scale_image(img):
    MAX_IMG_WIDTH = 600

    width = img.shape[1]
    height = img.shape[0]
    if (width > MAX_IMG_WIDTH):
        scale_factor = MAX_IMG_WIDTH / width
        dim = (MAX_IMG_WIDTH, int(height * scale_factor))
        return cv2.resize(img, dim)

    return img


def test_image(img_path, isolator, duration=2000):
    detect_image = cv2.imread(img_path)
    # detect_image = scale_image(detect_image)
    cropped_img = isolator.detectFeature(detect_image, detect_image,
                                         duration=duration, testing=True)
    if (cropped_img is not None):
        cv2.imshow("cropped img", cropped_img)
        cv2.waitKey(duration)
    else:
        print("no plate found in this image")


def main():
    feature_img = cv2.imread('plate_template.jpg')
    isolator = PlateIsolator(feature_img)
    isolator.show_ref_and_keypoints()

    test_image('dev_images/test0.png', isolator)
    test_image('dev_images/test1.png', isolator)
    test_image('dev_images/test2.png', isolator)
    test_image('dev_images/test3.png', isolator)
    test_image('dev_images/test_skew0.png', isolator)
    test_image('dev_images/test_skew1.png', isolator)
    test_image('dev_images/test_small0.png', isolator)
    test_image('dev_images/test_small1.png', isolator)

    print("end of main")


if __name__ == "__main__":
    main()
