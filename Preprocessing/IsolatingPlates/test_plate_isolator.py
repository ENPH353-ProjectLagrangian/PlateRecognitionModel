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


def test_image(img_path, isolator, img_num, duration=5000):
    detect_image = cv2.imread(img_path)
    # detect_image = scale_image(detect_image)
    cropped_img = isolator.detectFeature(detect_image, detect_image,
                                         duration=duration, testing=True)

    if (cropped_img is not None):
        cv2.imshow("cropped img", cropped_img)
        cv2.imwrite("paired_plates/plate_{}.png".format(img_num), cropped_img)
    else:
        print("no plate found in this image")


def main():
    feature_img = cv2.imread('plate_template.jpg')
    isolator = PlateIsolator(feature_img)
    isolator.show_ref_and_keypoints()

    # test_image('plate_images/image4_test.png', isolator, 21)
    # test_image('plate_images/image4.png', isolator, 7)
    test_image('plate_images/img_32_test_2.png', isolator, 19)
    test_image('plate_images/img_32_test_3.png', isolator, 20)
    test_image('plate_images/img_32_test_4.png', isolator, 19)
    # test_image('plate_images/image32.png', isolator, 21)
    # test_image('plate_images/image6.png', isolator, 9)
    # test_image('plate_images/image7.png', isolator, 10)
    # test_image('plate_images/image8.png', isolator, 11)
    # test_image('plate_images/image9.png', isolator, 12)
    # test_image('plate_images/image14.png', isolator, 13)
    # test_image('plate_images/image16.png', isolator, 14)
    # test_image('plate_images/image18.png', isolator, 15)
    # test_image('plate_images/image20.png', isolator, 16)
    # test_image('plate_images/image21.png', isolator, 17)
    # test_image('plate_images/image22.png', isolator, 18)

    # test_image('dev_images/test0.png', isolator, 0)
    # test_image('dev_images/test1.png', isolator, 1)
    # test_image('dev_images/test2.png', isolator, 2)
    # test_image('dev_images/test3.png', isolator, 3)
    # test_image('dev_images/test4.png', isolator, 4)
    # test_image('dev_images/test5.png', isolator, 5)
    # test_image('dev_images/test6.png', isolator, 6)

    # test_image('dev_images/test_skew0.png', isolator, 4)
    # test_image('dev_images/test_skew1.png', isolator, 5)
    # test_image('dev_images/test_small0.png', isolator)
    # test_image('dev_images/test_small1.png', isolator)

    print("end of main")


if __name__ == "__main__":
    main()
