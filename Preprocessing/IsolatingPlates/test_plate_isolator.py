#!/usr/bin/env python3

import cv2
from plate_isolator_orb import PlateIsolatorOrb as PlateIsolator


def scale_image(img):
    MAX_IMG_WIDTH = 600

    width = img.shape[1]
    height = img.shape[0]
    if (width > MAX_IMG_WIDTH):
        scale_factor = MAX_IMG_WIDTH / width
        dim = (MAX_IMG_WIDTH, int(height * scale_factor))
        return cv2.resize(img, dim)

    return img


def test_image(img_path, isolator, duration=10000):
    detect_image = cv2.imread(img_path)
    # detect_image = scale_image(detect_image)
    matches = isolator.detectFeature(detect_image, detect_image,
                                     duration=duration)
    print(matches)
    pass


def main():
    feature_img = cv2.imread('dev_images/blue_crop_parking_only.jpg')
    isolator = PlateIsolator(feature_img)
    isolator.show_ref_and_keypoints()

    test_image('dev_images/test0_nobg.jpg', isolator)
    test_image('dev_images/test1_nobg.jpg', isolator)
    test_image('dev_images/test2_nobg.jpg', isolator)
    test_image('dev_images/test3_nobg.jpg', isolator)
    # test_image('dev_images/blue_nocrop.jpg', isolator)
    # test_image('dev_images/green_nocrop.jpg', isolator)
    # test_image('dev_images/green_nocrop2.jpg', isolator)
    # test_image('dev_images/yellow_nocrop.jpg', isolator)
    # test_image('dev_images/blue_crop.jpg', isolator)
    # test_image('dev_images/blue_crop_noletters.jpg', isolator)
    # test_image('dev_images/green_crop.jpg', isolator)
    # test_image('dev_images/yellow_crop.jpg', isolator)


    print("end of main")


if __name__ == "__main__":
    main()
