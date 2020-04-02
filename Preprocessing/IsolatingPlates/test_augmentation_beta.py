#!/usr/bin/env python
import cv2
import augmentation_utils_beta as util


def test_scale(img):
    for i in range(1, 5):
        for j in range(1, 5):
            new_img = util.scale(img, i / 6, j / 6)
            cv2.imshow("scale", new_img)
            cv2.waitKey(1000)


def test_rotate(img):
    for i in range(-3, 3):
        new_img = util.rotate(img, 15 * i)
        cv2.imshow("rotate", new_img)
        cv2.waitKey(1000)
    # consider: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders


def test_blur(img):
    for i in range(1, 15):
        new_img = util.blur(img, i * 2 + 1)
        cv2.imshow("blur", new_img)
        cv2.waitKey(1000)


def test_motion_blur(img):
    for i in range(1, 15):
        new_img = util.motion_blur(img, i * 2 + 1)
        cv2.imshow("motion_blur", new_img)
        cv2.waitKey(1000)


def test_noise(img):
    modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    # Notes: salt, pepper, s&p are not as useful I think
    for mode in modes:
        new_img = util.noise(img, mode)
        cv2.imshow("noise", new_img)
        cv2.waitKey(1000)

    for i in range(10):
        new_img = util.rand_noise(img)
        cv2.imshow("noise", new_img)
        cv2.waitKey(1000)


def test_hsb(img):
    for i in range(10):
        new_img = util.hsb(img)
        cv2.imshow("hsb", new_img)
        cv2.waitKey(1000)


def test_shift(img):
    for dx in range(-2, 2):
        for dy in range(-2, 2):
            new_img = util.shift(img, dx * 50, dy * 50)
            cv2.imshow("shift", new_img)
            cv2.waitKey(1000)


def test_perspective_transform(img):
    for i in range(10):
        new_img = util.perspective_transform(img)
        cv2.imshow("perspective", new_img)
        cv2.waitKey(1000)


def test_randomised_augmentations(img):
    augmentations = util.get_augmentations()
    for func in augmentations:
        for i in range(5):
            new_img = func(img)
            cv2.imshow("randomised augmentation", new_img)
            cv2.waitKey(500)


def test_random_augmentation_selector(img):
    for i in range(20):
        new_img = util.randomise_augmentation(img, letters=True)
        cv2.imshow("random augmentation", new_img)
        cv2.waitKey(500)


def main():
    print("exterminate: being test")
    # img = cv2.imread('dalek_test.jpeg')
    img = cv2.imread('test_letter.png')
    # test_scale(img)
    # test_rotate(img)
    # test_blur(img)
    # test_motion_blur(img)
    # test_noise(img)
    # test_hsb(img)
    # test_shift(img)
    # test_perspective_transform(img)
    # test_randomised_augmentations(img)
    test_random_augmentation_selector(img)
    print("Test complete: successful extermination")


if __name__ == "__main__":
    main()
