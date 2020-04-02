#!/usr/bin/env python3
import os 

import cv2
import augmentation_utils_beta as util

REPEATS_PER_INPUT = 9

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
path = os.path.dirname(os.path.realpath(__file__)) + "/"

# get all image paths for numbers
input_path_num = path + 'labelled_letters/Numbers/'
output_path_num = path + 'TrainingData/TrainingNums/'
files_nums = [f for f in os.listdir(input_path_num)]

# get all image paths for characters
input_path_chars = path + 'labelled_letters/Letters/'
output_path_chars = path + 'TrainingData/TrainingChars/'
files_chars = [f for f in os.listdir(input_path_num)]

for i in range(len(files_nums)):
    img = cv2.imread(input_path_num + files_nums[i])
    cv2.imwrite(output_path_num + files_nums[i][:3] + str(i) + '.png',
                img)
    for j in range(REPEATS_PER_INPUT):
        new_img = util.randomise_augmentation(img, letters=True)
        cv2.imwrite(output_path_num + files_nums[i][:3] +
                    str(i) + '_{}.png'.format(str(j)),
                    new_img)

for i in range(len(files_chars)):
    img = cv2.imread(input_path_chars + files_chars[i])
    cv2.imwrite(output_path_chars + files_chars[i][:3] + str(i) + '.png',
                img)
    for j in range(REPEATS_PER_INPUT):
        new_img = util.randomise_augmentation(img, letters=True)
        cv2.imwrite(output_path_chars + files_chars[i][:3] +
                    str(i) + '_{}.png'.format(str(j)),
                    new_img)
