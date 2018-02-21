import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Merge, MaxPooling2D, LSTM
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from keras import initializers
import cv2
from keras.models import load_model

def normalize_to_uv_space(my_landmarks, my_large_images_dim_x, my_large_images_dim_y):
    uv_landmarks = np.zeros((my_landmarks.shape[0],),dtype=np.float)  # This loop transform is not written well,
    # np has another way to convert int to float (astype....) TODO: Re-write this is np.astype()
    for j in range(0, my_landmarks.shape[0]):
        if j % 2 == 0:
            uv_landmarks[j] = my_landmarks[j] * 1.0 / my_large_images_dim_x * 1.0
        if j % 2 == 1:
            uv_landmarks[j] = my_landmarks[j] * 1.0 / my_large_images_dim_y * 1.0

    return uv_landmarks


def get_landmarks(my_csv_file, my_number_of_images_to_use):
    landmark_arr = np.ones((my_number_of_images_to_use, 11), dtype=int)
    line_count = 0
    with open(my_csv_file) as input_file:
        for line in input_file:
            if line_count < my_number_of_images_to_use: # The first 2 lines are count total and the column descriptions
                split_up = line.split()
                file_name = split_up[0]
                file_name = int(file_name[0:6])  # filename0 left eye_x left eye_y right eye_x right eye_y nose_x
                # nose_y left mouth_x left mouth_y right mouth_x right mouth
                split_up[0] = file_name
                landmark_arr[line_count] = np.array(split_up)
            line_count += 1
    return landmark_arr


def get_full_file_string_from_int(my_file_num):
    file_str = str(my_file_num)
    extra_zeros_to_pre_append = 6 - len(file_str)
    file_nm = '0' * extra_zeros_to_pre_append
    file_nm = str(file_nm) + file_str + '.jpg'

    if my_file_num == 0:
        file_nm = '000000.jpg'

    return file_nm


def get_training_data_from_raw_batch(my_raw_batch, my_small_images_dim_x, my_small_images_dim_y, my_large_images_dim_x,
                                     my_large_images_dim_y, my_small_images_dir):

    x_np = np.zeros((my_raw_batch.shape[0],my_small_images_dim_y,my_small_images_dim_x, 3), dtype=np.float)
    y_np = np.zeros((my_raw_batch.shape[0], 10,), dtype=np.float)

    for i in range(0, my_raw_batch.shape[0]):  # I changed data set so there is a dupe at 0 , 0 and 1 are the same
        file_num = my_raw_batch[i][0]
        file_name = my_small_images_dir + get_full_file_string_from_int(file_num)
        image = cv2.imread(file_name, 3)  # image.shape is (27, 22, 3)
        #print(file_name)
        image = image / 255.0
        x_np[i] = image

        landmarks = my_raw_batch[i][1:11]
        landmarks = normalize_to_uv_space(landmarks, my_large_images_dim_x, my_large_images_dim_y)

        y_np[i] = landmarks

    return x_np, y_np


# ----------------
super_seed = 1234
random.seed(super_seed)
np.random.seed(super_seed)

small_images_dir = '/home/foo/data/celeba/celeba-images-22-27/'
large_images_dir = '/home/foo/data/celeba/celeba-images-176-216/'

start_index = 300 #

number_of_images_to_use = 99000
batch_size = 256
batch_epochs = 2500

upsampled_number_of_images_to_use = 99000
upsampled_batch_size = 50#45
upsampled_batch_epochs = 10000

small_images_dim_x = 22
small_images_dim_y = 27

large_images_dim_x = 176
large_images_dim_y = 216

