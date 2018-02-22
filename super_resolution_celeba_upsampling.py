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


super_seed = 1234
random.seed(super_seed)
np.random.seed(super_seed)

small_images_dir = '/home/foo/data/celeba/celeba_images_low_res/'
large_images_dir = '/home/foo/data/celeba/celeba_images_high_res/'

# small_images_dir = '/low/'
# large_images_dir = '/high/
# landmarks_floyd = '/misc/list_landmarks_align_celeba.txt'

start_index = 300 #

upsampled_number_of_images_to_use = 2000  # 200000
upsampled_batch_size = 50#45
upsampled_batch_epochs = 10000

small_images_dim_x = 44
small_images_dim_y = 54

large_images_dim_x = 176
large_images_dim_y = 216