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


def make_landmark_model_functional(my_small_images_dim_y, my_small_images_dim_x):
    small_images = Input(shape=(my_small_images_dim_x, my_small_images_dim_y, 3))

    conv1 = Conv2D(120, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same',
                    kernel_initializer=initializers.RandomNormal(stddev=0.035))(small_images)

    drop1 = Dropout(.2)(conv1)
    conv2 = Conv2D(50, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same',
                   kernel_initializer=initializers.RandomNormal(stddev=0.023))(drop1)
    drop2 = Dropout(.2)(conv2)
    flat = Flatten()(drop2)

    # flat_raw = Flatten()(small_images)
    #
    # flat_concat = keras.layers.concatenate([flat_raw, flat])

    feature_rich = Dense(100, activation='relu')(flat)
    facial_features = Dense(10, activation='linear')(feature_rich)

    model = Model(inputs=small_images, outputs=facial_features)
    optim = Adam(lr=0.00029, beta_1=0.8)
    model.compile(loss='mse', optimizer=optim)

    return model

def train_landmark_model(my_landmark_model, my_landmark_array, my_start_index, my_batch_epochs, my_batch_size,
                         my_number_of_images_to_use, my_small_images_dim_x,my_small_images_dim_y,
                         my_large_images_dim_x, my_large_images_dim_y, my_small_images_dir):

    for batch_epoch in range(0, my_batch_epochs):
        random_indices = np.random.randint(my_start_index, my_number_of_images_to_use,
                                           size=my_batch_size)  # Get a random batch of indices to the new array

        raw_batch = my_landmark_array[random_indices]  # Get the actual batch of landmarks

        x, y = get_training_data_from_raw_batch(raw_batch, my_small_images_dim_x, my_small_images_dim_y,
                                                my_large_images_dim_x, my_large_images_dim_y, my_small_images_dir)
        # TODO:REWRITE THIS IT'S OLD Transform the data into usable keras samples (i.e., convert rgb integers 0-255,
        # to floats in range 0.0-1.0 , & convert pixel location values to uv space percentages 0.0 - 1.0
        # x.shape=(32, 27, 22, 3), y.shape=(32, 10) x=low resolution images of faces,
        # and y is the location of eyes, nose and mouth in uv space

        loss = my_landmark_model.train_on_batch(x, y)
        print(loss)

        if batch_epoch % 20 == 0:
            print('----insdie-------')
            print('batch epoch', batch_epoch)
            print('Loss: ', loss)
            # #        print('Image-id', raw_batch[0][0])
            # print('Image-id', random_indices[0])
            # p = landmark_model.predict(x)
            # p = p * 176.0
            # print('Prediction: ', p[0][0], p[0][2])  # first 2 values
            # h5_filename = '/home/foo/data/celeba/models/landmark_model_batch_' + str(batch_epoch) + '_loss_' + str(loss) + '.h5'
            #
            # my_landmark_model.save(h5_filename, overwrite=True)  # 107,068,040 bytes
            # #my_landmark_model.save(h5_filename)
            # print('Written hdf5')
            # print('------------------')

    return my_landmark_model


# ----------------
super_seed = 1234
random.seed(super_seed)
np.random.seed(super_seed)
small_images_dir = '/home/foo/data/celeba/celeba_images_hr/' # TODO: changeh this later!
large_images_dir = '/home/foo/data/celeba/celeba_images_high_res/'

start_index = 300 #

number_of_images_to_use = 2000#150000
batch_size = 256
batch_epochs = 2500

# upsampled_number_of_images_to_use = 99000
# upsampled_batch_size = 50#45
# upsampled_batch_epochs = 10000

small_images_dim_x = 44
small_images_dim_y = 54

large_images_dim_x = 176
large_images_dim_y = 216

# ----------------

landmark_array = get_landmarks('/home/foo/data/celeba/list_landmarks_align_celeba.txt', number_of_images_to_use)  # Turn the raw landmarks text file into an integer np array
landmark_model = make_landmark_model_functional(small_images_dim_x, small_images_dim_y) # Build a supervised model that accepts very low resolution face images, and predicts the locations of 10 facial landmarks (such as nose location) -- (as uv percentages)

trained_landmark_model = train_landmark_model(landmark_model, landmark_array, start_index, batch_epochs, batch_size, number_of_images_to_use, small_images_dim_x, small_images_dim_y, large_images_dim_x, large_images_dim_y, small_images_dir)
trained_landmark_model.save('/home/foo/data/celeba/models/landmark_model_final.h5')
print('Saved)')


