import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Merge, MaxPooling2D, LSTM, BatchNormalization, AtrousConv2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from keras import initializers
import cv2
from keras.models import load_model
import h5py

def get_full_file_string_from_int(my_file_num):
    file_str = str(my_file_num)
    extra_zeros_to_pre_append = 6 - len(file_str)
    file_nm = '0' * extra_zeros_to_pre_append
    file_nm = str(file_nm) + file_str + '.jpg'

    if my_file_num == 0:
        file_nm = '000000.jpg'

    return file_nm

def load_predict_and_save_images(my_image_numbers, my_upsampled_model, my_trained_landmarks_model, my_write_images,
                            my_small_images_dim_x, my_small_images_dim_y, my_large_images_dim_x,my_large_images_dim_y,
                            my_small_images_dir, my_large_images_dir, my_batch_epoch, my_loss, my_system_mode):

    total_images = my_image_numbers.shape[0]
    low_res_imgs = np.zeros((total_images, my_small_images_dim_y, my_small_images_dim_x, 3))
    bicubic_imgs = np.zeros((total_images, my_large_images_dim_y, my_large_images_dim_x, 3))

    for i in range(0, total_images):
        img_number = my_image_numbers[i]
        low_res_path = my_small_images_dir + get_full_file_string_from_int(img_number)
        high_res_path = my_large_images_dir + get_full_file_string_from_int(img_number)

        low_res_image = cv2.imread(low_res_path, 3)
        high_res_image = cv2.imread(high_res_path, 3)
#        bicubic_image = cv2.resize(low_res_image, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        bicubic_image = cv2.resize(low_res_image, None, fx=8, fy=8, interpolation=cv2.INTER_LANCZOS4)


        #dir_prefix = '/home/foo/data/celeba/generated/'
        dir_prefix = '/output/'
        if my_system_mode == 'local_cpu':
            dir_prefix = '/home/foo/data/celeba/generated/'


        low_res_output_filename = dir_prefix + str(img_number) + '_low_res_' + '.png'
        high_res_output_filename = dir_prefix + str(img_number) + '_high_res_' + '.png'
        bicubic_output_filename = dir_prefix + str(img_number) + '_bicubic_' + '.png'

        if my_write_images:
            cv2.imwrite(low_res_output_filename, low_res_image)
            cv2.imwrite(high_res_output_filename, high_res_image)
            cv2.imwrite(bicubic_output_filename,bicubic_image)

        low_res_image = low_res_image / 255.0
        bicubic_image = bicubic_image / 255.0

        low_res_imgs[i] = low_res_image
        bicubic_imgs[i] = bicubic_image

    landmark_model_predictions = my_trained_landmarks_model.predict(low_res_imgs)

    predictions = my_upsampled_model.predict([low_res_imgs, landmark_model_predictions])

    for i in range(0, total_images):

        prediction = predictions[i]
        prediction_image_large = np.reshape(prediction, (prediction.shape[0], prediction.shape[1], prediction.shape[2]))
        prediction_image_large = prediction_image_large * 255.0
        for a in range(0, prediction_image_large.shape[0]):
            for b in range(0, prediction_image_large.shape[1]):
                for c in range(0, 3):
                    if prediction_image_large[a][b][c] < 0.0:
                        prediction_image_large[a][b][c] = 0.0
                    if prediction_image_large[a][b][c] > 255.0:
                        prediction_image_large[a][b][c] = 255.0

        filename = dir_prefix + str(my_image_numbers[i]) + '-be-' + str(my_batch_epoch) + '-ls-' + str(my_loss) + '.png'
        output_text = 'Batch, ' + str(my_batch_epoch) + ' - Loss, ' + str(my_loss) + ' Prediction from Image #' + str(my_image_numbers[i]) + ' Written'
        print(output_text)
        cv2.imwrite(filename, prediction_image_large)

    return low_res_imgs, predictions

# def make_upsampled_model_functional(my_small_images_dim_x, my_small_images_dim_y, my_large_images_dim_x, my_large_images_dim_y):
#good settings
#     number_of_extra_inferred_dimensions = 12#24
#
#     images_small = Input(shape=(my_small_images_dim_y, my_small_images_dim_x, 3))
#     landmarks = Input(shape=(10,))
#
#     extract1 = Conv2D(120, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(images_small)
#     pool1 = MaxPooling2D()(extract1)
#     drop1 = Dropout(.2)(pool1)
#     extract2 = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(drop1)
#     pool2 = MaxPooling2D()(extract2)
#     drop2 = Dropout(.2)(pool2)
# #    extract3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
#  #   pool3 = MaxPooling2D()(extract3)
#     flat = Flatten()(drop2)
#     inferred_features = Dense(100, activation='sigmoid')(flat)
#
#     landmarks1 = Dense(100, activation='sigmoid')(landmarks)
#
#     #raw_rgb = Conv2D(4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(images_small)
#     upsampled_rgb = UpSampling2D()(images_small)
#     upsampled_rgb_conv = Conv2D(32, kernel_size=(9, 9), strides=(1, 1), padding='same', activation='relu')(upsampled_rgb)
#
#
#     #put the upsampled back???
#     landmarks_and_inferred_features = keras.layers.concatenate([landmarks1, inferred_features])
#     #mid = Dense(60, activation='sigmoid')(landmarks_and_inferred_features)
#     #drop = Dropout(.25)(landmarks_and_inferred_features)
#
#     flattened_medium_sized_generated_image = Dense(my_small_images_dim_y * 1 * my_small_images_dim_x * number_of_extra_inferred_dimensions, activation='relu')(landmarks_and_inferred_features)
#     rectangular_medium_sized_image = Reshape((my_small_images_dim_y , my_small_images_dim_x , number_of_extra_inferred_dimensions))(flattened_medium_sized_generated_image)
#     bigger_image = UpSampling2D()(rectangular_medium_sized_image)
#
#
#     #full_resolution_inferred = UpSampling2D()(bigger_image) # The third dimensions, usually 3 for RGB, is not RGB but rather n-dimensional feature space
#
#     n_dimensional_image = keras.layers.concatenate([bigger_image,  upsampled_rgb_conv])
#     n_dimensional_image_hr =  UpSampling2D()(n_dimensional_image)
#     n_conv1 = Conv2D(32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(n_dimensional_image_hr)
#     final_image = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(n_conv1)
#
#     model = Model(inputs=[images_small, landmarks], outputs=final_image)
#     optim = Adam(lr=0.00037, beta_1=0.91)
#     #optim = Adadelta(lr=0.00037, beta_1=0.91)
#     model.compile(loss='mse', optimizer=optim)
#
#     return model

def make_upsampled_model_functional(my_small_images_dim_x, my_small_images_dim_y, my_large_images_dim_x, my_large_images_dim_y):
    number_of_extra_inferred_dimensions = 8#5#12#24#12#24

    images_small = Input(shape=(my_small_images_dim_y, my_small_images_dim_x, 3))
    landmarks = Input(shape=(10,))

    extract1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(images_small)

    pool1 = MaxPooling2D()(extract1)
    drop1 = Dropout(.2)(pool1)
    extract2 = Conv2D(26, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(drop1)
    pool2 = MaxPooling2D()(extract2)
    drop2 = Dropout(.2)(pool2)
    #extract3 = Conv2D(14, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(drop2)
    #pool3 = MaxPooling2D()(extract3)

    flat = Flatten()(drop2)
    inferred_features = Dense(110, activation='sigmoid')(flat)

    landmarks1 = Dense(110, activation='sigmoid')(landmarks)

    landmarks_and_inferred_features = keras.layers.concatenate([landmarks1, inferred_features])
    batchnormal = BatchNormalization()(landmarks_and_inferred_features)

    flattened_medium_sized_generated_image = Dense(my_small_images_dim_y * 1 * my_small_images_dim_x * number_of_extra_inferred_dimensions, activation='relu')(batchnormal)
    rectangular_medium_sized_image = Reshape((my_small_images_dim_y , my_small_images_dim_x , number_of_extra_inferred_dimensions))(flattened_medium_sized_generated_image)

    smush = keras.layers.concatenate([rectangular_medium_sized_image, images_small])

    bigger_image = UpSampling2D()(smush)
    big_conv1 = Conv2D(50, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(bigger_image)
    big_conv2 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(big_conv1)




    full_resolution_inferred = UpSampling2D()(big_conv2) # The third dimensions, usually 3 for RGB, is not RGB but rather n-dimensional feature space


    n_conv1 = Conv2D(24, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(full_resolution_inferred)
    n_conv2 = Conv2D(12, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(n_conv1)

    #almost_final_image = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(n_conv2)

    fi = UpSampling2D()(n_conv2)
    fi_conv1 = Conv2D(24, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(fi)
    fi_conv2 = Conv2D(12, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(fi_conv1)
    final_image = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(fi_conv2)



    model = Model(inputs=[images_small, landmarks], outputs=final_image)
    optim = Adam(lr=0.0005, beta_1=0.91)

    model.compile(loss='mse', optimizer=optim)

    return model

# def make_upsampled_model_functional(my_small_images_dim_x, my_small_images_dim_y, my_large_images_dim_x, my_large_images_dim_y):
# 40 really good after 5000 - 8000
#     number_of_extra_inferred_dimensions = 5#12#24#12#24
#
#     images_small = Input(shape=(my_small_images_dim_y, my_small_images_dim_x, 3))
#     landmarks = Input(shape=(10,))
#
#     extract1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(images_small)
#
#     pool1 = MaxPooling2D()(extract1)
#     drop1 = Dropout(.2)(pool1)
#     extract2 = Conv2D(20, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(drop1)
#     pool2 = MaxPooling2D()(extract2)
#     drop2 = Dropout(.2)(pool2)
#     extract3 = Conv2D(8, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(drop2)
#     pool3 = MaxPooling2D()(extract3)
#
#
#     flat = Flatten()(pool3)
#     inferred_features = Dense(80, activation='sigmoid')(flat)
#
#     landmarks1 = Dense(60, activation='sigmoid')(landmarks)
#
#     landmarks_and_inferred_features = keras.layers.concatenate([landmarks1, inferred_features])
#     batchnormal = BatchNormalization()(landmarks_and_inferred_features)
#
#     flattened_medium_sized_generated_image = Dense(my_small_images_dim_y * 1 * my_small_images_dim_x * number_of_extra_inferred_dimensions, activation='relu')(batchnormal)
#     rectangular_medium_sized_image = Reshape((my_small_images_dim_y , my_small_images_dim_x , number_of_extra_inferred_dimensions))(flattened_medium_sized_generated_image)
#
#     smush = keras.layers.concatenate([rectangular_medium_sized_image, images_small])
#
#     bigger_image = UpSampling2D()(smush)
#     big_conv1 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(bigger_image)
#     big_conv2 = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(big_conv1)
#
#
#
#
#     full_resolution_inferred = UpSampling2D()(big_conv2) # The third dimensions, usually 3 for RGB, is not RGB but rather n-dimensional feature space
#
#
#     n_conv1 = Conv2D(22, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(full_resolution_inferred)
#     n_conv2 = Conv2D(8, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(n_conv1)
#     final_image = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(n_conv2)
#
#     model = Model(inputs=[images_small, landmarks], outputs=final_image)
#     optim = Adam(lr=0.0005, beta_1=0.91)
#
#     model.compile(loss='mse', optimizer=optim)
#
#     return model

def get_upsampled_training_data_from_raw_batch(my_random_indices, my_small_images_dim_x, my_small_images_dim_y,
                                               my_large_images_dim_x, my_large_images_dim_y, my_small_images_dir,
                                               my_large_images_dir):

    scale_factor = int(my_large_images_dim_x / my_small_images_dim_x)

    x_np = np.zeros((my_random_indices.shape[0], my_small_images_dim_y, my_small_images_dim_x, 3), dtype=np.float)
    y_np = np.zeros((my_random_indices.shape[0], my_large_images_dim_y, my_large_images_dim_x, 3), dtype=np.float)
    x_small_np = np.zeros((my_random_indices.shape[0], my_small_images_dim_y, my_small_images_dim_x, 3), dtype=np.float)

    for i in range(0, my_random_indices.shape[0]):
        file_num = my_random_indices[i]
        small_file_name = my_small_images_dir + get_full_file_string_from_int(file_num)
        large_file_name = my_large_images_dir + get_full_file_string_from_int(file_num)
        small_image = cv2.imread(small_file_name, 3)  # image.shape is (27, 22, 3)
        large_image = cv2.imread(large_file_name, 3)  # image.shape is (216, 176, 3)

        small_image = small_image / 255.0
        large_image = large_image / 255.0

        x_small_np[i] = small_image.copy()
        #upscaled_image = cv2.resize(small_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        #upscaled_image = cv2.resize(small_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

        x_np[i] = small_image  # upscaled_image
        y_np[i] = large_image     # input x is upsampled, bicubic images of faces, y (truth) is high resolution image

    return x_np, y_np, x_small_np

def train_upsampled_model(my_upsampled_model, my_trained_landmarks_model, my_start_index, my_upsampled_batch_epochs,
                          my_upsampled_batch_size, my_upsampled_number_of_images_to_use, my_small_images_dim_x,
                          my_small_images_dim_y, my_large_images_dim_x, my_large_images_dim_y, my_small_images_dir,
                          my_large_images_dir, my_system_mode):

    for batch_epoch in range(0, my_upsampled_batch_epochs):
        if batch_epoch % 100:
            print('Batch Epoch, #', str(batch_epoch))
        random_indices = np.random.randint(my_start_index, my_upsampled_number_of_images_to_use, size=my_upsampled_batch_size)
        # Get a random batch of indices to the new array

        x, y, x_small = get_upsampled_training_data_from_raw_batch(random_indices, my_small_images_dim_x,
                                                                   my_small_images_dim_y, my_large_images_dim_x,
                                                                   my_large_images_dim_y, my_small_images_dir,
                                                                   my_large_images_dir)

        #x_landmarks = my_trained_landmarks_model.predict(x_small)  # we use the previously trained  (fully supervised)
        x_landmarks = my_trained_landmarks_model.predict(x)
        # landmarks model, to make some predictions that are then used (here) as input to the upsampling model


        loss = my_upsampled_model.train_on_batch([x, x_landmarks], y)

        print('Batch Loss is: ',loss)
        if (batch_epoch % 500 == 0) or (batch_epoch in [0, 10, 100, 500]):
            print('Loss',loss)

            image_numbers = np.array([ 35, 40, 20, 4, 111,126,138,169,172,255])
            #image_numbers = np.array([26, 35, 88, 4])
            low_res_images, upsampled_images = load_predict_and_save_images(image_numbers, my_upsampled_model,
                                                                            my_trained_landmarks_model, True,
                                                                            my_small_images_dim_x, my_small_images_dim_y,
                                                                            my_large_images_dim_x,my_large_images_dim_y,
                                                                            my_small_images_dir, my_large_images_dir,
                                                                            batch_epoch, loss, my_system_mode)
            #json_name = '/home/foo/data/celeba/models/model_architecture-' + str(batch_epoch) + '.json'

            dir_prefix = '/output/upsampled_model_'

            if my_system_mode == 'local_cpu':
                dir_prefix = '/home/foo/data/celeba/models/upsampled_model_'

            h5_name = dir_prefix + str(batch_epoch) + '-' + str(loss) + '.h5'


            my_upsampled_model.save(h5_name, overwrite=True, include_optimizer=True)


    return my_upsampled_model

#-------------------------
super_seed = 1234
random.seed(super_seed)
np.random.seed(super_seed)

system_mode = 'floyd_gpu' #''local_cpu' # else 'floyd_hub_gpu'
#system_mode = 'local_cpu'

#floyd run --data photox/datasets/celeba_images_low_res_floyd/1:low --data photox/datasets/celeba_images_high_res_floyd/1:high --data photox/datasets/celeba_misc_floyd/3:misc --gpu+ "python super_resolution_celeba_upsampling.py"
#photox/datasets/celeba_misc_floyd/3
#photox/datasets/lowest/1
# photox/datasets/celeba_misc_floyd/4

small_images_dir = '/lowest/'#'/low/'  # these are the floyd hub dataset mount points
large_images_dir = '/high/'
landmark_model = '/misc/super_res_landmark_model_floyd_lowest.h5' # my /misc data set contains various files, hdf5s, txt and configs
upsampled_batch_size = 80
upsampled_number_of_images_to_use = 200000
upsampled_batch_epochs = 8000

if system_mode == 'local_cpu':
    small_images_dir = '/home/foo/data/celeba/celeba_images_lowest_res/'
    large_images_dir = '/home/foo/data/celeba/celeba_images_high_res/'
    landmark_model = '/home/foo/data/celeba/models/super_res_landmark_model_floyd_lowest.h5'
    upsampled_batch_size = 2
    upsampled_number_of_images_to_use = 2000
    upsampled_batch_epochs = 1000

start_index = 300 #

small_images_dim_x = 22
small_images_dim_y = 27

large_images_dim_x = 176
large_images_dim_y = 216
#----------------------------

# f = h5py.File(landmark_model, 'r+') # If loading a modeling locally whiuch had been trained elsewhere, you may need to run this hack
# del f['optimizer_weights']
# f.close()

trained_landmark_model = load_model(landmark_model)  # leakyrelu causes load error doiscussed here:   https://github.com/keras-team/keras/issues/7107
# ------>

# Beginning the upsampled and inferred high resolution model.

upsampled_model = make_upsampled_model_functional(small_images_dim_x, small_images_dim_y, large_images_dim_x, large_images_dim_y)

trained_upsampled_model = train_upsampled_model(upsampled_model, trained_landmark_model, start_index,
                                                upsampled_batch_epochs, upsampled_batch_size,
                                                upsampled_number_of_images_to_use, small_images_dim_x,small_images_dim_y,
                                                large_images_dim_x,large_images_dim_y,small_images_dir,large_images_dir, system_mode)
#trained_upsampled_model.save('/home/foo/data/celeba/models/upsampled_model_final.h5')