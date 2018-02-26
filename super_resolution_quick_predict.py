import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Merge, MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from keras import backend as K
from keras import initializers
import cv2
from keras.models import load_model
from keras.models import model_from_json



small_image_name = '/home/foo/data/celeba/test/fake-sm.jpg'
#small_image_name = '/home/foo/data/celeba/celeba-images-22-27/000040.jpg'

small_image = cv2.imread(small_image_name, 3)
si = small_image.copy()
bicubic_image = cv2.resize(si, None, fx=8, fy=8, interpolation=cv2.INTER_LANCZOS4)

bicubic_image = np.reshape(bicubic_image, (1, bicubic_image.shape[0], bicubic_image.shape[1], bicubic_image.shape[2]))
#small_image = np.reshape(small_image, (1, small_image.shape[0], small_image.shape[1], small_image.shape[2]))  # keras' model.predict is expecting a 'batch' of images, we are just using 1 here

small_image = small_image / 255.0
bicubic_image = bicubic_image / 255.0

trained_landmark_model = load_model('/home/foo/data/celeba/models/landmark_model_batch_2560_loss_0.000165728.h5')
upsampled_model = load_model('/home/foo/data/celeba/models/my_upsampled_model_model_2600.h5')

small_image = np.reshape(small_image, (1, small_image.shape[0], small_image.shape[1], small_image.shape[2]))  # keras' model.predict is expecting a 'batch' of images, we are just using 1 here

landmark_features = trained_landmark_model.predict(small_image)

upscaled_prediction = upsampled_model.predict([bicubic_image, landmark_features])
upscaled_prediction = upscaled_prediction * 255.0
upscaled_prediction = np.reshape(upscaled_prediction,(upscaled_prediction.shape[1], upscaled_prediction.shape[2], upscaled_prediction.shape[3]))
upscaled_prediction_filename = '/home/foo/data/celeba/test/upscaled_prediction.png'
cv2.imwrite(upscaled_prediction_filename, upscaled_prediction)




#print(upscaled_prediction)



