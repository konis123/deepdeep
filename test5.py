# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.cond.control_flow_ops = tf

#with open('small_train_traffic.p', mode='rb') as f:
#    data = pickle.load(f)

#X_train, y_train = data['features'], data['labels']

from keras.datasets import mnist
from keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

import keras
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Build Convolutional Pooling Neural Network with Dropout in Keras Here
model = Sequential()

model.add(Convolution2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(MaxPooling2D((2, 2), padding="same"))
model.add(Activation('relu'))
'''
model.add(Convolution2D(64, (3, 3), input_shape=(None, 14, 14, 1), padding="same"))
model.add(MaxPooling2D((2, 2), padding="same"))
model.add(Activation('relu'))
'''
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# preprocess data
X_normalized = np.array(x_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)
