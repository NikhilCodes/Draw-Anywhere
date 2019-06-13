from numpy import array, expand_dims, append

from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from keras.callbacks import TensorBoard

from scipy.stats import zscore

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Crunching Data to Feed into Model
x_train, x_test = expand_dims(x_train/255, axis=3), expand_dims(x_test/255, axis=3)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# Model Structure
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', kernel_regularizer='l2'),
    Conv2D(32, kernel_size=(3,3), activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer='l2'),
    Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),

    Dense(32, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Config. Tensorboard
callbacks = [TensorBoard(log_dir='logs')]
model.build(input_shape=(None,28,28,1))
model.summary()
# Fitting Model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test,y_test), callbacks=callbacks, verbose=1)
model.save('MODELS/digit_recog_model.h5')
