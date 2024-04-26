from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#Loading train and test data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
num_classes = 10
#reshape train and test data into a matrix
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#normalize the data
x_train /= 255
x_test /= 255

#turn the y matrix into categorical type
y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)


def create_model(epochs=10):
    #define the model
    model = Sequential()
    model.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    #display the model
    model.summary()
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    #fit and return model
    history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test))
    return model

if __name__=="__main__":
    #create and save the model
    model=create_model(10)
    model.save('mnist_model.h5')
    