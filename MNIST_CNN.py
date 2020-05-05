# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:38:02 2020

@author: Mihul

MNIST classification CNN
"""
# Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
import json

# Download a MNIST dataset from tf.datasets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Plot one image
import matplotlib.pyplot as plt
#%matplotlib inline
im_index = 100
print(y_train[im_index])
plt.imshow(X_train[im_index], cmap = "gray")

X_train.shape
X_test.shape

# Normalizing and standardizing our images
from sklearn.preprocessing import StandardScaler, Normalizer

# Standardizing
std = StandardScaler()
plt.imshow(X_train[1500], cmap = "gray")
X_train_std = std.fit_transform(X_train.reshape(60000, -1)).reshape(60000, 28, 28)
plt.imshow(X_train_std[1500], cmap='gray')

plt.imshow(X_test[1500], cmap = "gray")
X_test_std = std.transform(X_test.reshape(10000, -1)).reshape(10000, 28, 28)
plt.imshow(X_test_std[1500], cmap='gray')

# Normalizing does nothing
norm = Normalizer()
X_train_norm = norm.fit_transform(X_train.reshape(60000, -1)).reshape(60000, 28, 28)
plt.imshow(X_train_norm[1500], cmap='gray')
 
X_test_norm = norm.transform(X_test.reshape(10000, -1)).reshape(10000, 28, 28)
plt.imshow(X_test_norm[1500], cmap='gray')

# Reshaping images so we can use Conv2D layer, which requires 4-dim array as input
X_train = X_train_std.reshape(60000, 28, 28, 1)
X_test = X_test_std.reshape(10000, 28, 28, 1)

# Building a CNN with 2 convolutional layers
clf = Sequential()
clf.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))
clf.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))
clf.add(Flatten())
clf.add(Dense(units = 128, activation = 'relu'))
clf.add(Dropout(0.2))    
clf.add(Dense(units = 10, activation = tf.nn.softmax))

clf.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
clf.fit(X_train, y_train, epochs = 15)

clf.evaluate(X_test, y_test)


# _____ Save and load the model _______

# Save the whole model
clf.save('MNIST_clf_whole.h5')

# Save only arcitecture
clf_json = clf.to_json(indent = 4)
with open('MNIST_CLF.json', 'w') as file:
    json.dump(clf_json, file)
      
# Load the whole model from file
from tensorflow.keras.models import load_model
clf = load_model('MNIST_clf_whole.h5')
clf.evaluate(X_test, y_test)

# _____ Let's try to predict the written number from independent source _________________

from skimage import data
from skimage.color import rgb2gray
img = data.astronaut()
plt.imshow(img)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap = 'gray')
type(img_gray)
img_gray.shape

# Download some handwritten numbers images
# Rename all .png images for convinience
import os
i = 0 # Start number
ext = 'png'
direct = 'numbers/test'
images = {}
for file in os.listdir(direct):
    if file.endswith(ext):
        os.rename(f'{direct}/{file}', f'{direct}/{i}.{ext}')
        #images[file] = 0
        i += 1

# Reshape all test images to 28x28 format and turn to gray scale and invert black and white
import cv2

for file in os.listdir(direct):
    img = cv2.imread(f'{direct}/{file}', cv2.IMREAD_UNCHANGED)
    resized = cv2.cvtColor(cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    # write an image back
    cv2.imwrite(f'{direct}/{file}', ~(resized))
    # write each image to array
    images[file] = resized

#cv2.imshow("Resized image", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.imshow(img)

# Try to make one big matrix from all images
img_arr = (255 - images['0.png']).reshape(1, 28, 28)
for key in images:
    if key != '0.png':
        img_arr = np.concatenate((img_arr, (255 - images[key]).reshape(1, 28, 28)), axis = 0)

# Try to standardize images from array
plt.imshow(img_arr[5], cmap = "gray")
img_arr = std.transform(img_arr.reshape(9, -1)).reshape(9, 28, 28)
plt.imshow(img_arr[5], cmap='gray')

# Make a prediction for standardized images
images_pred = clf.predict(img_arr.reshape(9, 28, 28, 1))

# Make a definite prediction
predictions = [np.argmax(images_pred[i]) for i in range(9)]
images_pred = {}
i = 0
for key in images:
    images_pred[key] = predictions[i]
    i += 1

# 7 out of 9 true predictions