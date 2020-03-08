import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

#Load Data
DATA_DIR = "Traffic DATA"
images = []
signLabels = []
dir_list = os.listdir(DATA_DIR)[11:23]
imageDimensions = (64, 64, 3)

def label_image(img):
    image_label = img.split('_')[-3]
    return image_label

def pic(img):
    image_extension = img.split('.')[-1]
    if image_extension == 'png':
        return True
    else:
        return False

frameAnnotations = []

for i in dir_list:
    frameAnnotation = os.listdir(DATA_DIR + "/" + i)[0]
    frameAnnotations.append(frameAnnotation)

for vid, annotation in zip(dir_list, frameAnnotations):
    picList = os.listdir(DATA_DIR + '/' + vid + '/' + annotation)
    print('Processing vid:' + vid)
    for j in picList:
        if pic(j):
            curImg = cv2.imread(DATA_DIR + '/' + vid + '/' + annotation + "/" + j)
            curImg = cv2.resize(curImg, (64, 64))
            images.append(curImg)
            signLabels.append(label_image(j))
        else:
            continue

images = np.array(images)
signLabels = np.array(signLabels)

print(images.shape)
print(signLabels.shape)

#Splitting Data
X_train, X_test, y_train, y_test = train_test_split(images, signLabels, test_size = 0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size= 0.2)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

#Looking at Data Distribution
countSigns ={}
for i in y_train:
    if i in countSigns:
        countSigns[i] += 1
    else:
        countSigns[i] = 0

print(countSigns)

#plt.bar(*zip(*countSigns.items()))
#plt.show()


#PreProcessing Data
def preProcessing(img):
    #Gray Scale Conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Lighting Equalization
    img = cv2.equalizeHist(img)
    #Normalize to 0-1
    img = img/255
    return img

#Gray Scale Everything
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

#Adding Depth

#Augment Data Sets

#Making Model


def Model(self):
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape = (imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(self.noOfFilters, self.sizeOfFilter1, activation='relu')))

    model.add(MaxPooling2D(pool_size = sizeOfPool))

    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))

    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(countSigns), activation = 'softmax'))
    model.compile(Adam(lr = 0.001), loss = 'categorical crossentropy', metrics = ['accuracy'])

    return model




