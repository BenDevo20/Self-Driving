import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pickle
import cv2
import numpy as np
import pandas as pd
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
anno = pd.read_csv(DATA_DIR + '/' + 'allAnnotations.csv')

#Dictionary with numerical values
num_encode = {'signalAhead': 0, 'stop': 1, 'stopAhead': 2, 'speedLimitUrdbl': 3, 'rightLaneMustTurn': 4, 'speedLimit40': 5, 'pedestrianCrossing': 6, 'keepRight': 7, 'speedLimit45': 8, 'speedLimit35': 9, 'speedLimit25': 10, 'addedLane': 11, 'merge': 12, 'school': 13, 'yield': 14, 'turnRight': 15, 'speedLimit65': 16, 'schoolSpeedLimit25': 17, 'slow': 18, 'truckSpeedLimit55': 19, 'yieldAhead': 20, 'intersection': 21, 'speedLimit50': 22, 'rampSpeedAdvisory50': 23, 'noRightTurn': 24, 'laneEnds': 25, 'rampSpeedAdvisory45': 26, 'dip': 27, 'noLeftTurn': 28, 'zoneAhead45': 29, 'rampSpeedAdvisory20': 30, 'rampSpeedAdvisoryUrdbl': 31, 'turnLeft': 32, 'speedLimit55': 33, 'doNotPass': 34}
print(len(num_encode))

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

#Crop Image
def crop_image(img_dest, img):
    #Pull image row data
    img_set = []
    sign_set = []
    row_idx = anno.index[anno['Filename'] == img_dest].tolist()
    for idx in row_idx:
        sign = anno.iloc[idx]['Annotation tag']
        top_x = anno.iloc[idx]['Upper left corner X']
        top_y = anno.iloc[idx]['Upper left corner Y']
        bot_x = anno.iloc[idx]['Lower right corner X']
        bot_y = anno.iloc[idx]['Lower right corner Y']
        img_crop = img[top_y:bot_y, top_x:bot_x]
        img_crop = cv2.resize(img_crop, (imageDimensions[0], imageDimensions[1]))
        img_set.append(img_crop)
        sign_val = num_encode.get(sign)
        sign_set.append(sign_val)
    return [img_set, sign_set]

for vid, annotation in zip(dir_list, frameAnnotations):
    picList = os.listdir(DATA_DIR + '/' + vid + '/' + annotation)
    print('Processing vid:' + vid)
    for j in picList:
        if pic(j):
            url = vid + '/' + annotation + '/' + j
            curImg = cv2.imread(DATA_DIR + '/' + url)
            img_set = crop_image(url,curImg)
            images.extend(img_set[0])
            signLabels.extend(img_set[1])
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

num_class = len(countSigns)
print(countSigns)
print(len(countSigns))
'''
#print(countSigns)
print({k: v for k, v in sorted(countSigns.items(), key=lambda item: item[1])})
print(countSigns)
plt.bar(*zip(*countSigns.items()))
plt.show()
'''

#PreProcessing Data
def preProcessing(img):
    #Gray Scale Conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Lighting Equalization
    img = cv2.equalizeHist(img)
    #Normalize to 0-1
    img = img/255
    return img

'''
cv2.imshow("Preprocess crap", X_train[20])
cv2.waitKey(0)
print(X_train[20].shape)
'''

#Gray Scale Everything
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

#Adding Depth
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2], 1)

y_train = to_categorical(y_train, num_classes = num_class)
y_test = to_categorical(y_test, num_classes = num_class)
y_validation = to_categorical(y_validation, num_classes = num_class)

#Augment Data Sets
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=20)
dataGen.fit(X_train)

#Making Model
def CNN():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape = (imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))

    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))

    model.add(MaxPooling2D(pool_size = sizeOfPool))

    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))

    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation = 'softmax'))
    model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = CNN()
print(model.summary())


batchSizeval = 1
epochs = 150
steps_epoch = 2000

model.fit_generator(dataGen.flow(X_train, y_train, batch_size = batchSizeval),
                    steps_per_epoch=steps_epoch,
                    epochs = epochs,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)

#Store as Pickle Object
pickle_out = open('model_trained.p', 'wb')
pickle.dump(model,pickle_out)
pickle_out.close()
cv2.waitKey(0)



