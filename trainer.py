import matplotlib
# allows export plots as image files to disk
matplotlib.use('Agg')

from trafficsignnet import TrafficSignNet
from keras.preprocessing.image import ImageDataGenerator
# optimization and noe-hot encoding
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
# parsing command line arguments
import argparse
import random
# grab operating system path separator
import os

# function to load data from disk
def load_split(basePath, csvPath):
    # initialize the list data and labels
    data = []
    labels = []
    # load contents of file
    rows = open(csvPath).read().strip().split('/n')[1:]
    random.shuffle(rows)
    """
    Loops over the rows. Displays status update to terminal every 1000th image 
    extract classID and image path from the row 
    derive full path to the imageFile and load image to scikit 
    """
    for (i, row) in enumerate(rows):
        if i > 0 and i % 1000 == 0:
            print('[INFO] processed {} total images'.format(i))
        # splits rows into components and grabs class ID
        (label, imagePath) = row.strip().split(',')[-2:]
        # derive the full path to the image file and load it
        imagePath = os.path.sep.join([basePath,imagePath])
        image = io.imread(imagePath)
        """
        resize the images, ignoring aspect ratio -- improves clarity of lower res images 
        for training model 
        """
        image = transform.resize(image,(32,32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)
        # update the list of data and label
        data.append(image)
        labels.append(int(label))
    # converts data and labels to np.array
    data = np.array(data)
    labels = np.array(labels)
    # return tuple of data and labels
    return (data,labels)

""" 
parsing command line arguments. dataset - the path to GTSRB dataset. Model -- path/filename of output model 
plot -- path to traing history plot 
"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", '--dataset', required=True, help='path to input GTSRB')
ap.add_argument("-m", '--model', required=True, help='path to output model')
ap.add_argument("-p", '--plot', type=str, default='plot.png', help='[ath to training history plot')
args = vars(ap.parse_args())
# intialize number of epochs to train and batch size
num_epochs = 30
init_lr = 1e-3
bs = 64

# load label names from csv
labelNames = open('signnames.csv').read().strip().split('\n')[1:]
labelNames = [l.split(',') for l in labelNames]

# load and preprocess the data
# derive path to the traing and test data -- included in project directory
trainPath = os.path.sep.join([args['dataset'], 'Train.csv'])
testPath = os.path.sep.join([args['dataset'], 'Test.csv'])
# load the training and test data
print('[INFO] loading training and test data..')
(trainX, trainY) = load_split(args['dataset'], trainPath)
(testX,testY) = load_split(args['dataset'], testPath)

# scale data
trainX = trainX.astype('float32')/225.0
testX = testX.astype('float32')/225.0

# encode training data and testing labels
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

# skew in the labeled data assigning a weight to each class during training
classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

# construct the image generator for data augmentation
# random rotating, zoom, shift and shear settings -- no objects will be flipped = false
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

# initialize the optimizer and compile model
# use Adam optimizer and learning rate decay
print('INFO: Compiling model...')
opt = Adam(lr=init_lr, decay=init_lr/(num_epochs*0.5))
model = TrafficSignNet.build(width=32,height=32,depth=3,
                             classes=numLabels)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# training the network
print('INFO: Training the network....')
H = model.fit_generator(
    aug.flow(trainX,trainY,batch_size=bs),
    validation_data=(testX,testY),
    steps_per_epoch=trainX.shape[0]//bs,
    epochs=num_epochs,
    class_weight=classWeight,
    verbose=1
)

# evaluating the network, print classification report, serializes keras model to disk
print('[INFO] evaluating the network...')
predictions = model.predict(testX, batch_size=bs)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=labelNames))
model.save(args['model'])

# plot the training loss and accuracy
N = np.arange(0, num_epochs)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='trainLoss')
plt.plot(N, H.history['valLoss'], label='valLoss')
plt.plot(N, H.history['accuracy'], label='trainAcc')
plt.plot(N, H.history['valAccuracy'], label='valAcc')
plt.title('Training loss and accuracy on Dataset ')
plt.xlabel('Eopch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='Lower left')
plt.savefig(args['plot'])
