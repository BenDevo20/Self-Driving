from keras.models import Sequential
from keras.layers import BatchNormalization,Conv2D,MaxPooling2D, Activation,Flatten,Dropout,Dense

"""
defines build method -- accepts four parameters: 
dimensions(height width), depth and number of classes
convolution neural net to recognize traffic signs 
"""
class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initializing the model -- input shape to be channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # uses 5*5 kernel to learn larger images -- better res for distinguishing characteristics
        model.add(Conv2D(8,(5,5),padding='same',input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # deeper network stacking two sets of layers before max-pooling to reduce volume dimensionality
        model.add(Conv2D(16,(3,3),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # second set -> pooling
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # head of network - two sets of fully connected layers and softmax classifier
        model.add(Flatten())
        model.add(Dense(18))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # second set of fc relu layers
        model.add(Flatten())
        model.add(Dense(18))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # dropout used to avoid overfitting -- more generalizable model
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        #return the constructed network
        return model