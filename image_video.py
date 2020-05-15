import cv2
import numpy as np
import pickle

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = .80
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
num_encode = {'signalAhead': 0, 'stop': 1, 'stopAhead': 2, 'speedLimitUrdbl': 3, 'rightLaneMustTurn': 4, 'speedLimit40': 5, 'pedestrianCrossing': 6, 'keepRight': 7, 'speedLimit45': 8, 'speedLimit35': 9, 'speedLimit25': 10, 'addedLane': 11, 'merge': 12, 'school': 13, 'yield': 14, 'turnRight': 15, 'speedLimit65': 16, 'schoolSpeedLimit25': 17, 'slow': 18, 'truckSpeedLimit55': 19, 'yieldAhead': 20, 'intersection': 21, 'speedLimit50': 22, 'rampSpeedAdvisory50': 23, 'noRightTurn': 24, 'laneEnds': 25, 'rampSpeedAdvisory45': 26, 'dip': 27, 'noLeftTurn': 28, 'zoneAhead45': 29, 'rampSpeedAdvisory20': 30, 'rampSpeedAdvisoryUrdbl': 31, 'turnLeft': 32, 'speedLimit55': 33, 'doNotPass': 34}

#Import Trained Model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo):
    for key, value in num_encode.items():
        if classNo == value:
            return key

while True:
    success, imgOriginal = cap.read()

    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (64, 64))
    img = preprocess(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1,64,64,1)
    cv2.putText(imgOriginal, "Class:  ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability: ", (20,75), font, 0.75, (255, 0,0), 2, cv2.LINE_AA)

    prediction = model.predict(img)
    classIndex = model.predict_classes(img)
    probVal = np.amax(prediction)

    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probVal*100 , 2)) +"%", (180, 75), font, 0.75, (255, 0,0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break