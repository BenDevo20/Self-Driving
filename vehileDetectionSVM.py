"""
Vehicle detection using SVM -- will need to change the image input and resizing numbers for live video feed but
this is basically a working model for vehicle detection

HOG - histogram of oriented gradients - the steps of how package is used is as follows:

1. Normalizing the image prior to description.
2. Computing gradients in both the x and y directions.
3. Obtaining weighted votes in spatial and orientation cells.
4. Contrast normalizing overlapping spatial cells.
5. Collecting all Histograms of Oriented gradients to form the final feature vector.

i keep getting a fucking syntax error when I import seaborn so comment seaborn back in
"""

import cv2
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import glob
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# importing training dataset of cars and not cars from file that is in project directory
# glob finds all the pathnames matching a specified pattern according to the rules of the machine
car = glob.glob('data/car/**/*.png')
noCar = glob.glob('data/no_car/**/*.png')
image_color = cv2.imread(car[300])
# convert image to gray scale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Getting the hog features -- these will change depending on method use for detection for RasPiCar
features, hog_image = hog(image_gray,
                          orientations=11,
                          pixels_per_cell=(16,16),
                          cells_per_block=(2, 2),
                          transform_sqrt=False,
                          visualize=True,
                          feature_vector = True)

# extracting features and creating training data
car_hog_acc = []
for i in car:
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    car_hog_feature, car_hog_img = hog(image_color[:,:,0],
                                       orientations=11,
                                       pixels_per_cell=(16, 16),
                                       cells_per_block=(2, 2),
                                       transform_sqrt=False,
                                       visualize=True,
                                       feature_vector=True)
    car_hog_acc.append(car_hog_feature)
# vstack stack arrays in sequence vertically -- good practice for pixel-data --- data with max(3) dims
X_car = np.vstack(car_hog_acc).astype(np.float64)
y_car = np.zeros(len(X_car))

nocar_hog_acc = []
for i in noCar:
    image_color = mpimg.imread(i)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    nocar_hog_feature, car_hog_img = hog(image_color[:, :, 0],
                                         orientations=11,
                                         pixels_per_cell=(16, 16),
                                         cells_per_block=(2, 2),
                                         transform_sqrt=False,
                                         visualize=True,
                                         feature_vector=True)

    nocar_hog_acc.append(nocar_hog_feature)
X_nocar = np.vstack(nocar_hog_acc).astype(np.float64)
y_nocar = np.zeros(len(X_nocar))
# creating variables that will be used in test and training data for SVM
X = np.vstack((X_car, X_nocar))
y = np.hstack((y_car, y_nocar))

# creating the svm model classifier -- training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
svc_model = LinearSVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

# want to computer confusion matrix to evaluate the accuracy of the classification -- pos, fal and falpos
cm = confusion_matrix(y_test, y_predict)
# outputting heatmap using seaborn
sns.heatmap(cm, annot=True, fmt="d")
print(classification_report(y_test,y_predict))

model_predict = svc_model.predict(X_test[0:50])
model_truelabel = y_test[0:50]

"""
Improving the model using gridSearchCv which implements a fit and score method -- which methodically builds
and evaluates a model for each combination of algorithm parameters specific in a grid 

parameters for support vector classifier (SVC) are C, kernel and gamma 
search the hyper-parameter space for the best cross validation score  
"""
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid, refit=True, verbose=4)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)

grid_predict = grid.predict(X_test)
cm = confusion_matrix(y_test, grid_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, grid_predict))

# testing the model with images of cars that are in directory -- again this will change with video - but same concept
test_image = mpimg.imread('testImage.jpg')
plt.imshow(test_image)
test_image = test_image.astype(np.float32)/225

h_start = 300
h_stop = 480
pixels_in_cell = 16
Hog_orient = 11
cell_block = 2
cell_step = 3
masked_region = test_image[h_start:h_stop,:,:]
plt.imshow(masked_region)
# sizing factor will change -- will probably have to be smaller increments with video
resizing_factor = 2
masked_region_shape = masked_region.shape
# resizing length and width of vectorized image by defined factor size
L = masked_region_shape[1]/resizing_factor
W = masked_region_shape[0]/resizing_factor
masked_region_resized = cv2.resize(masked_region, (np.int(L), np.int(W)))
masked_region_resized_R = masked_region_resized[:,:,0]

n_blocks_x = (masked_region_resized_R.shape[1] // pixels_in_cell) + 1
n_blocks_y = (masked_region_resized_R.shape[0] // pixels_in_cell) + 1
masked_region_hog_feature_all, hog_img = hog(masked_region_resized_R,
                                             orientations = 11,
                                             pixels_per_cell = (16, 16),
                                             cells_per_block = (2, 2),
                                             transform_sqrt = False,
                                             visualize = True,
                                             feature_vector = False)

# nfeat_per_block = orientations * cells_in_block **2
blocks_in_window = (64 // pixels_in_cell) - 1

steps_x = (n_blocks_x - blocks_in_window) // cell_step
steps_y = (n_blocks_y - blocks_in_window) // cell_step

rectangles_found = []

for xb in range(steps_x):
    for yb in range(steps_y):
        y_position = yb * cell_step
        x_position = xb * cell_step

        hog_feat_sample = masked_region_hog_feature_all[y_position: y_position + blocks_in_window,
                          x_position: x_position + blocks_in_window].ravel()
        x_left = x_position * pixels_in_cell
        y_top = y_position * pixels_in_cell
        print(hog_feat_sample.shape)

        # predict using trained SVM
        test_prediction = svc_model.predict(hog_feat_sample.reshape(1, -1))

        # test_prediction = grid.predict(hog_feat_sample.reshape(1,-1))
        if test_prediction == 1:
            rectangle_x_left = np.int(x_left * resizing_factor)
            rectangle_y_top = np.int(y_top * resizing_factor)
            window_dim = np.int(64 * resizing_factor)
            rectangles_found.append(((rectangle_x_left, rectangle_y_top + h_start),
                                     (rectangle_x_left + window_dim, rectangle_y_top + window_dim + h_start)))

# drawing on the vehicles that are found in the image
Image_with_Rectangles_Drawn = np.copy(test_image)

# using openCv to draw images
for rectangle in rectangles_found:
    cv2.rectangle(Image_with_Rectangles_Drawn, rectangle[0], rectangle[1], (0, 255, 0), 20)

# outputting image with rect
plt.imshow(Image_with_Rectangles_Drawn)
