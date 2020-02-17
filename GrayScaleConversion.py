import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

# loading image that is directory of project -- any image or stream of images can be loaded as function
image_color = mpimg.imread('image_lane_c.jpg')
plt.imshow(image_color)
shape = image_color.shape
print(shape)

# converting the image to grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray, cmap='gray')
new_shape = image_gray.shape
# notice the difference in size between the grayscale image and the color image
print(new_shape)

# if we want to sort out the white color out of the image 
image_copy = np.copy(image_gray)
# this for any value that is not white 
image_copy[(image_copy[:,:] < 250)] = 0 
plt.imshow(image_copy, cmap='gray')

# saving the grayscale image
cv2.imwrite('image_lane_c.jpg', image_gray)
