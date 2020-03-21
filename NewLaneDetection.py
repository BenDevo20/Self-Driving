import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


# canny function -- calculates the derivative in both x and y directions -- see changes in value
# large derivatives = sharp changes in slope | small derivatives = slight change in slope | slope == lane line
def canny_edge(frame):
    # convert the image to grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # reduce noise from the images
    # gaussian blur - reduces computational stress [arguments(src, dst, kernel size, standard dev in x)]
    #blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    lower = np.array([60, 42, 40])
    upper = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # finding edge in an image -- [arguments(image, threshold1, threshold2)
    canny = cv2.Canny(mask, 200, 400)

    return canny

# locating region of interest -- cropping image to lower portion -- detected camera for lane will have lower angle
def region_interest(frame):
    height, width = frame.shape[1], frame.shape[0]
    polygons = np.array([[(0, height*1/3), (width, height*1/3), (width, height), (0, height)]], np.int32)
    mask = np.zeros_like(frame)
    # filling the poly function deals with multi polygon
    # built in opencv function -- last argument is computer max for color recognition by computers 256-1
    cv2.fillPoly(mask, polygons, 255)
    # bitwise operation between canny frame and mask image
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame

# finding the coordinates of our road lane -- using polar coordinates - more accurate
def create_cord(frame, line_params):

    slope, intercept = line_params
    y1 = frame.shape[0]
    y2 = int(y1*1/2)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)

    return np.array([x1, y1, x2, y2])

# differentiate left and right road lanes with the help of positive and negative slopes
# negative slope - road lane belongs to the left side of the vehicle
# positive slope - road lane belongs to the right side of the vehicle
def avg_slope_int(frame, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # fitting the polynomial, intercept and slope
            param = np.polyfit((x1, x2), (y1, y2), 1)
            slope = param[0]
            y_intercept = param[1]
            if slope < 0:
                left_fit.append((slope, y_intercept))
            elif slope > 0:
                right_fit.append((slope, y_intercept))

        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)
        left_line = None if not left_fit else create_cord(frame, left_fit_avg)
        right_line = None if not right_fit else create_cord(frame, right_fit_avg)
        return np.array([left_line, right_line])

# fitting the coordinates into actual frame and return frame with the detected line(lane)
def display_lines(frame, lines):
    line_frame = np.zeros_like(frame)
    if lines[0] is not None and lines[1] is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
    return line_frame

# read and decode video frames - HoughLine method - call all the functions and show lanes on output
# have camera detect webcam or raspberry camera module
#cap = cv2.VideoCapture('test2.avi')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    canny_image = canny_edge(frame)
    #cv2.imshow("canny", canny_image)
    #cv2.waitKey(0)
    cropped_image = region_interest(canny_image)
    #cv2.imshow("cropped", cropped_image)
    #cv2.waitKey(0)

    linesp = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, minLineLength=20, maxLineGap=5)
    average_lines = avg_slope_int(frame, linesp)

    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image,1,1)
    cv2.imshow('results', combo_image)
"""
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""