import cv2
import numpy as np
def skin_detector(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, (0, 15, 0), (17, 170, 255))
    y_cr_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_cr_cb_mask = cv2.inRange(y_cr_cb_img, (0, 135, 85), (255, 180, 135))
    mask = cv2.bitwise_and(y_cr_cb_mask, hsv_mask)
    result = cv2.bitwise_not(mask)
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)
    result = cv2.dilate(result, kernel, iterations=1)
    return result