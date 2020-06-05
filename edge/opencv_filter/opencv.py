import time

import cv2
import math
import numpy as np


def skel(orig_img):
    start_time = time.time()
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

    return skel, time.time() - start_time


def _ycc(r, g, b):
    y = .299 * r + .587 * g + .114 * b
    cb = 128 - .168736 * r - .331364 * g + .5 * b
    cr = 128 + .5 * r - .418688 * g - .081312 * b
    return y, cb, cr


def opencv(hand):
    start_time = time.time()
    img_ycc = shape(hand)
    return img_ycc, time.time() - start_time


def shape(hand):
    img_ycc = cv2.cvtColor(hand, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = _ycc(195, 154, 108)
    avg_skin_color = [y, cb, cr]
    error = 35
    for i in range(0, hand.shape[0]):  # looping through the rows( height)
        for j in range(0, hand.shape[1]):  # looping through the columns(width)
            # calculate the Euclidean distance between a pixel and the skin color defined above
            """
            if (math.sqrt(
                    (int(img_ycc[i, j, 0]) - avg_skin_color[0]) ** 2 +
                    (int(img_ycc[i, j, 1]) - avg_skin_color[1]) ** 2 +
                    (int(img_ycc[i, j, 2]) - avg_skin_color[2]) ** 2) <= error):
                img_ycc[i, j] = [255, 255, 255]  # make the pixel white
            else:
                img_ycc[i, j] = [0, 0, 0]  # make the pixel black
            """
            if img_ycc[i, j, 0] > 80 and 80 <= img_ycc[i, j, 1] <= 120 and 133 <= img_ycc[i, j, 2] <= 173:
                img_ycc[i, j] = [255, 255, 255]
            else:
                img_ycc[i, j] = [0, 0, 0]
    kernel = np.ones((7, 7), np.uint8)
    img_ycc = cv2.dilate(img_ycc, kernel, iterations=1)
    img_ycc = cv2.erode(img_ycc, kernel, iterations=1)
    ret = cv2.cvtColor(img_ycc, cv2.COLOR_BGR2GRAY)
    return ret


def skin_detector(img):
    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    global_result = cv2.bitwise_not(global_mask)
    return global_result


def paln_point(hand):
    bw = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    x_iner, y_iner = np.unravel_index(dist.argmax(), dist.shape)
    hand[x_iner, y_iner] = [125, 125, 125]
    rad, _, _ = inner_circle_max_radius(hand, y_iner, x_iner, [0, 0, 0])
    points = wrist_points_and_palm_mask(rad, x_iner, y_iner, hand)

    max_distance = 0
    max_x1, max_y1 = 0, 0
    max_x2, max_y2 = 0, 0
    for i in range(len(points) - 1):
        dist = np.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        if dist > max_distance:
            max_distance = dist
            max_x1, max_y1 = points[i][0], points[i][1]
            max_x2, max_y2 = points[i + 1][0], points[i + 1][1]
    cv2.line(hand, (max_x1, max_y1), (max_x2, max_y2), (255, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.circle(hand, (y_iner, x_iner), rad, (255, 0, 0), thickness=1, lineType=8, shift=0)
    # palm_point = (y_iner, x_iner)
    # wrist_point1 = (max_y1, max_x1)
    # wrist_point2 = (max_y2, max_x2)
    # return clear_sub_wrist(hand, wrist_point1, wrist_point2, palm_point)
    return hand


def inner_circle_max_radius(hand, x, y, comp):
    tmp = np.zeros(hand.shape, np.uint8)
    rad = 0
    while True:
        rad = rad + 1
        cv2.circle(tmp, (x, y), rad, (0, 255, 0), -1)
        combined = tmp[:, :, 0] + tmp[:, :, 1] + tmp[:, :, 2]
        rows, cols = np.where(combined > 0)
        for i in range(len(rows)):
            if (hand[rows[i], cols[i]] == comp).all():
                return rad - 1, rows[i], cols[i]


def get_near_border(hand, x, y):
    tmp = np.zeros(hand.shape, np.uint8)
    rad = 0
    while True:
        rad = rad + 1
        cv2.circle(tmp, (x, y), rad, (0, 255, 0), -1)
        combined = tmp[:, :, 0] + tmp[:, :, 1] + tmp[:, :, 2]
        rows, cols = np.where(combined > 0)
        for i in range(len(rows)):
            has_black = False
            has_white = False
            if rows[i] - 1 > -1:
                if cols[i] - 1 > 0:
                    if (hand[rows[i] - 1, cols[i] - 1] == [0, 0, 0]).all():
                        has_black = True
                    if (hand[rows[i] - 1, cols[i] - 1] == [255, 255, 255]).all():
                        has_white = True

                if (hand[rows[i] - 1, cols[i]] == [0, 0, 0]).all():
                    has_black = True
                if (hand[rows[i] - 1, cols[i]] == [255, 255, 255]).all():
                    has_white = True
                if cols[i] + 1 < hand.shape[1]:
                    if (hand[rows[i] - 1, cols[i] + 1] == [0, 0, 0]).all():
                        has_black = True
                    if (hand[rows[i] - 1, cols[i] + 1] == [255, 255, 255]).all():
                        has_white = True

            if cols[i] - 1 > 0:
                if (hand[rows[i], cols[i] - 1] == [0, 0, 0]).all():
                    has_black = True
                if (hand[rows[i], cols[i] - 1] == [255, 255, 255]).all():
                    has_white = True

            if cols[i] + 1 < hand.shape[1]:
                if (hand[rows[i], cols[i] + 1] == [0, 0, 0]).all():
                    has_black = True
                if (hand[rows[i], cols[i] + 1] == [255, 255, 255]).all():
                    has_white = True

            if rows[i] + 1 < hand.shape[0]:
                if cols[i] - 1 > 0:
                    if (hand[rows[i] + 1, cols[i] - 1] == [0, 0, 0]).all():
                        has_black = True
                    if (hand[rows[i] + 1, cols[i] - 1] == [255, 255, 255]).all():
                        has_white = True

                if (hand[rows[i] + 1, cols[i]] == [0, 0, 0]).all():
                    has_black = True
                if (hand[rows[i] + 1, cols[i]] == [255, 255, 255]).all():
                    has_white = True
                if cols[i] + 1 < hand.shape[1]:
                    if (hand[rows[i] + 1, cols[i] + 1] == [0, 0, 0]).all():
                        has_black = True
                    if (hand[rows[i] + 1, cols[i] + 1] == [255, 255, 255]).all():
                        has_white = True

            if has_white and has_black:
                return rad, rows[i], cols[i]


def wrist_points_and_palm_mask(rad, x, y, hand):
    large_rad = int(1.2 * rad)
    ret = list()
    t = 5
    X = [int(large_rad * math.cos(teta * math.pi / 180) + y) for teta in range(0, 361, t)]
    Y = [int(large_rad * math.sin(teta * math.pi / 180) + x) for teta in range(0, 361, t)]
    for i in range(len(X)):
        # cv2.circle(hand, (X[i], Y[i]), 2, (0, 255, 0))
        rad, x1, y1 = get_near_border(hand, X[i], Y[i])
        # cv2.circle(hand, (y1, x1), 2, (0, 255, 0))
        cv2.line(hand, (X[i], Y[i]), (y1, x1), (0, 255, 0), thickness=1, lineType=8, shift=0)
        ret.append((y1, x1))
    cv2.circle(hand, (y, x), large_rad, (0, 0, 255), thickness=1, lineType=8, shift=0)
    return ret


def clear_sub_wrist(hand, wrist_point1, wrist_point2, palm_point):
    x1, y1 = wrist_point1
    x2, y2 = wrist_point2
    x3, y3 = palm_point
    D = (x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)
    for i in range(0, hand.shape[0]):  # looping through the rows( height)
        for j in range(0, hand.shape[1]):  # looping through the columns(width)
            D_ij = (i - x1) * (y2 - y1) - (j - y1) * (x2 - x1)
            if D_ij * D < 0:
                hand[i, j] = [0, 0, 0]
    return hand
