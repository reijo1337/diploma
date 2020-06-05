from __future__ import division

import time

import cv2
import numpy as np

protoFile = "skelet_dnn/hand/pose_deploy.prototxt"
weightsFile = "skelet_dnn/hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]
#net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


def skeleton(frame):
    start_time = time.time()
    frame_copy = np.copy(frame)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = frame_width / frame_height
    threshold = 0.1

    in_height = 368
    in_width = int(((aspect_ratio * in_height) * 8) // 8)
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inp_blob)

    output = net.forward()
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        prob_map = output[0, i, :, :]
        prob_map = cv2.resize(prob_map, (frame_width, frame_height))

        # Find global maxima of the prob_map.
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        if prob > threshold:
            cv2.circle(frame_copy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame_copy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        part_a = pair[0]
        partB = pair[1]

        if points[part_a] and points[partB]:
            cv2.line(frame, points[part_a], points[partB], (0, 255, 255), 2)
            # cv2.circle(frame, points[part_a], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return frame, time.time() - start_time

#
# t = time.time()
# # input image dimensions for the network
# print("time taken by network : {:.3f}".format(time.time() - t))
#
# # Empty list to store the detected keypoints
#
#
# cv2.imshow('Output-Keypoints', frameCopy)
# cv2.imshow('Output-Skeleton', frame)
#
#
# cv2.imwrite('Output-Keypoints.jpg', frameCopy)
# cv2.imwrite('Output-Skeleton.jpg', frame)
#
# print("Total time taken : {:.3f}".format(time.time() - t))
#
# cv2.waitKey(0)
