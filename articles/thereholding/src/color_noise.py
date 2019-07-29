kernel = np.ones((7, 7), np.uint8)
imgYCC = cv2.dilate(imgYCC, kernel, iterations=1)
imgYCC = cv2.erode(imgYCC, kernel, iterations=1)