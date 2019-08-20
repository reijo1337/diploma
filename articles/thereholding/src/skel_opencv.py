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
for i in range(0, skel.shape[0]):  # looping through the rows( height)
    for j in range(0, skel.shape[1]):  # looping through the columns(width)
        if (skel[i, j] == [255, 255, 255]).all():
            skel[i, j] = [255, 0, 0]
            orig_img[i, j] = [255, 0, 0]