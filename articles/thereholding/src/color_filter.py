import math

for i in range(0, hand.shape[0]):
    for j in range(0, hand.shape[1]):  
        if (math.sqrt(
                (int(imgYCC[i, j, 0]) - avg_skin_color[0]) ** 2 +
                (int(imgYCC[i, j, 1]) - avg_skin_color[1]) ** 2 +
                (int(imgYCC[i, j, 2]) - avg_skin_color[2]) ** 2) <= error):
            imgYCC[i, j] = [255, 255, 255] 
        else:
            imgYCC[i, j] = [0, 0, 0]  