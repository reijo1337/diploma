for row in range(self.width-len(sobelx)):
    for col in range(self.height-len(sobelx)):
        gx = 0
        gy = 0
        for i in range(len(sobelx)):
            for j in range(len(sobely)):
                val = mat[row+i, col+j] * lin_scale
                gx += sobelx[i][j] * val
                gy += sobely[i][j] * val
        pixels[row+1, col+1] = int(math.sqrt(gx*gx + gy*gy))