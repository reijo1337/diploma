robertsx = [[1,0],[0,-1]]
robertsy = [[0,1],[-1,0]]
lin_scale = .7
for row in range(self.width-len(robertsx)):
    for col in range(self.height-len(robertsy)):
        Gx = 0
        Gy = 0
        for i in range(len(robertsx)):
            for j in range(len(robertsy)):
                val = mat[row+i, col+j] * lin_scale
                Gx += robertsx[i][j] * val
                Gy += robertsy[i][j] * val
        pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
