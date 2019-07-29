def hysteresis(self, img):
    M, N = img.shape
    weak = self.weak_pixel
    strong = self.strong_pixel
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) 
                        or (img[i + 1, j] == strong) 
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) 
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) 
                        or (img[i - 1, j] == strong) 
                        or (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img