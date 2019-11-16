import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img)
arrsize = len(imgarr[0]), len(imgarr)

# Downsampling
def DownSampling(arr):
    xnum = int(arrsize[0] / 8)
    ynum = int(arrsize[1] / 8)
    for yt in range(ynum):
        y = yt * 8
        for xt in range(xnum):
            x = xt * 8
            if(imgarr[y][x] >= 128):
                arr[yt][xt] = 255
            else:
                arr[yt][xt] = 0

# Yokoi
def Yokoi(refarr, output):
    size = len(refarr[0]), len(refarr)
    for y in range(size[1]):
        for x in range(size[0]):
            if(refarr[y][x] == 255):
                q, r = (0, 0)
                if((x + 1) < 64 and refarr[y][x + 1] == 255):
                    if((y - 1) < 0):
                        q += 1
                    else:
                        if(refarr[y - 1][x + 1] == 255 and refarr[y - 1][x] == 255):
                            r += 1
                        else:
                            q += 1
                if((y - 1) >= 0 and refarr[y - 1][x] == 255):
                    if((x - 1) < 0):
                        q += 1
                    else:
                        if(refarr[y - 1][x - 1] == 255 and refarr[y][x - 1] == 255):
                            r += 1
                        else:
                            q += 1
                if((x - 1) >= 0 and refarr[y][x - 1] == 255):
                    if((y + 1) > 63):
                        q += 1
                    else:
                        if(refarr[y + 1][x - 1] == 255 and refarr[y + 1][x] == 255):
                            r += 1
                        else:
                            q += 1
                if((y + 1) < 64 and refarr[y + 1][x] == 255):
                    if((x + 1) > 63):
                        q += 1
                    else:
                        if(refarr[y + 1][x + 1] == 255 and refarr[y][x + 1] == 255):
                            r += 1
                        else:
                            q += 1
                if q == 1:
                    output[y][x] = '1'
                if q == 2:
                    output[y][x] = '2'
                if q == 3:
                    output[y][x] = '3'
                if q == 4:
                    output[y][x] = '4'
                if r == 4:
                    output[y][x] = '5'
            else:
                output[y][x] = ' '

def CharYokoi(refarr, output):
    size = len(refarr[0]), len(refarr)
    for y in range(size[1]):
        for x in range(size[0]):
            if(refarr[y][x] != ' '):
                q, r = (0, 0)
                if((x + 1) < 64 and refarr[y][x + 1] != ' '):
                    if((y - 1) < 0):
                        q += 1
                    else:
                        if(refarr[y - 1][x + 1] != ' ' and refarr[y - 1][x] != ' '):
                            r += 1
                        else:
                            q += 1
                if((y - 1) >= 0 and refarr[y - 1][x] != ' '):
                    if((x - 1) < 0):
                        q += 1
                    else:
                        if(refarr[y - 1][x - 1] != ' ' and refarr[y][x - 1] != ' '):
                            r += 1
                        else:
                            q += 1
                if((x - 1) >= 0 and refarr[y][x - 1] != ' '):
                    if((y + 1) > 63):
                        q += 1
                    else:
                        if(refarr[y + 1][x - 1] != ' ' and refarr[y + 1][x] != ' '):
                            r += 1
                        else:
                            q += 1
                if((y + 1) < 64 and refarr[y + 1][x] != ' '):
                    if((x + 1) > 63):
                        q += 1
                    else:
                        if(refarr[y + 1][x + 1] != ' ' and refarr[y][x + 1] != ' '):
                            r += 1
                        else:
                            q += 1
                if q == 1:
                    output[y][x] = '1'
                if q == 2:
                    output[y][x] = '2'
                if q == 3:
                    output[y][x] = '3'
                if q == 4:
                    output[y][x] = '4'
                if r == 4:
                    output[y][x] = '5'
            else:
                output[y][x] = ' '

# Pair relationship Operator
def Pair(refarr, output):
    size = len(refarr[0]), len(refarr)
    for y in range(size[1]):
        for x in range(size[0]):
            count = 0
            if(refarr[y][x] == '1'):
                if((x + 1) < 64 and refarr[y][x + 1] == '1'):
                    count += 1
                if((y - 1) >= 0 and refarr[y - 1][x] == '1'):
                    count += 1
                if((x - 1) >= 0 and refarr[y][x - 1] == '1'):
                    count += 1
                if((y + 1) < 64 and refarr[y + 1][x] == '1'):
                    count += 1
                if count < 1:
                    output[y][x] = 'q'
                else:
                    output[y][x] = 'p'
            else:
                if(refarr[y][x] != ' '):
                    output[y][x] = 'q'                 
        #     print(pairarr[y][x], end = '')
        # print()

# Connected Shrink Operator
def Shrink(refarr, output):
    change = False
    size = len(refarr[0]), len(refarr)
    for y in range(size[1]):
        for x in range(size[0]):
            count = 0      
            if output[y][x] == 'p':      
                count = 0          
                if((x + 1) < 64 and output[y][x + 1] != ' '):
                    if((y - 1) < 0):
                        count += 1
                    else:
                        if(output[y - 1][x + 1] == ' ' or output[y - 1][x] == ' '):
                            count += 1
                if((y - 1) >= 0 and output[y - 1][x] != ' '):
                    if((x - 1) < 0):
                        count += 1
                    else:
                        if(output[y - 1][x - 1] == ' ' or output[y][x - 1] == ' '):
                            count += 1
                if((x - 1) >= 0 and output[y][x - 1] != ' '):
                    if((y + 1) > 63):
                        count += 1
                    else:
                        if(output[y + 1][x - 1] == ' ' or output[y + 1][x] == ' '):
                            count += 1
                if((y + 1) < 64 and output[y + 1][x] != ' '):
                    if((x + 1) > 63):
                        count += 1
                    else:
                        if(output[y + 1][x + 1] == ' ' or output[y][x + 1] == ' '):
                            count += 1
                if count == 1:
                    change = True
                    output[y][x] = ' '
    return change

# Thinning
def Thinning(refarr, output):
    size = len(refarr[0]), len(refarr)
    pairarr = [[' ' for i in range(64)] for j in range(64)]

    while True:
        Pair(refarr, pairarr)    
        if Shrink(refarr, pairarr) == True:
            CharYokoi(pairarr, refarr)
        else:
            break

    for y in range(size[1]):
        for x in range(size[0]):
            if pairarr[y][x] != ' ':
                output[y][x] = 255
            else:
                output[y][x] = 0     
        #     print(pairarr[y][x], end = '')
        # print()


# Downsampling
downarr = np.zeros([64, 64])
DownSampling(downarr)

# Yokoi
yokoiarr = [[' ' for i in range(64)] for j in range(64)]
Yokoi(downarr, yokoiarr)

# Thinning
thinarr = np.zeros([64, 64])
Thinning(yokoiarr, thinarr)
plt.imshow(thinarr, cmap = "gray"), plt.title("Thinning")
plt.show()