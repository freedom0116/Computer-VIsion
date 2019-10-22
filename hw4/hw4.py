import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img)
arrsize = len(imgarr), len(imgarr[0])

# Binarize 
def Binarize(arr):
    for i in range(arrsize[0]):
        for j in range(arrsize[1]):
            if arr[i][j] < 128: 
                arr[i][j] = 0
            else:
                arr[i][j] = 255

# Dilatio use octogonal 3-5-5-5-3 kernel
def Dilation(arr, x, y):
    for i in range(-2, 3):
        if(i == 0 or i == 4):
            if((y + i) >= 0 and (y + i) < arrsize[0]):
                for j in range(-1, 2):
                    if((x + j) >= 0 and (x + j) < arrsize[1]):
                        arr[y + i][x + j] = 255
        else:
            if((y + i) >= 0 and (y + i) < arrsize[0]):
                for j in range(-2, 3):
                    if((x + j) >= 0 and (x + j) < arrsize[1]):
                        arr[y + i][x + j] = 255

# Erosion use octogonal 3-5-5-5-3 kernel
def Erosion(arr, x, y):
    for i in range(-2, 3):
        for j in range(-2, 3): 
            if((y + i) >= 0 and (y + i) < arrsize[0] and (x + j) >= 0 and (x + j) < arrsize[1]):        
                if(i == 0 or i == 4):
                    if(j != -2 or j != 2):
                        if(arr[y + i][x + j] != 255):
                            return False
                else:
                    if(arr[y + i][x + j] != 255):
                        return False
    return True

# Hit-and-miss Transform
def L_Derosion(arr, x, y):
    if((x - 1) >= 0 and (x - 1) < arrsize[1] and (y + 1) >= 0 and (y + 1) < arrsize[0]):
        if(arr[y][x - 1] == 255 and arr[y + 1][x] == 255):
            return True
    return False

def R_Terosion(arr, x, y):
    if((x + 1) >= 0 and (x + 1) < arrsize[1] and (y - 1) >= 0 and (y - 1) < arrsize[0]):
        if(arr[y - 1][x + 1] == 0 and arr[y][x + 1] == 0 and arr[y - 1][x] == 0):
            return True
    return False


    


Binarize(imgarr)

# Dilation
dilarr = np.array(img)
Binarize(dilarr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(imgarr[y][x] == 255):
            Dilation(dilarr, x, y)

# Erosion
eroarr = np.array(img)
Binarize(eroarr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(imgarr[y][x] == 255):
            if(Erosion(imgarr, x, y) == False):
                eroarr[y][x] = 0

# Opening
openarr = np.array(img)
Binarize(openarr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(eroarr[y][x] == 255):
            Dilation(openarr, x, y)

# Closing
closearr = np.array(img)
Binarize(closearr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(dilarr[x][y] == 255):
            if(Erosion(dilarr, x, y) == False):
                closearr[y][x] = 0

# Hit-and-miss Transform
LDarr = np.array(img)
Binarize(LDarr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(imgarr[y][x] == 255):
            if(L_Derosion(imgarr, x, y) == False):
                LDarr[y][x] = 0

RTarr = np.array(img)
Binarize(RTarr)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(R_Terosion(imgarr, x, y) == False):
            RTarr[y][x] = 0
        else:
            RTarr[y][x] = 255

HMarr = np.zeros((arrsize[0], arrsize[1]))
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        if(RTarr[y][x] == LDarr[y][x]):
            HMarr[y][x] = 255
        else:
            HMarr[y][x] = 0

plt.subplot(2,3,1)
plt.imshow(dilarr, cmap="gray"), plt.title('Dilation')
plt.subplot(2,3,2)
plt.imshow(eroarr, cmap="gray"), plt.title('Erosion')
plt.subplot(2,3,3)
plt.imshow(openarr, cmap="gray"), plt.title('Opening')
plt.subplot(2,3,4)
plt.imshow(closearr, cmap="gray"), plt.title('Closing')
plt.subplot(2,3,5)
plt.imshow(HMarr, cmap="gray"), plt.title('Hit-and-miss Transform')
plt.show()




                        


