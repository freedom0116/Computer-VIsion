import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img)
arrsize = len(imgarr[0]), len(imgarr)

# Dilatio use octogonal 3-5-5-5-3 kernel
def Dilation(arr, refarr, x, y):
    for i in range(-2, 3):
        if(i == 0 or i == 4):
             for j in range(-1, 2):
                if((y + i) >= 0 and (y + i) < arrsize[0] and (x + j) >= 0 and (x + j) < arrsize[1]): 
                    if(arr[y][x] < refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]
        else:
            for j in range(-2, 3):
                if((y + i) >= 0 and (y + i) < arrsize[0] and (x + j) >= 0 and (x + j) < arrsize[1]): 
                    if(arr[y][x] < refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]
    

# Erosion use octogonal 3-5-5-5-3 kernel
def Erosion(arr, refarr, x, y):
    for i in range(-2, 3):
        if(i == 0 or i == 4):
             for j in range(-1, 2):
                if((y + i) >= 0 and (y + i) < arrsize[0] and (x + j) >= 0 and (x + j) < arrsize[1]): 
                    if(arr[y][x] > refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]
        else:
            for j in range(-2, 3):
                if((y + i) >= 0 and (y + i) < arrsize[0] and (x + j) >= 0 and (x + j) < arrsize[1]): 
                    if(arr[y][x] > refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]




# Dilation
dilarr = np.array(img)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        Dilation(dilarr, imgarr, x, y)

# Erosion
eroarr = np.array(img)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        Erosion(eroarr,imgarr, x, y)

# Opening
openarr = np.array(img)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        Dilation(openarr, eroarr, x, y)

# Closing
closearr = np.array(img)
for y in range(arrsize[0]):
    for x in range(arrsize[1]):
        Erosion(closearr, dilarr, x, y)



plt.subplot(2,2,1)
plt.axis('off')
plt.imshow(dilarr, cmap="gray"), plt.title('Dilation')
plt.subplot(2,2,2)
plt.axis('off')
plt.imshow(eroarr, cmap="gray"), plt.title('Erosion')
plt.subplot(2,2,3)
plt.axis('off')
plt.imshow(openarr, cmap="gray"), plt.title('Opening')
plt.subplot(2,2,4)
plt.axis('off')
plt.imshow(closearr, cmap="gray"), plt.title('Closing')
plt.show()



                        


