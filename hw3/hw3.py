import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image


img = cv2.imread("lena.bmp")
imgarr = np.array(img)
arrsize = len(imgarr), len(imgarr[0]), len(imgarr[0][0])

#Origin
hist1 = np.zeros(256)
for i in range(arrsize[0]):
    for j in range(arrsize[1]):            
        hist1[imgarr[i][j][0]] += 1
plt.figure()
plt.subplot(2,3,1)
plt.imshow(img), plt.title('original')
plt.subplot(2,3,4)
x = np.arange(256)
plt.bar(x, hist1)

#Divided3
hist2 = np.zeros(256)
divarr = np.zeros(arrsize,dtype = np.uint8)
#intarr = np.zeros(arrsize)
for i in range(arrsize[0]):
    for j in range(arrsize[1]):
        divarr[i][j] = imgarr[i][j] / 3
        hist2[divarr[i][j][0]] += 1

plt.subplot(2,3,2)
plt.imshow(divarr), plt.title('divided3')
plt.subplot(2,3,5)
x2 = np.arange(256)
plt.bar(x2, hist2)

#Equalization
hist3 = np.zeros(256)
histref = np.zeros(256,dtype = np.uint8)
CDF = 0
for i in range(256):
    CDF += 255*hist1[i]/(512*512)    
    hist3[int(CDF)] = hist1[i]    
    histref[i] = int(CDF)
equarr = np.zeros(arrsize,dtype = np.uint8)
for i in range(arrsize[0]):
    for j in range(arrsize[1]):
        num = 0
        while imgarr[i][j][0] != num:
            num += 1
        equarr[i][j] += histref[num] 

plt.subplot(2,3,3)
plt.imshow(equarr), plt.title('equalization')
plt.subplot(2,3,6)
x3 = np.arange(256)
plt.bar(x3, hist3)

plt.show()

