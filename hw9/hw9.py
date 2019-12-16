import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img, dtype = np.int)
arrsize = len(imgarr[0]), len(imgarr)

def Roberts(ref, threshold):
    output = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            mag = math.sqrt((int(ref[y + 1][x + 1]) - int(ref[y][x]))**2 + (int(ref[y + 1][x]) - int(ref[y][x + 1]))**2)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Prewitt(ref, threshold):
    output = np.zeros([arrsize[0] - 2, arrsize[1] - 2])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            yc = y + 1
            xc = x + 1
            p1 = (ref[yc + 1][xc - 1] + ref[yc + 1][xc] + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + ref[yc - 1][xc] + ref[yc - 1][xc + 1])
            p2 = (ref[yc - 1][xc + 1] + ref[yc][xc + 1] + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + ref[yc][xc - 1] + ref[yc + 1][xc - 1])
            mag = math.sqrt(p1**2 + p2**2)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Sobel(ref, threshold):
    output = np.zeros([arrsize[0] - 2, arrsize[1] - 2])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            yc = y + 1
            xc = x + 1
            p1 = (ref[yc + 1][xc - 1] + 2 * ref[yc + 1][xc] + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + 2 * ref[yc - 1][xc] + ref[yc - 1][xc + 1])
            p2 = (ref[yc - 1][xc + 1] + 2 * ref[yc][xc + 1] + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + 2 * ref[yc][xc - 1] + ref[yc + 1][xc - 1])
            mag = math.sqrt(p1**2 + p2**2)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output
    
def Frei_and_Chen(ref, threshold):
    output = np.zeros([arrsize[0] - 2, arrsize[1] - 2])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            yc = y + 1
            xc = x + 1
            p1 = (ref[yc + 1][xc - 1] + math.sqrt(ref[yc + 1][xc]) + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + math.sqrt(ref[yc - 1][xc]) + ref[yc - 1][xc + 1])
            p2 = (ref[yc - 1][xc + 1] + math.sqrt(ref[yc][xc + 1]) + ref[yc + 1][xc + 1]) - (ref[yc - 1][xc - 1] + math.sqrt(ref[yc][xc - 1]) + ref[yc + 1][xc - 1])
            mag = math.sqrt(p1**2 + p2**2)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Kirsch_Compass(ref, threshold):
    output = np.zeros([arrsize[0] - 2, arrsize[1] - 2])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            mag = Kirsch_Gradient(ref, x + 1, y + 1)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Kirsch_Gradient(ref, x, y):
    I = np.zeros(8)
    point = [ref[y - 1][x - 1], ref[y - 1][x], ref[y - 1][x + 1], ref[y][x + 1], ref[y + 1][x + 1], ref[y + 1][x], ref[y + 1][x - 1], ref[y][x - 1]]
    k0 = [-3, -3, 5, 5, 5, -3, -3, -3]
    k1 = [-3, 5, 5, 5, -3, -3, -3, -3]
    k2 = [5, 5, 5, -3, -3, -3, -3, -3]
    k3 = [5, 5, -3, -3, -3, -3, -3, 5]
    k4 = [5, -3, -3, -3, -3, -3, 5, 5]
    k5 = [-3, -3, -3, -3, -3, 5, 5, 5]
    k6 = [-3, -3, -3, -3, 5, 5, 5, -3]
    k7 = [-3, -3, -3, 5, 5, 5, -3, -3]
    I[0] = OneDArrayConv(k0, point)
    I[1] = OneDArrayConv(k1, point)
    I[2] = OneDArrayConv(k2, point)
    I[3] = OneDArrayConv(k3, point)
    I[4] = OneDArrayConv(k4, point)
    I[5] = OneDArrayConv(k5, point)
    I[6] = OneDArrayConv(k6, point)
    I[7] = OneDArrayConv(k7, point)
    return I.max()

def Robinson_Compass(ref, threshold):
    output = np.zeros([arrsize[0] - 2, arrsize[1] - 2])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            mag = Robinson_Gradient(ref, x + 1, y + 1)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Robinson_Gradient(ref, x, y):
    I = np.zeros(8)
    point = [ref[y - 1][x - 1], ref[y - 1][x], ref[y - 1][x + 1], ref[y][x + 1], ref[y + 1][x + 1], ref[y + 1][x], ref[y + 1][x - 1], ref[y][x - 1]]
    k0 = [-1, 0, 1, 2, 1, 0, -1, -2]
    k1 = [0, 1, 2, 1, 0, -1, -2, -1]
    k2 = [1, 2, 1, 0, -1, -2, -1, 0]
    k3 = [2, 1, 0, -1, -2, -1, 0, 1]
    k4 = [1, 0, -1, -2, -1, 0, 1, 2]
    k5 = [0 ,-1, -2, -1, 0, 1, 2, 1]
    k6 = [-1, -2, -1, 0, 1, 2, 1, 0]
    k7 = [-2, -1, 0, 1, 2, 1, 0, -1]
    I[0] = OneDArrayConv(k0, point)
    I[1] = OneDArrayConv(k1, point)
    I[2] = OneDArrayConv(k2, point)
    I[3] = OneDArrayConv(k3, point)
    I[4] = OneDArrayConv(k4, point)
    I[5] = OneDArrayConv(k5, point)
    I[6] = OneDArrayConv(k6, point)
    I[7] = OneDArrayConv(k7, point)
    return I.max()

def Nevatia_Babu(ref, threshold):
    output = np.zeros([arrsize[0] - 4, arrsize[1] - 4])
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            mag = Nevatia_Babu_Gradient(ref, x + 2, y + 2)
            if mag > threshold:
                output[y][x] = 0
            else:
                output[y][x] = 255
    return output

def Nevatia_Babu_Gradient(ref, x, y):
    I = np.zeros(6)
    point = [[ref[y - 2][x - 2], ref[y - 2][x - 1], ref[y - 2][x], ref[y - 2][x + 1], ref[y - 2][x + 2]],
                [ref[y - 1][x - 2], ref[y - 1][x - 1], ref[y - 1][x], ref[y - 1][x + 1], ref[y - 1][x + 2]],
                [ref[y][x - 2], ref[y][x - 1], ref[y][x], ref[y][x + 1], ref[y][x + 2]],
                [ref[y + 1][x - 2], ref[y + 1][x - 1], ref[y + 1][x], ref[y + 1][x + 1], ref[y + 1][x + 2]],
                [ref[y + 2][x - 2], ref[y + 2][x - 1], ref[y + 2][x], ref[y + 2][x + 1], ref[y + 2][x + 2]]]

    k0 = [[100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100],
        [0, 0, 0, 0, 0],
        [-100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100]]

    k1 = [[100, 100, 100, 100, 100],
        [100, 100, 100, 78, -32],
        [100, 92, 0, -92, -100],
        [32, -78, -100, -100, -100],
        [-100, -100, -100, -100, -100]]

    k2 = [[100, 100, 100, 32, -100],
        [100, 100, 92, -78, -100],
        [100, 100, 0, -100, -100],
        [100, 78, -92, -100, -100],
        [100, -32, -100, -100, -100]]

    k3 = [[-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, 0, 100, 100]]

    k4 = [[-100, 32, 100, 100, 100],
        [-100, -78, 92, 100, 100],
        [-100, -100, 0, 100, 100],
        [-100, -100, -92, 78, 100],
        [-100, -100, -100, -32, 100]]

    k5 = [[100, 100, 100, 100, 100],
        [-32, 78, 100, 100, 100],
        [-100, -92, 0, 92, 100],
        [-100, -100, -100, -78, 32],
        [-100, -100, -100, -100, -100]]

    I[0] = MultiDArrayConv(k0, point)
    I[1] = MultiDArrayConv(k1, point)
    I[2] = MultiDArrayConv(k2, point)
    I[3] = MultiDArrayConv(k3, point)
    I[4] = MultiDArrayConv(k4, point)
    I[5] = MultiDArrayConv(k5, point)
    return I.max()


def OneDArrayConv(k, p):
    I = 0
    for i in range(len(k)):
        I += k[i] * p[i]
    return I

def MultiDArrayConv(k, p):
    I = 0
    for i in range(len(k)):
        for j in range(len(k[0])):
            I += k[i][j] * p[i][j]
    return I

threshold = [17, 60, 80, 30, 250, 70, 15000]

robarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
robarr = Roberts(imgarr, threshold[0])
title_rob = 'Roberts Operator thresholds = ' + str(threshold[0])
plt.axis('off')
plt.imshow(robarr, cmap="gray"), plt.title(title_rob)

prewarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
prewarr = Prewitt(imgarr, threshold[1])
title_prew = 'Prewitt Operator thresholds = ' + str(threshold[1])
plt.axis('off')
plt.imshow(prewarr, cmap="gray"), plt.title(title_prew)

sobelarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
sobelarr = Sobel(imgarr, threshold[2])
title_sobel = 'Sobel Operator thresholds = ' + str(threshold[2])
plt.axis('off')
plt.imshow(sobelarr, cmap="gray"), plt.title(title_sobel)

chenarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
chenarr = Frei_and_Chen(imgarr, threshold[3])
title_chen = 'Frei_and_Chen Gradient Operator thresholds = ' + str(threshold[3])
plt.axis('off')
plt.imshow(chenarr, cmap="gray"), plt.title(title_chen)

krisarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
krisarr = Kirsch_Compass(imgarr, threshold[4])
title_kris = 'Krisch Compass Operator thresholds = ' + str(threshold[4])
plt.axis('off')
plt.imshow(krisarr, cmap="gray"), plt.title(title_kris)

robinarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
robinarr = Robinson_Compass(imgarr, threshold[5])
title_robin = 'Robinson Compass Operator thresholds = ' + str(threshold[5])
plt.axis('off')
plt.imshow(robinarr, cmap="gray"), plt.title(title_robin)

NBarr = np.zeros([arrsize[0] - 1, arrsize[1] - 1])
NBarr = Nevatia_Babu(imgarr, threshold[6])
title_NB = 'Nevatia_Babu 5x5 Operator thresholds = ' + str(threshold[6])
plt.axis('off')
plt.imshow(NBarr, cmap="gray"), plt.title(title_NB)

plt.show()




