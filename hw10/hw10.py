import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img, dtype = np.int)
arrsize = len(imgarr[0]), len(imgarr)

def Laplacian(ref, mask, threshold):
    output = np.zeros([len(ref)-2, len(ref[0])-2])
    zeroCross = np.array(output)
    size = len(output[0]), len(output)
    for y in range(size[1]):
        for x in range(size[0]):
            I = Laplacian_Magnitude(ref, x+1, y+1, mask)
            if I > threshold:
                zeroCross[y][x] = 1
            elif I < -threshold:
                zeroCross[y][x] = -1
            else:
                zeroCross[y][x] = 0
    for y in range(size[1]):
        for x in range(size[0]):
            output[y][x] = Zero_Cross_Edge_Detector(zeroCross, x, y)
    return output

def Laplacian_Magnitude(arr, x, y, mask):
    ref = np.array(mask, dtype = np.int)
    for i in range(-1, 2):
        for j in range(-1, 2):
            ref[i+1][j+1] = arr[y+i][x+j]
    I = MultiDArrayConv(ref, mask)
    return I

def LOG(ref, mask, threshold):
    output = np.zeros([len(ref)-10, len(ref[0])-10])
    zeroCross = np.zeros([len(ref)-10, len(ref[0])-10]) 
    for y in range(len(ref)-10):
        for x in range(len(ref[0])-10):
            I = LOG_Magnitude(ref, x+5, y+5, mask)
            if I > threshold:
                zeroCross[y][x] = 1
            elif I < -threshold:
                zeroCross[y][x] = -1
            else:
                zeroCross[y][x] = 0
    for y in range(len(output)):
        for x in range(len(output[0])):
            output[y][x] = Zero_Cross_Edge_Detector(zeroCross, x, y)
    return output

# # kernel size 11x11
# def DOG(ref, inhibitory_sigma, excitatory_sigma, threshold):



def LOG_Magnitude(arr, x, y, mask):
    ref = np.zeros([11, 11])
    for i in range(-5, 6):
        for j in range(-5, 6):
            ref[i+5][j+5] = arr[y+i][x+j]
    I = MultiDArrayConv(ref, mask)
    return I

def Zero_Cross_Edge_Detector(arr, x, y):
    I = 255
    if arr[y][x] == 1:
        for i in range(-1, 2):
            if(y+i >= 0 and y+i < len(arr)):
                for j in range(-1, 2):
                    if(x+j >= 0 and x+j < len(arr[0])):
                        if(arr[y+i][x+j] == -1):
                            I = 0
    return I

def MultiDArrayConv(k, p):
    I = 0
    for i in range(len(k)):
        for j in range(len(k[0])):
            I += k[i][j] * p[i][j]
    return I

threshold = [15, 15, 20, 3000, 1]

# lap1arr = np.zeros([arrsize[1] - 2, arrsize[0] - 2])
# lap1Mask = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
# lap1arr = Laplacian(imgarr, lap1Mask, threshold[0])
# title_lap1 = 'Laplacian Mask1 thresholds = ' + str(threshold[0])
# plt.axis('off')
# plt.imshow(lap1arr, cmap="gray"), plt.title(title_lap1)

# lap2arr = np.zeros([arrsize[1] - 2, arrsize[0] - 2])
# lap2Mask = [[1/3, 1/3, 1/3], [1/3, -8/3, 1/3], [1/3, 1/3, 1/3]]
# lap2arr = Laplacian(imgarr, lap2Mask, threshold[1])
# title_lap2 = 'Laplacian Mask2 thresholds = ' + str(threshold[1])
# plt.axis('off')
# plt.imshow(lap2arr, cmap="gray"), plt.title(title_lap2)

# minlaparr = np.zeros([arrsize[1] - 2, arrsize[0] - 2])
# minlapMask = [[2/3, -1/3, 2/3], [-1/3, -4/3, -1/3], [2/3, -1/3, 2/3]]
# minlaparr = Laplacian(imgarr, minlapMask, threshold[2])
# title_minlap = 'minimum-variance Laplacian thresholds = ' + str(threshold[2])
# plt.axis('off')
# plt.imshow(minlaparr, cmap="gray"), plt.title(title_minlap)

# LOGarr = np.zeros([arrsize[1]-10, arrsize[0]-10])
# LOGmask = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
#         [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
#         [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
#         [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
#         [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
#         [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
#         [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
#         [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
#         [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
#         [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
#         [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]
# LOGarr = LOG(imgarr, LOGmask, threshold[3])
# title_LOG = 'LOG thresholds = ' + str(threshold[3])
# plt.axis('off')
# plt.imshow(LOGarr, cmap="gray"), plt.title(title_LOG)

DOGarr = np.zeros([arrsize[1]-10, arrsize[0]-10])
DOGmask = [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]
DOGarr = LOG(imgarr, DOGmask, threshold[4])
title_DOG = 'Difference of Gaussian thresholds = ' + str(threshold[4])
plt.axis('off')
plt.imshow(DOGarr, cmap="gray"), plt.title(title_DOG)

plt.show()




