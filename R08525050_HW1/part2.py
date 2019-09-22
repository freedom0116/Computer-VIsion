from PIL import Image
import numpy as np
import cv2
import math

img = Image.open("lena.bmp")
imgarr = np.array(img)
arrsize = img.size

def Rotate45():    
    newarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        for j in range(0, arrsize[1]):
            point = np.array([i-256, j-256])
            rotatematrix = np.array([[math.cos(-math.pi/4), math.sin(-math.pi/4)],\
                                     [-math.sin(-math.pi/4), math.cos(-math.pi/4)]])
            newpoint = rotatematrix.dot(point)
            newpoint +=256
            if newpoint[0] >= 0 and newpoint[0] < 512 and newpoint[1] >= 0 and newpoint[1] < 512:
                newarr[int(newpoint[0])][int(newpoint[1])] = imgarr[i][j]  
    newimg = Image.fromarray(newarr)
    newimg.show()

def ShrinkHalf():    
    newarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        for j in range(0, arrsize[1]):
            if i%2 == 0 and j% 2 == 0:
                newarr[int(i/2)+128][int(j/2)+128] = imgarr[i][j]
    newimg = Image.fromarray(newarr)
    newimg.show()

def Binarize():
    newarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        for j in range(0, arrsize[1]):
            if imgarr[i][j] < 128:
                newarr[i][j] = 0
            else:
                newarr[i][j] = 255
    newimg = Image.fromarray(newarr)
    newimg.show()

if __name__=="__main__":
    Rotate45()
    ShrinkHalf()
    Binarize()
            
