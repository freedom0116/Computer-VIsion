from PIL import Image
import numpy as np
import cv2

img = Image.open("lena.bmp")
imgarr = np.array(img)
arrsize = img.size

def UpsideDown():    
    nowarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        nowarr[i] = imgarr[arrsize[0]-1-i]    
    newimg = Image.fromarray(nowarr)
    newimg.show()

def RightsideLeft():
    nowarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        for j in range(0, arrsize[1]):
            nowarr[i][j] = imgarr[i][arrsize[0]-1-j]    
    newimg = Image.fromarray(nowarr)
    newimg.show()
        
def DiagonallyMirrored():
    nowarr = np.zeros(arrsize)
    for i in range(0, arrsize[0]):
        for j in range(0, arrsize[1]):
            nowarr[i][j] = imgarr[j][i]    
    newimg = Image.fromarray(nowarr)
    newimg.show()

if __name__=="__main__":
    UpsideDown()
    RightsideLeft()
    DiagonallyMirrored()
    