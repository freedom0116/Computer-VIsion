from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = Image.open("lena.bmp")
imgarr = np.array(img)
arrsize = img.size
temparr = np.zeros(arrsize)

def Binarize():    
    newarr = np.zeros(arrsize)
    hist = np.zeros(256)
    for i in range(arrsize[0]):
        for j in range(arrsize[1]):
            if imgarr[i][j] < 128:
                newarr[i][j] = 0
                temparr[i][j] = 0
            else:
                newarr[i][j] = 255                
                temparr[i][j] = 1
            hist[imgarr[i][j]] += 1
    newimg = Image.fromarray(newarr)
#    newimg.show()
    x = np.arange(256)
    plt.bar(x, hist)
#    plt.show()


def Connected():    
    group = 1 #the group of white
    for i in range(arrsize[0]): #row
        for j in range(arrsize[1] - 1): #colume
            right_one = j + 1
            if temparr[i][j] > 0: #this is white                 
                temparr[i][j] = group
                if temparr[i][right_one] == 0:                 
                    group += 1
                else:                       
                    temparr[i][right_one] = temparr[i][j]

    change = 0
    a = 0
    for c in range(1):
        change = 0
        print("charge0")
        #infect right
        count = 0
        for i in range(arrsize[0]): #row
            for j in range(arrsize[1] - 1): #colume
                right_one = j + 1
                if temparr[i][j] > 0: #this is white
                    if temparr[i][right_one] > 0:                        
                        if temparr[i][right_one] != temparr[i][j]: #whether change
                            count += 1
                            temparr[i][right_one] = temparr[i][j]
        if(count == 0):
            change += 1
            print("charge1")  
        #infect down
        count = 0
        for i in range(arrsize[0] - 1): #row
            down_one = i + 1
            for j in range(arrsize[1]): #colume
                if temparr[i][j] > 0: #this is white
                    if temparr[down_one][j] > 0:
                        if temparr[down_one][j] != temparr[i][j]:
                            count += 1
                            temparr[down_one][j] = temparr[i][j]
        if(count == 0):
            change += 1
            print("charge2") 
        #infect left
        count = 0
        for i in range(arrsize[0]): #row
            for j in range(arrsize[1] - 1): #colume
                new_j = arrsize[1] - 1 - j
                left_one = new_j - 1
                if temparr[i][new_j] > 0: #this is white
                    if temparr[i][left_one] > 0:
                        if temparr[i][left_one] != temparr[i][new_j]:
                            count += 1
                            temparr[i][left_one] = temparr[i][new_j]
        if(count == 0):
            change += 1 
            print("charge3")
        #infect up
        count = 0
        for i in range(arrsize[0] - 1): #row
            new_i = arrsize[0] - 1 - i
            up_one = i - 1
            for j in range(arrsize[1]): #colume
                if temparr[new_i][j] > 0: #this is white
                    if temparr[up_one][j] > 0:
                        if temparr[up_one][j] != temparr[new_i][j]:
                            count += 1
                            temparr[up_one][j] = temparr[new_i][j]
        if(count == 0):
            change += 1
            print("charge4")  
        outputimg = Image.fromarray(temparr)
        outputimg.show()
    
    
if __name__=="__main__":
    Binarize()
    Connected()
