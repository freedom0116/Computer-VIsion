from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

img = Image.open("lena.bmp")
imgarr = np.array(img)
arrsize = img.size
newarr = np.zeros(arrsize) #color
fianlimg = np.zeros((512,512,3))

grouparr = np.zeros(arrsize)
groupcontent = [] #1.group other.[x, y]
tempfind = []
group = [0] 

def Binarize():    
    hist = np.zeros(256)
    for i in range(arrsize[0]):
        for j in range(arrsize[1]):
            if imgarr[i][j] < 128: #black
                newarr[i][j] = 0
            else: #white
                newarr[i][j] = 255
                fianlimg[i][j] += 255
            hist[imgarr[i][j]] += 1
    newimg = Image.fromarray(newarr)
    #newimg.show()
    x = np.arange(256)
    plt.bar(x, hist)
    #plt.show()    

def FindGroup(x, y):
    ans = False #no group
    for g in range(len(groupcontent)):
        if(grouparr[x,y] == groupcontent[g][0]):
            ans = True
    if ans == False:
        grouparr[x,y] = group[0]
        group[0] += 1
    return ans

def GoArround():
    x, y = tempfind.pop()
    grouparr[x][y] = group[0]
    groupcontent[len(groupcontent)-1].append([x, y])
    left_x = x - 1
    right_x = x + 1
    top_y = y - 1
    down_y = y + 1
    if left_x >= 0 and grouparr[left_x, y] == 0 and newarr[left_x, y] == 255:
        tempfind.append([left_x, y])
        grouparr[left_x,y] = group[0]
    if left_x >= 0 and top_y >= 0 and grouparr[left_x, top_y] == 0 and newarr[left_x, top_y] == 255:
        tempfind.append([left_x, top_y])
        grouparr[left_x,top_y] = group[0]
    if top_y >= 0 and grouparr[x, top_y] == 0 and newarr[x, top_y] == 255:
        tempfind.append([x, top_y])
        grouparr[x,top_y] = group[0]
    if right_x < arrsize[1] and top_y >= 0 and grouparr[right_x, top_y] == 0 and newarr[right_x, top_y] == 255:
        tempfind.append([right_x, top_y])
        grouparr[right_x,top_y] = group[0]
    if right_x < arrsize[1] and grouparr[right_x, y] == 0 and newarr[right_x, y] == 255:
        tempfind.append([right_x, y])
        grouparr[right_x,y] = group[0]
    if down_y < arrsize[0] and right_x < arrsize[1] and grouparr[right_x, down_y] == 0 and newarr[right_x, down_y] == 255:
        tempfind.append([right_x, down_y])
        grouparr[right_x,down_y] = group[0]
    if down_y < arrsize[0] and grouparr[x, down_y] == 0 and newarr[x, down_y] == 255:
        tempfind.append([x, down_y])
        grouparr[x,down_y] = group[0]
    if left_x >= 0 and down_y < arrsize[0] and grouparr[left_x, down_y] == 0 and newarr[left_x, down_y] == 255:
        tempfind.append([left_x, down_y])
        grouparr[left_x,down_y] = group[0]
        

def Connected():    
    for i in range(arrsize[0]): #row
        print("row:"+str(i))    
        for j in range(arrsize[1]): #colume
            if(newarr[i][j] == 255):
                if FindGroup(i, j) == False:
                    groupcontent.append([group[0]]) #input new group
                    tempfind.append([i, j]) #temp find
                    while len(tempfind) != 0:
                        GoArround()

def Max(num, n):
    themax = groupcontent[num][1][n]
    for j in range(1, len(groupcontent[num])):
        if groupcontent[num][j][n] > themax:
            themax = groupcontent[num][j][n]
    return int(themax)

def Min(num, n):
    themin = groupcontent[num][1][n]
    for j in range(1, len(groupcontent[num])):
        if groupcontent[num][j][n] < themin:
            themin = groupcontent[num][j][n]
    return int(themin)

def Center(num):
    x = 0
    y = 0
    for j in range(1, len(groupcontent[num])):
        x += groupcontent[num][j][0]
        y += groupcontent[num][j][1]
    center_x = x / (len(groupcontent[num]) - 1)
    center_y = y / (len(groupcontent[num]) - 1)
    return [int(center_x), int(center_y)]

def Graphic():
    conlong = len(groupcontent)
    count = 0
    while count != conlong:
        if len(groupcontent[count]) < 500:
            groupcontent.pop(count)
            conlong -= 1
        else:
            count += 1
    cv2.imwrite('output.jpg', fianlimg)
    for i in range(len(groupcontent)):
        cv2.rectangle(fianlimg, (Min(i, 1), Min(i, 0)), (Max(i, 1), Max(i, 0)), (0, 255, 100), 1)
        cv2.line(fianlimg, (Center(i)[1]-5, Center(i)[0]-5), (Center(i)[1]+5, Center(i)[0]+5), (0,0,255),2)
        cv2.line(fianlimg, (Center(i)[1]-5, Center(i)[0]+5), (Center(i)[1]+5, Center(i)[0]-5), (0,0,255),2)
    cv2.imshow("asd", fianlimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    

if __name__=="__main__":
    Binarize()
    Connected()
    Graphic()
            

    