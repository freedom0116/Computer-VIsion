import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import random
from PIL import Image

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
imgarr = np.array(img)
arrsize = len(imgarr[0]), len(imgarr)

def setSize(noise, edge, output):
    size = len(noise[0]), len(noise)
    print("edge" + str(edge))
    for y in range(size[1]):
        for x in range(size[0]):
            output[y][x] = imgarr[y + edge][x + edge]

def SNR(noise, edge):
    size = len(noise[0]), len(noise)
    ref = np.zeros([size[1], size[0]])
    setSize(noise, edge, ref)

    Itotal = 0
    for y in range(size[0]):
        for x in range(size[1]):
            Itotal += ref[y][x]
    Us = Itotal / (size[0]*size[1])

    VStop = 0
    for y in range(size[0]):
        for x in range(size[1]):
            VStop += (ref[y][x] - Us)**2
    VS = VStop / (size[0]*size[1])

    noiseDif = 0
    for y in range(size[0]):
        for x in range(size[1]):
            noiseDif += (noise[y][x] - ref[y][x])
    Unoise = noiseDif / (size[0]*size[1])

    VNtop = 0
    for y in range(size[0]):
        for x in range(size[1]):
            VNtop += (noise[y][x] - ref[y][x] - Unoise)**2
    VN = VNtop / (size[0]*size[1])

    SNR = 20 * math.log10(math.sqrt(VS)/math.sqrt(VN))
    return SNR

    


# Guassian Noise
def Gaussian(amplitude, output):
    noise = np.random.normal(0.0, amplitude, [arrsize[0],arrsize[1]])
    for y in range(arrsize[1]):
        for x in range(arrsize[0]):
            output[y][x] = imgarr[y][x] + noise[y][x]
    return SNR(output, 0)

# Salt-and-Pepper Noise
def SaltAndPepper(prob, output):
    for y in range(arrsize[1]):
        for x in range(arrsize[0]):
            rdn = random.random()
            if rdn < prob:
                output[y][x] = 0
            elif rdn > (1 - prob):
                output[y][x] = 255
            else:
                output[y][x] = imgarr[y][x]
    return SNR(output, 0)

# Box filter
def Box(noise, size, output):
    ytime = arrsize[1] - size + 1
    xtime = arrsize[0] - size + 1
    for y in range(ytime):
        for x in range(xtime):
            total = 0
            for box_y in range(size):
                for box_x in range(size):
                    total += noise[y + box_y][x + box_x]
            output[y][x] = total / size**2
    edge = int((size - 1) / 2)
    return SNR(output, edge)

# Median filter
def Median(noise, size, output):
    ytime = arrsize[1] - size + 1
    xtime = arrsize[0] - size + 1
    for y in range(ytime):
        for x in range(xtime):
            objlist = [0.0 for i in range(size**2)]
            i = 0
            for med_y in range(size):
                for med_x in range(size):
                    objlist[i] = noise[y + med_y][x + med_x]
                    i += 1
            output[y][x] = np.median(objlist)
    edge = int((size - 1) / 2)
    return SNR(output, edge)

# Dilatio use octogonal 3-5-5-5-3 kernel
def Dilation(arr, refarr, x, y):
    size = len(arr[0]), len(arr)
    for i in range(-2, 3):
        if(i == 0 or i == 4):
             for j in range(-1, 2):
                if((y + i) >= 0 and (y + i) < size[0] and (x + j) >= 0 and (x + j) < size[1]): 
                    if(arr[y][x] < refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]
        else:
            for j in range(-2, 3):
                if((y + i) >= 0 and (y + i) < size[0] and (x + j) >= 0 and (x + j) < size[1]): 
                    if(arr[y][x] < refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]

# Erosion use octogonal 3-5-5-5-3 kernel
def Erosion(arr, refarr, x, y):
    size = len(arr[0]), len(arr)
    for i in range(-2, 3):
        if(i == 0 or i == 4):
             for j in range(-1, 2):
                if((y + i) >= 0 and (y + i) < size[0] and (x + j) >= 0 and (x + j) < size[1]): 
                    if(arr[y][x] > refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]
        else:
            for j in range(-2, 3):
                if((y + i) >= 0 and (y + i) < size[0] and (x + j) >= 0 and (x + j) < size[1]): 
                    if(arr[y][x] > refarr[y + i][x + j]):
                        arr[y][x] = refarr[y + i][x + j]

# Opening then Closing
def Open_Close(noise, output):
    size = len(noise[0]), len(noise)
    # Dilation
    dilarr = np.array(noise)
    for y in range(size[0]):
        for x in range(size[1]):
            Dilation(dilarr, noise, x, y)
    # Erosion
    eroarr = np.array(noise)
    for y in range(size[0]):
        for x in range(size[1]):
            Erosion(eroarr, noise, x, y)
    # Opening
    fuck = np.array(eroarr)
    for y in range(size[0]):
        for x in range(size[1]):
            Dilation(fuck, eroarr, x, y)
    # Closing     
    for y in range(size[0]):
        for x in range(size[1]):
            Erosion(fuck, dilarr, x, y)
    
    for y in range(size[0]):
        for x in range(size[1]):
            output[y][x] = fuck[y][x]
    return SNR(output, 0)

# closing the Opening
def Close_Open(noise, output):
    size = len(noise[0]), len(noise)
    # Dilation
    dilarr = np.array(noise)
    for y in range(size[0]):
        for x in range(size[1]):
            Dilation(dilarr, noise, x, y)
    # Erosion
    eroarr = np.array(noise)
    for y in range(size[0]):
        for x in range(size[1]):
            Erosion(eroarr, noise, x, y)    
    # Closing
    fuck = np.array(dilarr)
    for y in range(size[0]):
        for x in range(size[1]):
            Erosion(fuck, dilarr, x, y)
    # Opening
    for y in range(size[0]):
        for x in range(size[1]):
            Dilation(fuck, eroarr, x, y)
    
    for y in range(size[0]):
        for x in range(size[1]):
            output[y][x] = fuck[y][x]

    return SNR(output, 0)

gau10arr = np.zeros([arrsize[0],arrsize[1]], dtype = np.float)
snr_gau10 = Gaussian(10, gau10arr)
# title = 'Gaussian Noise 10 SNR = ' + str(round(snr_gau10, 3))
# plt.imshow(gau10arr, cmap = "gray"), plt.title(title)
# plt.show()

gau30arr = np.zeros([arrsize[0],arrsize[1]], dtype = np.float)
snr_gau30 = Gaussian(30, gau30arr)
# title = 'Gaussian Noise 30 SNR = ' + str(round(snr_gau30, 3))
# plt.imshow(gau30arr, cmap = "gray"), plt.title(title)
# plt.show()

SP005arr = np.zeros([arrsize[0],arrsize[1]], dtype = np.float)
snr_SP005 = SaltAndPepper(0.05, SP005arr)
# title = 'Salt-and-Pepper Noise 0.05 SNR = ' + str(round(snr_SP005, 3))
# plt.imshow(SP005arr, cmap = "gray"), plt.title(title)
# plt.show()

SP01arr = np.zeros([arrsize[0],arrsize[1]], dtype = np.float)
snr_SP01 = SaltAndPepper(0.1, SP01arr)
# title = 'Salt-and-Pepper Noise 0.1 SNR = ' + str(round(snr_SP01, 3))
# plt.imshow(SP01arr, cmap = "gray"), plt.title(title)
# plt.show()


def ShowNoise(noise, n_snr, name, toptitle):
    box3 = np.zeros([arrsize[0] - 1, arrsize[1] - 1], dtype = np.float)
    snr_box3 = Box(noise, 3, box3)
    title_box3 = 'box 3x3 SNR = ' + str(round(snr_box3, 3))

    box5 = np.zeros([arrsize[0] - 2, arrsize[1] - 2], dtype = np.float)
    snr_box5 = Box(noise, 5, box5)
    title_box5 = 'box 5x5 SNR = ' + str(round(snr_box5, 3))

    med3 = np.zeros([arrsize[0] - 1, arrsize[1] - 1], dtype = np.float)
    snr_med3 = Median(noise, 3, med3)
    title_med3 = 'median 3x3 SNR = ' + str(round(snr_med3, 3))

    med5 = np.zeros([arrsize[0] - 2, arrsize[1] - 2], dtype = np.float)
    snr_med5 = Median(noise, 5, med5)
    title_med5 = 'median 5x5 SNR = ' + str(round(snr_med5, 3))

    CO = np.zeros([arrsize[0], arrsize[1]], dtype = np.float)
    snr_co = Close_Open(noise, CO)
    title_co = 'Closing-Opening SNR = ' + str(round(snr_co, 3))
    
    OC = np.zeros([arrsize[0], arrsize[1]], dtype = np.float)
    snr_oc = Open_Close(noise, OC)
    title_oc = 'Opening-Closing SNR = ' + str(round(snr_oc, 3))

    plt.suptitle(toptitle)
    plt.subplot(3,2,1)
    plt.axis('off')
    plt.imshow(box3, cmap="gray"), plt.title(title_box3)
    
    plt.subplot(3,2,2)
    plt.axis('off')
    plt.imshow(box5, cmap="gray"), plt.title(title_box5)
    
    plt.subplot(3,2,3)
    plt.axis('off')
    plt.imshow(med3, cmap="gray"), plt.title(title_med3)
    
    plt.subplot(3,2,4)
    plt.axis('off')
    plt.imshow(med5, cmap="gray"), plt.title(title_med5)
    
    plt.subplot(3,2,5)
    plt.axis('off')
    plt.imshow(CO, cmap="gray"), plt.title(title_co)
    
    plt.subplot(3,2,6)
    plt.axis('off')
    plt.imshow(OC, cmap="gray"), plt.title(title_oc)
    plt.show()

ShowNoise(gau10arr, snr_gau10, "Gaussian", 'Gaussian noise, amplitude = 10')
ShowNoise(gau30arr, snr_gau30, "Gaussian", 'Gaussian noise, amplitude = 30')
ShowNoise(SP005arr, snr_SP005, "Salt-and-Pepper noise", 'Salt-and-Pepper noise, probability = 0.05')
# ShowNoise(SP01arr, snr_SP01, "Salt-and-Pepper noise", 'Salt-and-Pepper noise, probability = 0.1')