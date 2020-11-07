import cv2
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte

B=1 #blocksize
fn3= 'quantization.jpg'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
h , w = np.array(img1.shape[:2])/B * B
h = int(h)
w = int(w)
print(h)
print(w)
img1 = img1[:h,:w]
blocksV = h/B
blocksV = int(blocksV)
blocksH = w/B
blocksH = int(blocksH)
maxlength = h*w
random.seed('junior')
ChoiceKey = [1,-1]
keyIndex = [random.choice(ChoiceKey) for i in range(maxlength)]

print('--------keyIndex----------------')
print(keyIndex[0])
vis0 = np.zeros((h,w), np.double)
vis0[:h, :w] = img1
print('-------quantized image-------------------------------')
print(vis0)
print('------------------------------------------------------')
#--------Inverse quantization------------------------------------------------------------------------------------------------------------------
Inversequantized = np.zeros((h,w), np.double)

bit = 255
# print(np.max(vis0))
# print(np.min(vis0))
qmin = -365.64155180316163
# np.min(vis0)
qmax = 355.11415317021476 
# np.max(vis0)
qrange = qmax-qmin
step = qrange/((2^bit)-1)

# 355.11415317021476
# -365.64155180316163
print('-------Trans max-------------------------------')
print(qmax)
print('-------Trans min-------------------------------')
print(qmin)
print('-------step-------------------------------')
print(step)

for  row in range(blocksV):
    for col in range(blocksH):
        amponly = vis0[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        Inversequantized[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = np.round((amponly + qmin)*step)

# for row in range(blocksV):
#     amponly = vis0[row*B:(row+1)*B, 0:blocksH]
#     Inversequantized[row*B:(row+1)*B, 0:blocksH] = np.round((amponly + qmin)*step)

# cv2.imwrite('Inversequantized.jpg', Inversequantized)
print('-------Inversequantized-------------------------------')
print(Inversequantized)
print('------------------------------------------------------')
# print('-------Max quantized-------------------------------')
# print(np.max(vis0))
# print('-------Min quantized-------------------------------')
# print(np.min(vis0))
print('-------max Inversequantized quantized-------------------------------')
print(np.max(Inversequantized))
print('-------min Inversequantized quantized-------------------------------')
print(np.min(Inversequantized))
# % Inverse quantization
# iqamponly=step.*qamponly+min(min(amponly1));
# figure,imshow(iqamponly);

#-----------dct---------------------------------------------------------------------------------------------------------------

dctImage = np.zeros((h,w), np.double)

for row in range(blocksV):
    currentblock = cv2.dct(Inversequantized[row*B:(row+1)*B, 0:blocksH])
    dctImage[row*B:(row+1)*B, 0:blocksH] = currentblock

# cv2.imwrite('DcsdctImage.jpg', dctImage)

# print('-------max -------------------------------')
# print(np.min(dctImage))
# print('-------min -------------------------------')
# print(np.min(dctImage))
# print('-------Dnc dct -------------------------------')
# print(dctImage)
#-----------Sign Scrambling---------------------------------------------------------------------------------------------------------------

DcsSign = np.zeros((h,w), np.double)

global DscPositionKeyIndex
DscPositionKeyIndex = 0

for  row in range(blocksV):
    for col in range(blocksH):
        Dsck = dctImage[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        DsckN = Dsck*keyIndex[DscPositionKeyIndex]
        DcsSign[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = DsckN
        DscPositionKeyIndex = DscPositionKeyIndex + 1

DscPositionKeyIndex = 0

# global positionKeyIndex
# positionKeyIndex = 0

# for  row in range(blocksV):
#     for col in range(blocksH):
#         ck = dctImage[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
#         ck = ck*keyIndex[positionKeyIndex]
#         dctImage[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = ck
#         positionKeyIndex = positionKeyIndex + 1
        
# positionKeyIndex = 0

# cv2.imwrite('DcsSign.jpg', DcsSign)

print('-------sign dct -------------------------------')
print(DcsSign)

#-----------idct Scrambling---------------------------------------------------------------------------------------------------------------

DcsIdctImage = np.zeros((h,w), np.double)

for row in range(blocksV):
    cTrans = cv2.idct(dctImage[row*B:(row+1)*B, 0:blocksH])
    DcsIdctImage[row*B:(row+1)*B, 0:blocksH] = cTrans

cv2.imwrite('DcsIdctImage.jpg', DcsIdctImage)

print('-------DcsIdctImage Idct -------------------------------')
print(DcsIdctImage)
#--------quantization------------------------------------------------------------------------------------------------------------------
# Dcsquantized = np.zeros((h,w), np.double)

# print(np.max(DcsIdctImage))
# print(np.min(DcsIdctImage))
# Dcsqmin = np.min(DcsIdctImage)
# Dcsqmax = np.max(DcsIdctImage)
# Dcsqrange = Dcsqmax-Dcsqmin
# Dcsstep = Dcsqrange/((2^bit)-1)

# for row in range(blocksV):
#     Dcsamponly = DcsIdctImage[row*B:(row+1)*B, 0:blocksH]
#     Dcsquantized[row*B:(row+1)*B, 0:blocksH] = np.round((Dcsamponly - Dcsqmin)/Dcsstep)

# print('-------Dcsquantized-------------------------------')
# print(Dcsquantized)
# cv2.imwrite('Dcsquantized.jpg', Dcsquantized)
