import cv2
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte
# np.set_printoptions(threshold=sys.maxsize)
B=1 #blocksize
fn3= 't1.jpg'
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

# print('--------keyIndex----------------')
# print(keyIndex[0])
# print('--------------------------------')
vis0    = np.zeros((h,w), np.double)
Trans   = np.zeros((h,w), np.double)
BTrans  = np.zeros((h,w), np.double)
vis0[:h, :w] = img1

#------------------dct 1D-------------------------------------------------------------------------------------------------
for row in range(blocksV):
    currentblock = cv2.dct(vis0[row*B:(row+1)*B, 0:blocksH])
    Trans[row*B:(row+1)*B, 0:blocksH] = currentblock

#------------------sign scrambling-------------------------------------------------------------------------------------------------
global positionKeyIndex
positionKeyIndex = 0

for  row in range(blocksV):
    for col in range(blocksH):
        ck = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        ck = ck*keyIndex[positionKeyIndex]
        Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = ck
        positionKeyIndex = positionKeyIndex + 1
        
positionKeyIndex = 0
#--------------------------------------------------------------------------------------------------------------------------

#------------idct 1D--------------------------------------------------------------------------------------------------------------
TransIDCT   = np.zeros((h,w), np.double)

for row in range(blocksV):
    Trans01 = cv2.idct(Trans[row*B:(row+1)*B, 0:blocksH])
    TransIDCT[row*B:(row+1)*B, 0:blocksH] = Trans01

# print('-------Trans-------------------------------')
# print(Trans)
# cv2.imwrite('Transformed.jpg', Trans)
# print('-------TransIDCT-------------------------------')
# print(TransIDCT)
# cv2.imwrite('TransformedIDCT.jpg', TransIDCT)
#--------------------------------------------------------------------------------------------------------------------------

#--------quantization------------------------------------------------------------------------------------------------------------------
quantized = np.zeros((h,w), np.double)
# bit = 256
print(np.max(TransIDCT))
print(np.min(TransIDCT))
qmin = np.min(TransIDCT)
qmax = np.max(TransIDCT)
qrange = qmax-qmin
step = qrange/(255)
print('-------test tran00-------------------------------')
print(TransIDCT[0][0])
# for  row in range(blocksV):
#     for col in range(blocksH):
#         amponly = TransIDCT[row:(row+1), col:(col+1)][0][0]
#         quantized[row:(row+1), col:(col+1)][0][0] = np.round((amponly - qmin)/step)

amponly = TransIDCT[0:blocksV , 0:blocksH]
quantized[0:blocksV , 0:blocksH] = np.round((amponly - qmin)/step)

print('-------test quantized00-------------------------------')
print(quantized[0][0])
print('-------quantized-------------------------------')
print(quantized)
cv2.imwrite('quantization.jpg', quantized)

print('-------Max quantized-------------------------------')
print(np.max(quantized))
print('-------Min quantized-------------------------------')
print(np.min(quantized))
#--------------------------------------------------------------------------------------------------------------------------
print('-------Trans max-------------------------------')
print(qmax)
print('-------Trans min-------------------------------')
print(qmin)
print('-------step-------------------------------')
print(step)


#--------------invert------------------------------------------------------------------------------------------------------------
Inversequantized = np.zeros((h,w), np.double)

# for  row in range(blocksV):
#     for col in range(blocksH):
#         Inamponly = quantized[row:(row+1), col:(col+1)][0][0]
#         Inversequantized[row:(row+1), col:(col+1)][0][0] = step*(Inamponly + qmin)

Inamponly = quantized[0:blocksV , 0:blocksH]
Inversequantized[0:blocksV , 0:blocksH] = step*Inamponly + qmin

# iqamponly=step.*qamponly+min(min(amponly1));
print('-------Invert tran00-------------------------------')
print(Inversequantized[0][0])
print('-------Invert Trans max-------------------------------')
print(np.max(Inversequantized))
print('-------Invert Trans min-------------------------------')
print(np.min(Inversequantized))
print('-------step-------------------------------')
print(step)
print(Inversequantized[0:1, 0:1])

DdctImage = np.zeros((h,w), np.double)

for row in range(blocksV):
    currentblock = cv2.dct(Inversequantized[row*B:(row+1)*B, 0:blocksH])
    DdctImage[row*B:(row+1)*B, 0:blocksH] = currentblock

DcsSign = np.zeros((h,w), np.double)

global DscPositionKeyIndex
DscPositionKeyIndex = 0

for  row in range(blocksV):
    for col in range(blocksH):
        Dsck = DdctImage[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        DsckN = Dsck*keyIndex[DscPositionKeyIndex]
        DcsSign[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = DsckN
        DscPositionKeyIndex = DscPositionKeyIndex + 1

DscPositionKeyIndex = 0

DcsIdctImage = np.zeros((h,w), np.double)

for row in range(blocksV):
    cTrans = cv2.idct(DcsSign[row*B:(row+1)*B, 0:blocksH])
    DcsIdctImage[row*B:(row+1)*B, 0:blocksH] = cTrans

cv2.imwrite('DcsIdctImage.jpg', DcsIdctImage)












# % Quantization
# tic()
# bit=8;
# qmin=min(min(amponly1));
# qmax=max(max(amponly1));
# qrange=qmax-qmin;
# step=qrange/((2^bit)-1);
# qamponly=round( ( amponly1 - min(min(amponly1)) ) / step );
# toc()

# % Inverse quantization
# iqamponly=step.*qamponly+min(min(amponly1));
# figure,imshow(iqamponly);
#--------------------------------------------------------------------------------------------------------------------------


# print('-------Original-------------------------------')
# print(vis0)
# print('-------Trans-------------------------------')
# print(Trans)
# print('-------Trans-------------------------------')
# print(np.array(Trans.shape[:]))
# cv2.imwrite('Transformed.jpg', Trans)
# ptran = cv2.imwrite('Transformed.jpg', Trans)

# global DscPositionKeyIndex
# DscPositionKeyIndex = 0

# for  row in range(blocksV):
#     for col in range(blocksH):
#         Dsck = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
#         Dsck = Dsck*keyIndex[DscPositionKeyIndex]
#         Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = Dsck
#         DscPositionKeyIndex = DscPositionKeyIndex + 1

# DscPositionKeyIndex = 0

# for row in range(blocksV):
#     cTrans = cv2.idct(Trans[row*B:(row+1)*B, 0:blocksH])
#     BTrans[row*B:(row+1)*B, 0:blocksH] = cTrans
# cv2.imwrite('BackTransformed.jpg', BTrans)
# print('-------BTrans-------------------------------')
# print(BTrans)

