import cv2
import sys
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import img_as_float, color, viewer, exposure, data, img_as_ubyte
from skimage import io
from PIL import Image
# np.set_printoptions(threshold=sys.maxsize)
B=1 #blocksize
fn3= 'Transformed.jpg'
img1 = cv2.imread(fn3, 0)
print(np.array(img1.shape[:]))
h , w = np.array(img1.shape[:])/B * B
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
print('--------------------------------')
vis0    = np.zeros((h,w), np.double)
Trans   = np.zeros((h,w), np.double)
BTrans  = np.zeros((h,w), np.double)
# vis0[:h, :w] = img1
Trans[:h, :w] = img1

for row in range(blocksV):
    currentblock = cv2.dct(vis0[row*B:(row+1)*B, 0:blocksH])
    Trans[row*B:(row+1)*B, 0:blocksH] = currentblock

global positionKeyIndex
positionKeyIndex = 0

for  row in range(blocksV):
    for col in range(blocksH):
        ck = Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0]
        ck = ck*keyIndex[positionKeyIndex]
        Trans[row*B:(row+1)*B, col*B:(col+1)*B][0][0] = ck
        positionKeyIndex = positionKeyIndex + 1
positionKeyIndex = 0

print('-------Original-------------------------------')
print(vis0)
print('-------log-------------------------------')
print(np.uint8(np.log(Trans)))
print(np.uint8(np.log(np.min(np.min(Trans)))))
print(np.uint8(np.log(np.max(np.max(Trans)))))

Trans = np.uint8((np.log(Trans)-np.log(min(min(Trans))))*255/(np.log(max(max(Trans)))-np.log(min(min(Trans)))))

cv2.imwrite('Transformed.jpg', Trans)
print('-------Trans-------------------------------')
print(Trans)

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
# cv2.imwrite('BackTrans.jpg', BTrans)
# print('-------BTrans-------------------------------')
# print(BTrans)

