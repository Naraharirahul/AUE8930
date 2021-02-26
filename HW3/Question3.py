import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import cv2  
from pylab import *

img = cv2.imread('images/ParkingLot.jpg')
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3.1
hist(img.flatten(),256)
show()

thresh = 235

binary_img = np.zeros(shape = (img.shape[0], img.shape[1]))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if gray_scale[i,j] > thresh:
            binary_img[i,j] = 1

plt.imshow(binary_img,cmap = 'gray')
plt.show()

# 3.2

canny_edges = cv2.Canny(gray_scale,170,220,None,3)

lines = cv2.HoughLinesP(canny_edges,1,np.pi / 180, 50, None, 50, 10)

for line in lines:
    l = line[0]
    cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA) 
    

plt.imshow(img)
plt.show()


# 3.4
fig,ax = plt.subplots(1)

slope = np.zeros(len(lines))
parking_lines =np.zeros(shape = (len(lines),4))
for i in range(len(lines)):
    l = lines[i][0]
    # slope[i] = (l[3] - l[1])/(l[2] - l[0])
    dist = sqrt((l[0] - l[2])**2 + (l[1] - l[3])**2)
    if dist > 220 and dist < 245:
        parking_lines[i] = lines[i]

for i in range(len(parking_lines)):
    c = parking_lines[i]
    if (c[0] and c[1] > 0.5) and (c[2] and c[3] > 0.5):
        cv2.circle(img,(int(c[0]), int(c[1])), 10, (255,0,0), 3)
        cv2.circle(img,(int(c[2]), int(c[3])), 10, (255,0,0), 3)
        cv2.circle(img,(int((c[2] + c[0])/ 2), int((c[3] + c[1]) / 2)), 10, (0,255,0), 3)

pts1 = np.array([[204, 78], [285, 56], [353, 145], [280, 163]])
pts2 = np.array([[120, 88], [204, 78], [280, 163], [186, 183]])
pts3 = np.array([[184, 185], [253, 275], [351, 253], [280, 166]])
pts4 = np.array([[280, 166] , [362, 145], [430, 225], [345, 255]])
pts5 = np.array([[37, 110], [100, 207], [188, 186], [118, 92]])
pts6 = np.array([[188, 186] , [255, 277], [160, 306], [100, 206]])
pts1 = pts1.reshape((-1,1,2))
pts2 = pts2.reshape((-1,1,2))
pts3 = pts3.reshape((-1,1,2))
pts4 = pts4.reshape((-1,1,2))
pts5 = pts5.reshape((-1,1,2))
pts6 = pts6.reshape((-1,1,2))

# cv2.fillPoly(img,[pts1],(100,100,0))
# cv2.fillPoly(img,[pts2],(40,100,0))
# cv2.fillPoly(img,[pts3],(40,140,0))
# cv2.fillPoly(img,[pts4],(0,180,0))
# cv2.fillPoly(img,[pts5],(0,140,40))
# cv2.fillPoly(img,[pts6],(0,180,100))

plt.imshow(img)
plt.show()
