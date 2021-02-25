import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import matplotlib.image as mpimg

img = cv2.imread('Images/ParkingLot.jpg')
img_array = np.array(img)
plt.imshow(img)