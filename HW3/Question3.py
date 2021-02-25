import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import matplotlib.image as mpimg

img = mpimg.imread('Images/ParkingLot.jpg') 
plt.hist(img)
plt.show()