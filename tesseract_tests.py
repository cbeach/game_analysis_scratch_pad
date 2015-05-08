from glob import glob
import random
import sys
import time

import numpy as np
from PIL import Image
import cv2
from pytesseract import image_to_string


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


image = cv2.imread('data/283.png')
#image = auto_canny(image, sigma=1)
image = Image.fromarray(image)
image.show()
text = image_to_string(image)
print(text)
sys.exit()

with open('analyzed/image_tests', 'wb') as fp:
    files = glob('data/*')
    text = []
    for i, f in enumerate(files):
        print('file {} of {}: {}% done'.format(i, len(files), (float(i) / float(len(files))) * 100))
        text = image_to_string(Image.open(f))
        fp.write(text)
        fp.write('\n==========\n')
