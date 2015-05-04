import csv
from glob import glob
import os
import sys
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import rfft
import numpy as np

pngs = glob('data/*')


def get_ffts(image):
    ffts = []
    for line in image:
        ffts.append(rfft(line))
    return ffts


for i, path in enumerate(pngs):
    if i < 2219:
        continue
    print('Analyzing image {} of {}. {}% done'.format(i, len(pngs),
        (float(i) / float(len(pngs)) * 100)))
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_fft = get_ffts(gray_image)
    with open('analyzed/fft/{}.csv'.format(i), 'wb') as output:
        csv_out = csv.writer(output)
        for row in gray_image_fft:
            csv_out.writerow(row)
    del image
    del gray_image
    del gray_image_fft
