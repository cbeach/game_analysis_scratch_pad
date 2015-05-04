import sys
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import rfft
import numpy as np

image = cv2.imread('data/283.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fft = []
for i in gray_image:
    fft.append(rfft(i))
fft = np.array(fft)


def kern(line):
    diffs = 0
    last = line[0]
    for i in line[0:]:
        if i == last:
            diffs += 1
        last = i
    return diffs, np.std(line), np.mean(line)


def normalize(data):
    maximum = max(data)
    return [i / maximum for i in data]


def kill_first_twenty(data):
    for i in range(20):
        data[i] = 0
    return data

diffs = []
deviations = []
means = []
for i, f in enumerate(fft):
    d, s, m = kern(f)
    diffs.append(d)
    deviations.append(s)
    means.append(m)

total_mean = np.mean(means)
total_deviation = np.std(deviations)
counter = 0
for i, line in enumerate(fft):
    for j, pixel in enumerate(line[1:]):
        dist_from_mean = (pixel - total_mean) / total_deviation
        if abs(dist_from_mean) > 20:
            print('({}, {}): {}'.format(i, j, dist_from_mean))
            counter += 1

print(counter)
for i, j in enumerate(diffs):
    if i > 400:
        gray_image[i] = np.zeros(len(gray_image[i]))

#cv2.imshow('image', gray_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

a = map(abs, kill_first_twenty(fft[75]))
b = map(abs, kill_first_twenty(fft[100]))
c = map(abs, kill_first_twenty(fft[315]))
d = map(abs, kill_first_twenty(fft[424]))


plt.plot(a, 'r')
plt.plot(b, 'g')
plt.plot(c, 'b')
plt.plot(d, 'y')
plt.show()
