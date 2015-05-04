import itertools
import csv
from glob import glob
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def accumulate(acc, row):
    for i, j in enumerate(row):
        acc[i] += float(j)
    return acc


def first_derivitives(data):
    deriv = np.zeros(len(data))
    prev = data[0]
    for i, d in enumerate(data):
        if i == 0:
            continue
        deriv[i] = d - prev
        prev = d
    return deriv


def find_peaks(data):
    derive = first_derivitives(data)
    maximum = max(derive)
    minimum = min(derive)
    peaks = np.zeros(len(data))
    prev = derive[0]
    for i, d in enumerate(derive):
        if i == 0:
            continue
        if (prev >= 0 and d <= 0 and prev != d):
            peaks[i] = abs(prev)
        elif (prev <= 0 and d >= 0 and prev != d):
            peaks[i] = abs(prev)
        prev = d
    return peaks

#percent = 1
#files = glob('analyzed/fft/*')
#files = random.sample(files, int(len(files) * percent))
#accume = np.zeros(512)
#for i, f in enumerate(files):
#    print("File {} of {} processed. {}% complete".format(i, len(files),
#        (float(i) / float(len(files))) * 100))
#    with open(f, 'r') as fp:
#        reader = csv.reader(fp)
#        for row in reader:
#            for j in range(15):
#                row[j] = 0
#            accume = accumulate(accume, row)
#
#
#with open('analyzed/no_abs_accumulated_fft.csv', 'wb') as output:
#    csv_out = csv.writer(output)
#    csv_out.writerow(accume)
#
#sys.exit()
d = []
with open('analyzed/accumulated_fft.csv', 'r') as fp:
    reader = csv.reader(fp)
    for i in reader:
        d = i

d = [float(i) for i in d]
for i in d:
    if i < 0:
        print('negative')
for i in range(15):
    d[i] = 0

der = first_derivitives(d)

peak = find_peaks(d)
s_peak = sorted(peak)
#plt.plot(np.repeat(np.mean(peak), len(peak)), 'r')
#plt.plot(np.repeat(np.std(peak), len(peak)), 'g')
#plt.plot(s_peak, 'b')

print(sum([1 for i in peak if i > (np.mean(peak) + np.std(peak))]))

#plt.plot(d)
#plt.plot(der)
#plt.plot(peak)
#plt.show()
#sys.exit()

mean = np.mean(d)
std = np.std(d)
print(std)

m = np.repeat(mean, len(d))
#plt.plot(d, 'b')
#plt.plot(m, 'r')
#for i in np.linspace(.5, 5, 10):
#    plt.plot(np.repeat(i * std + mean, len(d)), 'g')

freqs = [i for i, j in enumerate(d) if j > mean + 2.5 * std]
print(freqs)

combos = list(itertools.combinations(freqs, 2))
print(combos)
freqs = []
for i in combos:
    print('{} % {} = {}'.format(i[1], i[0], i[1] % i[0]))
    if (i[1] % i[0]) == 0:
        freqs.append(i[0])
        freqs.append(i[1])

print(sorted(set(freqs)))
#plt.show()
