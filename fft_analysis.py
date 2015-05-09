import cv2
import itertools
import csv
from glob import glob
import random
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import rfft


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


def get_tiles(image, harmonics):
    ret_val = np.zeros(len(image))
    ret_val_2 = []
    for h in harmonics:
        for i, row in enumerate(image):
            ret_val_2.append((np.mean(row), np.std(row)))
            previous_slice = None
            for k, slc in enumerate(np.split(row, h)):
                if reduce(lambda a, b: a == True and b == True, np.equal(previous_slice, slc)):
                    ret_val[i] += 1
                previous_slice = slc
    return ret_val, ret_val_2


def analyze_files(func, source_files, dest, fraction=1.0, *args, **kwargs):
    files = random.sample(source_files, int(len(source_files) * fraction))
    for i, f in enumerate(files):
        if i < 1045:
            continue
        print("File {} of {} processed. {}% complete".format(i, len(files),
            (float(i) / float(len(files))) * 100))
        try:
            func(f, dest, i, *args, **kwargs)
        except Exception:
            traceback.print_exc()


def perform_fft(path, dest, file_name, vertical=False):
    gray_image = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if vertical is True:
        gray_image = np.rot90(gray_image)
    orientation = 'vertical' if vertical is True else 'horizontal'

    fft = []
    for i, row in enumerate(gray_image):
        fft.append(rfft(row))
    del gray_image

    with open('{}/{}/{}.csv'.format(dest, orientation, file_name), 'wb') as output:
        csv_out = csv.writer(output)
        for row in fft:
            csv_out.writerow(row)


def display_lines_with_repeating_tiles():
    d = []
    with open('analyzed/accumulated_fft.csv', 'r') as fp:
        reader = csv.reader(fp)
        for i in reader:
            d = i

    d = [float(i) for i in d]

    der = first_derivitives(d)
    peak = find_peaks(d)
    s_peak = sorted(peak)

    mean = np.mean(d)
    std = np.std(d)

    m = np.repeat(mean, len(d))

    freqs = [i for i, j in enumerate(d) if j > mean + 2.5 * std]
    print(freqs)

    combos = list(itertools.combinations(freqs, 2))
    freqs = []
    for i in combos:
        if (i[1] % i[0]) == 0:
            freqs.append(i[0])
            freqs.append(i[1])

    sorted_harmonics = sorted(set(freqs))
    print(sorted_harmonics)
    files = glob('data/*')

    percent = 1
    files = glob('data/*')
    image = 'data/283.png'
    image = cv2.imread(image)
    #image = image[423:439, :16]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiled_lines, averages = get_tiles(gray_image, [16])  # sorted_harmonics)

    for i, t in enumerate(tiled_lines):
        if averages[i][1] > 30 and t > 12:
            image[i] = [(0, 0, 255)] * len(image[t])

    cv2.rectangle(image, (0, 400), (31, 432), (255, 0, 0))
    cv2.namedWindow("Display", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("Display", image)
    cv2.waitKey(0)


def accumulate_csv_data(csv_files, dest):
    accumulator = None
    for i, f in enumerate(csv_files):
        print("File {} of {} processed. {}% complete".format(i, len(files),
            (float(i) / float(len(files))) * 100))
        with open(f, 'r') as fp:
            reader = csv.reader(fp)
            for row in reader:
                if accumulator is None:
                    accumulator = np.zeros(len(row))
                accumulator = np.add(accumulator, np.array(map(abs, map(float, row))))
    return accumulator


if __name__ == '__main__':
    files = glob('analyzed/fft/vertical/*')
    #analyze_files(perform_fft, files, 'analyzed/fft', fraction=1.0, vertical=True)
    accume = accumulate_csv_data(files, 'analyzed/vertical_fft_accume.csv')
    with open('analyzed/vertical_fft_accum.csv', 'wb') as fp:
        writer = csv.writer(fp)
        writer.writerow(accume)
    plt.plot(accume)
    plt.show()
