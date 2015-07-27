from collections import defaultdict
from glob import glob
import random
import sys

import cv2
import numpy as np
from termcolor import cprint

from sprite_sheet_tools import image_palette, get_sprites


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def BGR_to_palette(image, palette=None):
    if palette is None:
        palette = image_palette(image)

    return np.array([
        [palette[tuple(pixel)] for pixel in column]
        for column in image
    ])


def get_wavelets(image, palette=None, wavelet_height=4, wavelet_width=4):
    if image.shape[0] % wavelet_height != 0:
        remainder = image.shape[0] % wavelet_height
        image = image[:-remainder]
    if image.shape[1] % wavelet_width != 0:
        remainder = image.shape[1] % wavelet_height
        image = image[:, :-remainder]

    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, depth = image.shape
    else:
        raise ValueError(('Image array has incorrect dimensionality. Image shape has {} '
                         'dimensions, but must have either 2 or 3.').format(len(image.shape)))

    split_image = np.split(image, height / wavelet_height)

    wavelets = []
    for i, quad in enumerate(split_image):
        # TODO: Change this in the emulator!
        # A single pixel in the Nintendo is action a 2x2 pixel area in the
        # image. I'm consolidating the 4x4 ROI into a 2x2 ROI for efficiency.
        temp = [np.split(i, width / wavelet_width) for i in quad]
        column = []
        for j in range(len(temp[0])):
            column.append(np.array([
                temp[0][j][0], temp[1][j][0],
                temp[0][j][1], temp[1][j][1],
            ]))
        wavelets.append(column)

    wavelet_dict = defaultdict(list)
    for i, column in enumerate(wavelets):
        for j, w in enumerate(column):
            wavelet_dict[tuple(w.flatten())].append({
                'position': (i, j)
            })
    return wavelet_dict


def sprite_wavelets(image):
    pass


def reduce_image(image):
    return image[::2, ::2]


def byte_array_to_int(a):
    return (a[0] << 16) + (a[1] << 8) + a[2]


def get_background_color(image):
    return (255, 132, 138)


def index_sprite(sprite):
    index = {}
    # First non-transparent pixel
    b, g, r, a = cv2.split(sprite)
    for i, row in enumerate(sprite):
        for j, pixel in enumerate(row):
            if pixel[3] > 0:
                index['first_pixel'] = {
                    'color': tuple(pixel[:3]),
                    'position': (i, j)
                }
                break
        if 'first_pixel' in index:
            break

    return index


def match_pixel(a, b):
    if len(a) == 4:
        if a[3] == 0 and np.array_equal(a[:3], b):
            return True
    elif len(b) == 4:
        if b[3] == 0 and np.array_equal(a, b[:3]):
            return True
    return False


def trim_transparent(row):
    return row[first_non_transparent_pixel_pos(row):last_non_transparent_pixel_pos(row)]


def first_non_transparent_pixel_pos(row):
    for i in range(len(row)):
        if row[i][3] != 0:
            return i


def last_non_transparent_pixel_pos(row):
    if row[-1][3] != 0:
        return None
    for i in range(1, len(row)):
        if row[-i][3] != 0:
            return -i + 1


def match_sprites(image, position, sprites, possible_matches, indexed):
    counter = 0
    for pm in possible_matches:
        match = True
        sprite = sprites[pm]
        color = indexed[pm]['first_pixel']['color']
        offset = indexed[pm]['first_pixel']['position']
        image_slice = image[position[0] - offset[0]:position[0] + sprite.shape[0] + 1,
                            position[1] - offset[1]:position[1] + sprite.shape[1] + 1]
        print(position)
        print(offset)
        print(position[0] - offset[0], position[0] + sprite.shape[0] + 1)
        print(position[1] - offset[1], position[1] + sprite.shape[1] + 1)
        print
        cv2.imwrite('analyzed/segmented/{}.png'.format(counter), image_slice)
        counter += 1

        for i, row in enumerate(image_slice):
            for j, pixel in enumerate(row):
                try:
                    if match_pixel(pixel, sprite[i][j]) is False:
                        match = False
                        break
                except IndexError:
                    break
            if match is False:
                break

        if match is True:
            return pm
    return False


def naive_find_sprite(image, sprites, indexed):
    first_pixels = {}
    for k, v in indexed.items():
        if v['first_pixel']['color'] in first_pixels:
            first_pixels[v['first_pixel']['color']].append(k)
        else:
            first_pixels[v['first_pixel']['color']] = [k]

    matches = defaultdict(list)
    for i, row in enumerate(image):
        for j, p in enumerate(row):
            pixel_tuple = tuple(p[:3])
            if pixel_tuple in first_pixels:
                match = match_sprites(image, (i, j), sprites, first_pixels[pixel_tuple], indexed)
                if match is not False:
                    matches[match].append((i, j))
    return matches


def break_image_by_color(image):
    pass


if __name__ == '__main__':
    file_names = glob('data/*')
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    sprites = {
        'small_jump': sprites['small_jump'],
        'goomba_1': sprites['goomba_1'],
    }

    indexed = {}
    for k in sprites.keys():
        get_wavelets(sprites[k], None, 2, 2)
        indexed[k] = index_sprite(sprites[k])
    # sprite_wavelets = {k: get_wavelets(v) for k, v in sprites.items()}

    accume = 0
    image = reduce_image(cv2.imread('data/test/test_image.png', cv2.IMREAD_COLOR))
    # image = reduce_image(cv2.imread('data/1228.png', cv2.IMREAD_COLOR))
    locations = naive_find_sprite(image, sprites, indexed)
    for l in locations['goomba_1']:
        image[l[0]][l[1]] = np.array([0, 0, 255])
    sys.exit()

    for fn in random.sample(file_names, 100):
        image = reduce_image(cv2.imread(fn, cv2.IMREAD_COLOR))
        naive_find_sprite(image, sprites, indexed)

        #int_image = image.astype(int)
        #b, g, r = cv2.split(int_image)
        #nbins = 256
        #b = np.left_shift(b, 16)
        #g = np.left_shift(g, 8)
        #as_ints = np.add(np.add(b, g), r)
        #counts = np.bincount(as_ints.flatten())
        #counts = map(lambda i: '{0:x}'.format(i), counts.nonzero()[0])
        #palette = map(byte_array_to_int, image_palette(image).keys())
        #palette = map(lambda i: '{0:x}'.format(i), palette)

        # p_sprites = palettize_sprites(sprites, palette)
        # wavelets = get_wavelets(image)
        accume += 1
