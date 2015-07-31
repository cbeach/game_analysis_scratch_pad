from collections import defaultdict
from glob import glob
import sys

import cv2
import numpy as np
from termcolor import colored, cprint

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


def reduce_image(image):
    return image[::2, ::2]


def byte_array_to_int(a):
    return (a[0] << 16) + (a[1] << 8) + a[2]


def get_background_color(image):
    return (255, 132, 138)


def index_sprite(sprite):
    index = {}
    # First non-transparent pixel
    blue, green, red, alpha = cv2.split(sprite)
    for i, row in enumerate(alpha):
        for j, a in enumerate(row):
            if a > 0:
                index['first_pixel'] = {
                    'color': tuple(sprite[i][j][:3]),
                    'position': (i, j)
                }
                break
        if 'first_pixel' in index:
            break

    return index


def match_pixel(a, b):
    """
        b should always be the transparrent pixel
    """
    if b[3] == 0:
        return True
    else:
        return np.array_equal(a, b[:3])


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
    for pm in possible_matches:
        match = True
        sprite = sprites[pm]
        # The offset of x and y
        x_o, y_o = indexed[pm]['first_pixel']['position']

        # The size of the image
        x_si, y_si = image.shape[:2]
        # The size of the sprite
        x_ss, y_ss = sprite.shape[:2]

        a = position[0] - x_o if position[0] - x_o >= 0 else 0
        b = (position[0] - x_o) + x_ss + 1 if (position[0] - x_o) + x_ss + 1 < x_si else None
        c = position[1] - y_o if position[1] - y_o >= 0 else 0
        d = (position[1] - y_o) + y_ss + 1 if (position[1] - y_o) + y_ss + 1 < y_si else None

        offset_pos_x = a
        offset_pos_y = c

        # Slice the image down to the size of the current sprite so it's more
        # manageable
        image_slice = image[a:b, c:d]

        sprite_slice = sprite[:, :]
        if position[0] < x_o:
            a = abs(position[0] - x_o)
            sprite_slice = sprite_slice[a:]

        if position[1] < y_o:
            c = abs(position[1] - y_o)
            sprite_slice = sprite_slice[:, c:]

        match = match_sprite_by_pixel(image_slice, sprite_slice)
        if match is True:
            return {
                pm: (offset_pos_x, offset_pos_y),
            }
    return False


def match_sprite_by_pixel(image_slice, sprite_slice):
    for i, row in enumerate(image_slice):
        for j, pixel in enumerate(row):
            try:
                if match_pixel(pixel, sprite_slice[i][j]) is False:
                    return False
            except IndexError:
                break
    return True


def naive_find_sprite(image, sprites, indexed):
    """
        Profile:
            9522
            8582: Optimized match pixel
    """
    first_pixels = {}
    for k, v in indexed.items():
        if v['first_pixel']['color'] in first_pixels:
            first_pixels[v['first_pixel']['color']].append(k)
        else:
            first_pixels[v['first_pixel']['color']] = [k]

    matches = []
    for i, row in enumerate(image):
        for j, p in enumerate(row):
            pixel_tuple = tuple(p[:3])
            if pixel_tuple in first_pixels:
                match = match_sprites(image, (i, j), sprites, first_pixels[pixel_tuple], indexed)
                if match is not False:
                    matches.append(match)
    return matches


def find_sprites_by_run(image, sprites):
    run_count, run_colors = np_reduce_by_run(image)

    sprite_runs = defaultdict(dict)
    first_runs = defaultdict(lambda: defaultdict(list))
    for name, sprite in sprites.items():
        runs, colors, transparent = np_reduce_by_run(sprite, transparency=True)
        sprite_runs[name]['runs'] = runs
        sprite_runs[name]['colors'] = colors
        sprite_runs[name]['trans'] = transparent

        count, color = first_non_transparent_run(runs, colors, transparent)
        first_runs[count][tuple(color[:3])].append(name)

    for count, colors in first_runs.items():
        np.nonzero(np.where(run_count == count, run_count, 0))


def first_non_transparent_run(image_runs, image_colors, image_transparency):
    first = np.nonzero(image_transparency)
    x = first[0][0]
    y = first[1][0]
    return image_runs[x][y], image_colors[x][y]


def break_image_by_color(image):
    h, w = image.shape[:2]
    img = image.copy()
    master_mask = np.zeros((h + 2, w + 2), np.uint8)
    counter = 0
    for i, row in enumerate(image):
        # if not first row slice row according to
        for j, pixel in enumerate(image):
            if master_mask[i][j] > 0:
                continue
            else:
                temp_mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(img, temp_mask, (j, i), (255, 0, 0))
                master_mask = np.logical_or(master_mask, temp_mask)
                cv2.imwrite('analyzed/temp/{}.png'.format(counter), np.multiply(master_mask, 255))
                counter += 1
            # element wise or with master mask
            # get bounding box for flooded area
            # slice image and mask per bounding box
            # store slices in return dictionary
            pass


def reduce_by_run(image):
    runs = []
    color = None
    count = 0
    transparency = len(image.shape) == 3 and image.shape[-1] == 4
    transparent = False

    for i, row in enumerate(image):
        color = tuple(image[i][0])
        runs.append([])
        for j, pixel in enumerate(row):
            pixel_tuple = tuple(pixel[:3])
            if transparency and pixel[3] == 0:
                transparent = True

            if color == pixel_tuple:
                count += 1
            else:
                runs[i].append({
                    'color': color,
                    'count': count,
                    'transparent': transparent,
                })
                count = 1
                color = pixel_tuple
        runs[i].append({
            'color': color,
            'count': count,
            'transparent': transparent,
        })
        count = 0
    return runs


def np_reduce_by_run(image, transparency=False):
    runs = np.zeros(image.shape[:2])
    colors = np.zeros(image.shape)
    if transparency is True:
        transparent = np.zeros(image.shape[:2])

    count = 0
    for i, row in enumerate(image):
        color = image[i][0]
        run_y = 0
        for j, pixel in enumerate(row):
            if np.array_equal(color[:3], pixel[:3]):
                count += 1
            else:
                runs[i][run_y] = count
                colors[i][run_y] = color
                if transparency is True and color[-1] != 0:
                    transparent[i][run_y] = 1
                run_y += 1
                count = 1
                color = pixel
        runs[i][run_y] = count
        colors[i][run_y] = color
        if transparency is True and color[-1] != 0:
            transparent[i][run_y] = 1
        count = 0

    if transparency is True:
        return runs, colors, transparent
    else:
        return runs, colors


def test_runs(image):
    runs = reduce_by_run(image)
    restored_image = np.zeros_like(image)
    x = 0
    y = 0
    for i, row in enumerate(runs):
        x = i
        y = 0
        for j, run in enumerate(row):
            b, g, r = run['color']
            for k in range(run['count']):
                restored_image[x][y][0] = b
                restored_image[x][y][1] = g
                restored_image[x][y][2] = r
                y += 1


if __name__ == '__main__':
    file_names = glob('data/*')
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    sprite_runs = {k: np_reduce_by_run(v) for k, v in sprites.items()}
    #sprites = {
    #    'small_jump': sprites['small_jump'],
    #    'goomba_1': sprites['goomba_1'],
    #}

    #indexed = {}
    #for k in sprites.keys():
    #    indexed[k] = index_sprite(sprites[k])

    accume = 0
    for fn in range(len(file_names))[1224:1234]:
        fn = 'data/{}.png'.format(fn)
        image = reduce_image(cv2.imread(fn, cv2.IMREAD_COLOR))
        find_sprites_by_run(image, sprites)

        cprint('{}: {}'.format(accume, fn), 'cyan')
