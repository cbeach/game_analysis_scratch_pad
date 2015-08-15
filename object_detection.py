from collections import defaultdict, Counter
from glob import glob
import sys

import cv2
import numpy as np
from termcolor import cprint, colored

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


def match_sprite_by_run(image_runs, image_colors, sprite_runs, sprite_colors, sprite_transparency):
    for x, row in enumerate(sprite_transparency):
        for y, element in enumerate(row):
            if image_runs[x][y] != sprite_runs[x][y] or (element != 0 and
            match_pixel(image_colors[x][y], sprite_colors[x][y]) is False):
                return False
    return True


def naive_find_sprite(image, sprites, indexed):
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


def slice_image_and_sprite(image, position, sprite, offset):
    # The offset of x and y
    x_o, y_o = offset
    if y_o == 0:
        return image[position[0]:position[0] + sprite.shape[0],
                     position[1]:position[1] + sprite.shape[1]], sprite

    # The size of the image
    x_si, y_si = image.shape[:2]
    # The size of the sprite
    x_ss, y_ss = sprite.shape[:2]

    a = position[0] - x_o if position[0] - x_o >= 0 else 0
    b = (position[0] - x_o) + x_ss if (position[0] - x_o) + x_ss + 1 < x_si else None
    c = position[1] - y_o if position[1] - y_o >= 0 else 0
    d = (position[1] - y_o) + y_ss if (position[1] - y_o) + y_ss + 1 < y_si else None

    sliced_image = image[a:b, c:d]
    sprite_slice = sprite[:, :]
    if position[0] < x_o:
        a = abs(position[0] - x_o)
        sprite_slice = sprite_slice[a:]

    if position[1] < y_o:
        c = abs(position[1] - y_o)
        sprite_slice = sprite_slice[:, c:]

    return sliced_image, sprite_slice


def accumulate_image(image_runs):
    accum = np.zeros((image_runs.shape[0],), dtype='int32')
    image_runs = np.array(image_runs, dtype='int32')
    ret_val = []
    for row in image_runs.T:
        accum = np.add(accum, row)
        ret_val.append(accum)
    ret_val = np.array(ret_val)
    return ret_val.T


def hash_pixel(pixel):
    return (int(pixel[0]) << 24) + (int(pixel[1]) << 16) + (int(pixel[2]) << 8)


def hash_colors(colors):
    b = np.left_shift(colors[:, :, 0], 24)
    g = np.left_shift(colors[:, :, 1], 16)
    r = np.left_shift(colors[:, :, 2], 8)
    return np.add(np.add(b, g), r)


def runs_and_colors_to_int(runs, colors):
    return np.add(hash_colors(colors), runs)


def sort_sprites_by_size(sprites):
    sprite_names = sprites.keys()
    sprite_area = [sprites[s].shape[:2] for s in sprite_names]
    sprite_area = map(lambda p: p[0] * p[1], sprite_area)
    sorted_sprites = sorted(zip(sprite_names, sprite_area), key=lambda a: a[0])
    return [s[0] for s in sorted_sprites]


def sprite_leading_edge(runs, trans):
    leading_offsets = []
    for x, row in enumerate(trans):
        nz = np.nonzero(row)
        leading_offsets.append(int(sum(runs[x, :nz[0][0]])))
    return leading_offsets


def find_sprites_by_run(image, sprites, original):
    background_color = get_background_color(image)
    image_runs, image_colors = np_reduce_by_run(image)
    image_runs = image_runs.astype('int64')
    image_colors = image_colors.astype('int64')

    image_accum = accumulate_image(image_runs)
    hashed = runs_and_colors_to_int(image_runs, image_colors)

    sprite_runs = defaultdict(dict)
    first_runs = defaultdict(lambda: defaultdict(list))
    hashed_first = {}

    for name, sprite in sprites.items():
        runs, colors, transparent = np_reduce_by_run(sprite, transparency=True)
        sprite_runs[name]['runs'] = runs
        sprite_runs[name]['colors'] = colors
        sprite_runs[name]['trans'] = transparent

        count, color, offset = first_non_transparent_run(runs, colors, transparent)
        hashed_first[name] = hash_pixel(color) + count
        first_runs[count][tuple(color[:3])].append({
            'name': name,
            'offset': offset,
        })

    sorted_sprites = sort_sprites_by_size(sprites)
    leading_edge = {name: sprite_leading_edge(sprite_runs[name]['runs'],
        sprite_runs[name]['trans']) for name in sorted_sprites}

    restore_image(image_runs, image_colors)
    count = 0
    c_image = image.copy()
    for name, h in hashed_first.items():
        first_runs_found = np.nonzero(np.where(hashed == h, hashed, 0))
        for x, y in zip(*first_runs_found):
            img_y = int(image_accum[x][y - 1])

            if img_y < image.shape[1] / 2:
                continue

            if optimized_match_spite(image_runs, image_colors, image_accum,
                                     sprite_runs[name]['runs'], sprite_runs[name]['colors'],
                                     sprite_runs[name]['trans'], x, y, leading_edge[name], image):
                image_colors, hashed_image = erase_sprite_runs(image_runs, image_colors,
                    image_accum, sprite_runs[name]['runs'], sprite_runs[name]['colors'],
                    sprite_runs[name]['trans'], leading_edge[name],
                    background_color, x, y, hashed)
                cprint('True', 'green')
            else:
                cprint('False', 'red')
                # cv2.circle(original, (int(sum(image_runs[x][:y])) * 2, x * 2), 5, (0, 255, 0))
                # show_image(expand_image(restore_image(image_runs, image_colors)))
            count += 1
        # show_image(expand_image(c_image))


def optimized_match_spite(image_runs, image_colors, image_accum, sprite_runs, sprite_colors,
                          sprite_trans, x, y, leading_edge, image):
    #TODO: Create a decision tree for this damn thing
    img_y = int(sum(image_runs[x][:y]))
    tlc = (x, img_y - leading_edge[0])
    image_colors = image_colors.astype('ubyte')
    sprite_colors = sprite_colors.astype('ubyte')[:, :, :3]
    for i, l in enumerate(leading_edge):
        index = np.searchsorted(image_accum[x + i], tlc[1] + l)
        sprite_nz = np.nonzero(sprite_trans[i])
        first_nz = sprite_nz[0][0]
        last_nz = sprite_nz[0][-1] + 1
        for j, t in enumerate(sprite_trans[i][first_nz:last_nz]):
            print('{}: {}'.format(image_colors[x + i][index + j + 1], sprite_colors[i][first_nz + j]))
            if t != 0 and not np.array_equal(image_colors[x + i][index + j + 1],
                                             sprite_colors[i][first_nz + j]):
                #image_colors[x + i][index + j + 1]
                #TODO: - if this is the last run, and the image run is the same color, but a
                #        larger value, then the loop should just continue.
                #      - Similar situation for the first run.
                #      - For solid runs that can be contained in larger solid runs

                #return False
                if i == 4:
                    fill_color = np.array([255, 0, 255], dtype='ubyte')
                else:
                    fill_color = np.array([0, 0, 255], dtype='ubyte')
                np.copyto(image_colors[x + i][index + j + 1], fill_color)
            else:
                if i == 3:
                    fill_color = np.array([255, 0, 0], dtype='ubyte')
                else:
                    fill_color = np.array([0, 255, 0], dtype='ubyte')
                np.copyto(image_colors[x + i][index + j + 1], fill_color)

    show_image(expand_image(restore_image(image_runs, image_colors)))
        # np.copyto(image[x + i][tlc[1] + l], fill_color)
    #show_image(expand_image(restore_image(image_runs, image_colors)))
    return True


def erase_sprite_runs(image_runs, image_colors, image_accum, sprite_runs, sprite_colors,
                      sprite_trans, leading_edge, fill_color, x, y, hashed_image):
    fill_color = np.array(fill_color, dtype='ubyte')
    hashed_fill_color = hash_pixel(fill_color)
    img_y = int(sum(image_runs[x][:y]))
    tlc = (x, img_y - leading_edge[0])
    for i, l in enumerate(leading_edge):
        index = np.searchsorted(image_accum[x + i], tlc[1] - l)
        sprite_nz = np.nonzero(sprite_trans[i])
        first_nz = sprite_nz[0][0]
        last_nz = sprite_nz[0][-1] + 1
        for j, t in enumerate(sprite_trans[i][first_nz:last_nz]):
            if t != 0:
                np.copyto(image_colors[x + i][index + j + 1], fill_color)
                count = image_runs[x + i][index + j + 1]
                hashed_image[x + i][index + j + 1] = hashed_fill_color + count
    return image_colors, hashed_image


def old_find_sprites_by_run(first_runs, image_runs, image_colors, sprite, image, sprites, sprite_runs):
    counter = Counter()
    iterations = 0
    first = sorted(first_runs.items(), key=lambda x: x[0], reverse=True)
    for count, colors in first:
        found_runs = np.nonzero(np.where(image_runs == count, image_runs, 0))
        for x, y in zip(found_runs[0], found_runs[1]):
            for c in colors.keys():
                if c == tuple(image_colors[x][y][:3]):
                    for sprite_data in colors[c]:
                        iterations += 1
                        img_y = int(sum(image_runs[x][:y]))
                        adjusted_size = (image.shape[0] - sprite.shape[0],
                                        image.shape[1] - sprite.shape[1])
                        if x > adjusted_size[0] or img_y > adjusted_size[1]:
                            continue

                        sliced_image, sliced_sprite = slice_image_and_sprite(image, (x, img_y),
                            sprites[sprite_data['name']], sprite_data['offset'])

                        if 0 in sliced_image.shape:
                            continue

                        sliced_runs, sliced_colors = np_reduce_by_run(sliced_image)

                        s_runs = sprite_runs[sprite_data['name']]['runs']
                        s_colors = sprite_runs[sprite_data['name']]['colors']
                        s_trans = sprite_runs[sprite_data['name']]['trans']

                        if match_sprite_by_run(sliced_runs, sliced_colors, s_runs, s_colors,
                                               s_trans):
                            counter[sprite_data['name']] += 1
                            break

    print(iterations)
    #show_image(original)


def first_non_transparent_run(image_runs, image_colors, image_transparency):
    sx, sy = image_runs.shape[:2]
    first = np.nonzero(image_transparency)
    fallback = None
    for x, y in zip(first[0], first[1]):
        if image_runs[x][y] < sy:
            return image_runs[x][y], image_colors[x][y], (x, sum(image_runs[x][:y]))
        elif fallback is None:
            fallback = (image_runs[x][y], image_colors[x][y], (x, sum(image_runs[x][:y])))
    return fallback


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


def restore_image(image_runs, image_colors):
    restored_image = np.zeros(image_colors.shape, dtype='ubyte')
    x = 0
    y = 0
    for i, row in enumerate(image_runs):
        x = i
        y = 0
        for j, run in enumerate(row):
            b, g, r = image_colors[i][j][:3]
            for k in range(image_runs[i][j]):
                restored_image[x][y][0] = b
                restored_image[x][y][1] = g
                restored_image[x][y][2] = r
                y += 1
    return restored_image


def expand_image(image):
    h, w, d = image.shape[:3]
    expanded = np.zeros((h * 2, w * 2, d), dtype='ubyte')
    for x, row in enumerate(image):
        for y, p in enumerate(row):
            ex = x * 2
            ey = y * 2
            np.copyto(expanded[ex][ey], p)
            np.copyto(expanded[ex][ey + 1], p)
            np.copyto(expanded[ex + 1][ey], p)
            np.copyto(expanded[ex + 1][ey + 1], p)
    return expanded


class SpriteTree:
    def __init__(self, sprites):




def main():
    """
        Profile:
            9522
            8582: Optimized match pixel
            93.22: Find by run, mostly using Numpy

        Iterations:
            6167
            6071
    """
    file_names = glob('data/*')
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    indexed = {k: index_sprite(v) for k, v in sprites.items()}
    #sprites = {
    #    'small_jump': sprites['small_jump'],
    #    'qblock_1': sprites['qblock_1'],
    #    'qblock_2': sprites['qblock_2'],
    #    'qblock_3': sprites['qblock_3'],
    #}
    # sprite_runs = {k: np_reduce_by_run(v, transparency=True) for k, v in sprites.items()}

    accume = 0
    for fn in range(len(file_names))[1224:1234]:
        cprint('{}: {}'.format(accume, fn), 'cyan')
        fn = 'data/{}.png'.format(fn)
        original = cv2.imread(fn, cv2.IMREAD_COLOR)  # [300:, :]

        image = reduce_image(original)
        find_sprites_by_run(image, sprites, original)
        sys.exit()
        #naive_find_sprite(image, sprites, indexed)


if __name__ == '__main__':
    main()
