from glob import glob
from collections import Counter
import sys
import time

from matplotlib import pyplot as plt
from termcolor import cprint
import cv2
import numpy as np


def bgr2rgb(image):
    image = image.copy()
    for x, row in enumerate(image):
        for y, pixel in enumerate(row):
            image[x][y] = np.array((pixel[2], pixel[1], pixel[0]))
    return image


def get_alpha_connected_pixels(image, coord):
    """
        Two adjacent pixels are connected if both have alpha values of 255.
        Scan image until a pixel with an alpha value of 255 is found, flood fill until
        all connected pixels have been isolated.
    """

    x, y = coord
    x_size, y_size, depth = image.shape

    max_x = x_size - 1
    max_y = y_size - 1

    connected = []
    if x > 0 and image[x - 1][y][3] == 255:
        connected.append(((x - 1, y), image[x - 1][y]))
        image[x - 1][y][3] = 0

    if x < max_x and image[x + 1][y][3] == 255:
        connected.append(((x + 1, y), image[x + 1][y]))
        image[x + 1][y][3] = 0

    if y > 0 and image[x][y - 1][3] == 255:
        connected.append(((x, y - 1), image[x][y - 1]))
        image[x][y - 1][3] = 0

    if y < max_y and image[x][y + 1][3] == 255:
        connected.append(((x, y + 1), image[x][y + 1]))
        image[x][y + 1][3] = 0

    return connected


def get_color_connected_pixels(image, coord, null_color=(1, 1, 1)):
    """
        Two adjacent pixels are connected if both the same color:
    """
    x, y = coord
    x_size, y_size, depth = image.shape
    null_fill = np.array(null_color)
    pixel = image[x][y]
    if np.array_equal(pixel, null_fill):
        return []

    max_x = x_size - 1
    max_y = y_size - 1

    connected = []
    if x > 0 and np.array_equal(image[x - 1][y], pixel) and not np.array_equal(image[x - 1][y], null_fill):
        connected.append(((x - 1, y), image[x - 1][y]))
        image[x - 1][y] = np.array(null_fill)

    if x < max_x and np.array_equal(image[x + 1][y], pixel) and not np.array_equal(image[x + 1][y], null_fill):
        connected.append(((x + 1, y), image[x + 1][y]))
        image[x + 1][y] = np.array(null_fill)

    if y > 0 and np.array_equal(image[x][y - 1], pixel) and not np.array_equal(image[x][y - 1], null_fill):
        connected.append(((x, y - 1), image[x][y - 1]))
        image[x][y - 1] = np.array(null_fill)

    if y < max_y and np.array_equal(image[x][y + 1], pixel) and not np.array_equal(image[x][y + 1], null_fill):
        connected.append(((x, y + 1), image[x][y + 1]))
        image[x][y + 1] = np.array(null_fill)

    return connected


def get_color_groups(image):
    sprites = []
    for x, row in enumerate(image):
        for y, pixel in enumerate(row):
            if np.array_equal(pixel, np.array((1, 1, 1))):
                continue
            stack = [(x, y)]
            sprite = [((x, y), pixel)]
            while len(stack) > 0:
                coords = stack.pop()
                #print('{}: {}'.format(coords, image[coords[0]][coords[1]]))
                connected = get_color_connected_pixels(image, coords)
                for i in connected:
                    if np.array_equal(np.array(i[1]), np.array([1, 1, 1])):
                        cprint('Boooooo', 'red')
                connected_coords = [c[0] for c in connected]
                stack.extend(connected_coords)
                sprite.extend(connected)
            sprites.append(sprite)

    return sprites


def get_alpha_groups(image):
    sprites = []
    for x, row in enumerate(image):
        for y, pixel in enumerate(row):
            if pixel[3] == 255:
                stack = [(x, y)]
                sprite = [((x, y), pixel)]
                while len(stack) > 0:
                    coords = stack.pop()
                    connected = get_alpha_connected_pixels(image, coords)
                    connected_coords = [c[0] for c in connected]
                    stack.extend(connected_coords)
                    sprite.extend(connected)
                sprites.append(sprite)

    return sprites


def calc_bounding_boxes(sprites):
    bounding_box = lambda x1, y1, x2, y2: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    boxes = []
    x = lambda c: c[0]
    y = lambda c: c[1]
    for s in sprites:
        coords = [p[0] for p in s]
        boxes.append(bounding_box(min(coords, key=x)[0], min(coords, key=y)[1],
                                  max(coords, key=x)[0], max(coords, key=y)[1]))
    return boxes


def get_sprites(sprite_sheet):

    sprite_pixel_groups = get_alpha_groups(sprite_sheet)
    bounding_boxes = calc_bounding_boxes(sprite_pixel_groups)

    sprites = []
    for bb in bounding_boxes:
        sprites.append(sprite_sheet[bb['x1']:bb['x2'], bb['y1']:bb['y2']])

    return sprites


def get_uniform_areas(image):
    color_groups = get_color_groups(image)
    bounding_boxes = calc_bounding_boxes(color_groups)
    return zip(color_groups, bounding_boxes)


def get_palette(image, as_counter=False):
    palette = []
    for row in image:
        for pixel in row:
            palette.append((pixel[0], pixel[1], pixel[2]))

    if as_counter is False:
        return list(set(palette))
    if as_counter is True:
        return Counter(palette)


if __name__ == '__main__':
    templates = {
        'mario': [
            cv2.imread('sprites/mario/small_jump.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/mario/small_standing.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/mario/small_run_1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/mario/small_run_2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/mario/small_run_3.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/mario/big_standing.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
        ],
        'tile': [
            cv2.imread('sprites/tiles/ground.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/block.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/qblock_spent.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/bricks.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/qblock_3.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/qblock_1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/tiles/qblock_2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
        ],
        'entities': [
            cv2.imread('sprites/entities/mushroom.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/entities/goomba_1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
            cv2.imread('sprites/entities/goomba_2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE),
        ],
    }

    color_frame = cv2.imread('data/625.png', cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    frame = gray_frame.copy()
    frame2 = gray_frame.copy()

    template_wad = []
    template_wad.extend(templates['mario'])
    template_wad.extend(templates['tile'])
    template_wad.extend(templates['entities'])

    for template in template_wad:
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.99
        loc = np.where(res >= threshold)
        w, h = templates['tile'][0].shape
        for pt in zip(*loc[::-1]):
            cv2.rectangle(color_frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(bgr2rgb(color_frame))
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('tiles')

    plt.show()
