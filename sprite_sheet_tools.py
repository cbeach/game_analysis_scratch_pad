from collections import Counter
import sys
import time

from termcolor import cprint
import cv2
import numpy as np


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


def get_pallette(image):
    pallette = []
    for row in frame:
        for pixel in row:
            pallette.append((pixel[0], pixel[1], pixel[2]))
    return list(set(pallette))


if __name__ == '__main__':
    sprite_sheet = cv2.imread('sprites/mario_fixed_pallette.png', cv2.IMREAD_UNCHANGED)
    sprites = get_sprites(sprite_sheet)

    frame = cv2.imread('data/625.png', cv2.IMREAD_COLOR)
    areas = get_uniform_areas(frame)

    cv2.namedWindow("Display", cv2.CV_WINDOW_AUTOSIZE)
    for cg, bb in areas:
        rect_frame = np.array(frame, copy=True)
        cv2.rectangle(rect_frame, (bb['y1'], bb['x1']), (bb['y2'], bb['x2']), (0, 0, 0))
        cv2.imshow("Display", rect_frame)
        time.sleep(.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #for v, c in Counter(pallette).items():
    #    print('{}: {}'.format(v, c))

    #image = cv2.imread('mario-luigi-cropped.png')
    #for s in sprites:
    #    cv2.namedWindow("Display", cv2.CV_WINDOW_AUTOSIZE)
    #    cv2.imshow("Display", image)
    #    cv2.waitKey(0)
