"""


"""
from glob import glob
import sys
import time

import cv2
import numpy as np
from termcolor import cprint

from object_detection import reduce_image
from sprite_sheet_tools import get_sprites
from sprite_tree import SpriteTree


class SegmentedImage:
    cached_palette = set()

    def __init__(self, sprite_tree, image):
        self._sprite_tree = sprite_tree
        self._image = image
        self.hash_image()
        self._palettized_image = self.palettize_image()

    def unhash_pixel(self, px):
        return (px & 0xFF0000) >> 16, (px & 0x00FF00) >> 8, (px & 0x0000FF)

    def get_image_palette(self):
        return np.unique(self._hashed)

    def hash_image(self):
        if hasattr(self, '_hashed') and self._hashed is not None:
            return self._hashed
        cprint(type(self._image), 'blue')
        b, g, r = cv2.split(self._image)
        bs = np.multiply(b, 2 ** 16)
        gs = np.multiply(g, 2 ** 8)
        # cache the hashed image
        self._hashed = np.add(r, np.add(bs, gs))
        # return it for consistency
        return self._hashed

    def find_sprites(self):
        hashed = self.hash_image()
        bg_color = self.background_color()
        non_bg = np.nonzero(np.where(hashed != bg_color, hashed, 0))
        for x, y in zip(*non_bg):
            self._sprite_tree.id_sprite(self._image, x, y)

    def palettize_image(self):
        bg_color = np.array(self.background_color(hashed=False))
        palettized_image = np.zeros(self._image.shape[:2])
        for x, row in enumerate(self._image):
            for y, px in enumerate(row):
                if np.array_equal(px, bg_color):
                    continue
                c = self._sprite_tree._full_palette[tuple(px[:3])]
                palettized_image[x][y] = c

    def background_color(self, hashed=True):
        return 0xFF848a if hashed is True else (255, 132, 138)

    def marked_sprites(self):
        marked = np.zeros_like(self._image, dtype='float')
        for i, row in enumerate(self._hashed):
            for j, px in row:
                self._st


def main():
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    st = SpriteTree(sprites)

    single_sprite = cv2.imread('data/test/small_jump.png', cv2.IMREAD_COLOR)
    sys.exit()
    # obscured = cv2.imread('data/test/obscured.png', cv2.IMREAD_COLOR)

    single_sprite_si = SegmentedImage(st, single_sprite)
    single_sprite_si.marked_sprites()
    sys.exit()

    file_names = glob('data/*')
    images = []
    for fn in range(len(file_names))[1224:3224]:
        fn = 'data/{}.png'.format(fn)
        original = cv2.imread(fn, cv2.IMREAD_COLOR)  # [300:, :]
        image = reduce_image(original)
        images.append(image)

    start = time.time()
    for i, image in enumerate(images):
        si = SegmentedImage(st, image)
        si.find_sprites()
        cprint((time.time() - start) / float(i + 1), 'green')

    cprint((time.time() - start) / 1000.0, 'green')

    # obscured_si = SegmentedImage(st, obscured)

    # cv2.imwrite('single_sprite.png', single_sprite_si.marked_sprites())
    # cv2.imwrite('obscured.png', obscured_si.marked_sprites())

if __name__ == '__main__':
    main()