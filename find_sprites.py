from glob import glob
from hashlib import sha1
import os
import pickle
import sys
import time

import cv2
import numpy as np
from termcolor import cprint

from object_detection import reduce_image, show_image
from sprite_sheet_tools import get_sprites
from sprite_tree import SpriteTree
from test_images import ImageFactory


class SegmentedImage:
    cached_palette = set()

    def __init__(self, sprite_tree, image):
        self._sprite_tree = sprite_tree
        self._image = image
        self.hash_image()

    def unhash_pixel(self, px):
        return (px & 0xFF0000) >> 16, (px & 0x00FF00) >> 8, (px & 0x0000FF)

    def get_image_palette(self):
        return np.unique(self._hashed)

    def hash_image(self):
        if hasattr(self, '_hashed') and self._hashed is not None:
            return self._hashed
        b, g, r = cv2.split(self._image)
        bs = np.multiply(b, 2 ** 16)
        gs = np.multiply(g, 2 ** 8)
        # cache the hashed image
        self._hashed = np.add(r, np.add(bs, gs))
        # return it for consistency
        return self._hashed

    def hashed_bg_color_to_zero(self, hashed_image):
        bg_color = self.background_color()
        return np.where(hashed_image != bg_color, hashed_image, 0)

    def find_sprites(self):
        hashed = self.hashed_bg_color_to_zero(self.hash_image())
        probs = np.zeros(hashed.shape + (3,), dtype='ubyte')
        not_zeros = np.nonzero(hashed)
        names = self._sprite_tree._names
        sprite_names = []
        for x, y in zip(*not_zeros):
            p = self._sprite_tree.most_probable_sprite(hashed, x, y)
            index_of_max = np.argmax(p)
            sprite_names.append(names[index_of_max])
            if names[index_of_max] == 'small_jump':
                color = np.array((0, 255, 0), dtype='ubyte')
            else:
                color = np.array((0, 0, 255), dtype='ubyte')

            np.copyto(probs[x][y], color)

        return probs

    def background_color(self, hashed=True):
        return 0xFF848a if hashed is True else (255, 132, 138)

    def marked_sprites(self):
        marked = np.zeros_like(self._image, dtype='float')
        for i, row in enumerate(self._hashed):
            for j, px in row:
                self._st


def main():
    np.set_printoptions(linewidth=150)

    # Read the code for this file and hash it
    with open('sprite_tree.py', 'r') as fp:
        my_hash_value = sha1(fp.read()).hexdigest()

    # The pickling should really be done in the sprite_tree, but this is less
    # of a PITA
    file_name = 'pickled_sprites/{}'.format(my_hash_value)
    if os.path.exists(file_name):
        cprint('Unpickling SpriteTree', 'green')
        with open(file_name, 'r') as fp:
            st = pickle.load(fp)
    else:
        cprint('Pickling SpriteTree', 'yellow')
        sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
        st = SpriteTree(sprites)
        with open(file_name, 'w') as fp:
            pickle.dump(st, fp)

    img_f = ImageFactory(sprites, background_color=(255, 132, 138))

    image = np.zeros((224, 256, 3), dtype='ubyte')
    for row in image:
        for px in row:
            np.copyto(px, img_f._bg_color)

    s = sprites['small_jump']
    # blit(self, image, sprite, mask, x, y):
    image = img_f.blit(image, s[:, :, :3], s[:, :, 3], 50, 50)
    si = SegmentedImage(st, image)
    img = si.find_sprites()
    show_image(img)

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
