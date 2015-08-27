import random

import cv2
import numpy as np
from termcolor import cprint

from object_detection import reduce_image, show_image
from sprite_sheet_tools import get_sprites


class ImageFactory:
    def __init__(self, sprites, background_color=None):
        self._names = sprites.keys()
        self._sprites = [s[:, :, :3] for s in sprites.values()]
        self._masks = [s[:, :, 3] for s in sprites.values()]
        self._bg_color = np.array(background_color, dtype='ubyte') \
            if background_color is not None else None

    def new_image(self, shape=(224, 256, 3), num_sprites=1, stacking=False, sprite=None,
            cross_edge=False):
        image = np.zeros(shape, dtype='ubyte')
        if self._bg_color is not None:
            for row in image:
                for px in row:
                    np.copyto(px, self._bg_color)

        bounding_boxes = []
        sprites = []
        total_sprites = 0
        iterations = 0

        while(total_sprites < num_sprites):
            name, sprite, mask = random.choice(zip(self._names, self._sprites, self._masks))
            if cross_edge is False:
                x = random.randint(0, shape[0])
                y = random.randint(0, shape[1])
            else:
                x = random.randint(-(sprite.shape[0] - 1), shape[0])
                y = random.randint(-(sprite.shape[1] - 1), shape[1])

            bounding_box = (
                (x, y),
                (x + sprite.shape[0], y + sprite.shape[1])
            )
            iterations += 1
            if stacking is False and self.obscures(bounding_box, bounding_boxes) is True:
                continue

            max_x, max_y = sprite.shape[:2]
            if cross_edge is False and (x + max_x > image.shape[0] or y + max_y > image.shape[1]):
                continue

            bounding_boxes.append(bounding_box)
            sprites.append({
                'name': name,
                'position': (x, y),
                'bounding_box': bounding_box,
            })
            image = self.blit(image, sprite, mask, x, y)
            total_sprites += 1

        return {
            'image': image,
            'sprites': sprites,
        }

    def obscures(self, new_bounding_box, old_bounding_boxes):
        """
           http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        """
        def ccw(a, b, c):
            # Are these points listed in counter clockwise order?
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def intersects(a, b):
            return ccw(a[0], b[0], b[1]) != ccw(a[1], b[0], b[1]) and ccw(a[0], a[1], b[0]) != ccw(a[0], a[1], b[1])

        nbb = new_bounding_box
        nx1, ny1 = nbb[0][0], nbb[0][1]
        nx2, ny2 = nbb[1][0], nbb[1][1]

        tlc = (nx1, ny1)
        trc = (nx1, ny2)
        blc = (nx2, ny1)
        brc = (nx2, ny2)

        n_top = (tlc, trc)
        n_bottom = (blc, brc)
        n_right = (trc, brc)
        n_left = (tlc, blc)

        for obb in old_bounding_boxes:
            ox1, oy1 = obb[0][0], obb[0][1]
            ox2, oy2 = obb[1][0], obb[1][1]

            tlc = (ox1, oy1)
            trc = (ox1, oy2)
            blc = (ox2, oy1)
            brc = (ox2, oy2)

            o_top = (tlc, trc)
            o_bottom = (blc, brc)
            o_right = (trc, brc)
            o_left = (tlc, blc)

            # Do any of the edges of the bounding boxes intersect?
            if intersects(n_top, o_right) or intersects(n_top, o_left):
                return True
            elif intersects(n_bottom, o_right) or intersects(n_bottom, o_left):
                return True
            elif intersects(n_right, o_top) or intersects(n_right, o_bottom):
                return True
            elif intersects(n_left, o_top) or intersects(n_left, o_bottom):
                return True

            # Are any of the corners of one sprite contained within the
            # boundary of the other?
            if ox1 < nx1 < ox2 and oy1 < ny1 < oy2:
                return True
            elif ox1 < nx1 < ox2 and oy1 < ny2 < oy2:
                return True
            elif ox1 < nx2 < ox2 and oy1 < ny1 < oy2:
                return True
            elif ox1 < nx2 < ox2 and oy1 < ny2 < oy2:
                return True

        return False

    def blit(self, image, sprite, mask, x, y):
        i_shape = image.shape
        s_shape = sprite.shape
        img = image.copy()

        sliced_sprite = sprite[:, :, :]
        sliced_mask = mask[:, :]
        if x < 0 and x + s_shape[0] > 0:
            sliced_image = img[abs(x):, :, :]
            sliced_sprite = sprite[abs(x):, :, :]
            sliced_mask = mask[abs(x):, :]
        elif x + s_shape[0] > i_shape[0]:
            sliced_image = img[abs(x):, :, :]
            sliced_sprite = sprite[:i_shape[0] - x, :, :]
            sliced_mask = mask[:i_shape[0] - x, :]
        else:
            sliced_image = img[x:x + s_shape[0], :, :]

        if y < 0 and y + s_shape[1] > 0:
            sliced_image = sliced_image[:, abs(y):, :]
            sliced_sprite = sliced_sprite[:, abs(y):, :]
            sliced_mask = sliced_mask[:, abs(y):]
        elif y + s_shape[1] > i_shape[1]:
            sliced_image = sliced_image[:, y:, :]
            sliced_sprite = sliced_sprite[:, :i_shape[1] - y, :]
            sliced_mask = sliced_mask[:, :i_shape[1] - y]
        else:
            sliced_image = sliced_image[:, y: y + s_shape[1], :]

        blitted = np.where(np.dstack(np.array([sliced_mask] * 3)) == 0, sliced_image, sliced_sprite)
        np.copyto(sliced_image, blitted)

        return img


def main():
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}

    img_factory = ImageFactory(sprites, background_color=(255, 132, 138))
    blitted = img_factory.new_image(num_sprites=10)
    show_image(blitted['image'])


if __name__ == '__main__':
    main()
