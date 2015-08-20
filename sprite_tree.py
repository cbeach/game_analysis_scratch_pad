from collections import Counter, defaultdict
import sys

import numpy as np
from termcolor import cprint

from object_detection import reduce_image
from sprite_sheet_tools import get_sprites


class SpriteTree:
    def __init__(self, sprites):
        self._sprite_dict = sprites
        sprites = sprites.items()
        self._names = [k for k, v in sprites]
        self._sprites = [v for k, v in sprites]

        self._full_palette = self.get_full_palette(self._sprites)
        self._p_sprites = self.palettize_sprites(self._sprites)
        self._index = self.index_sprites()
        self._hashed = self.hash_index(self._index)
        self._probabilities = self.compute_probabilities()
        self.create_tree(self._sprites, self._probabilities)

    def hash_index(self, index):
        hashed = []
        for i in index:
            hashed.append(i['first_palette']
                          + (i['first_pixel'][0] << 8)
                          + (i['first_pixel'][1] << 16))
        return hashed

    def palettize_sprites(self, sprites):
        p_sprites = []
        for s in sprites:
            palettized = np.zeros(s.shape[:2], dtype='int')
            non_trans = np.nonzero(s[:, :, 3])
            for x, y in zip(*non_trans):
                palettized[x][y] = self._full_palette[tuple(s[x][y][:3])]
            p_sprites.append(palettized)
        return p_sprites

    def get_sprite_palette(self, sprite):
        if len(sprite.shape) == 2:
            height, width = sprite.shape
        elif len(sprite.shape) == 3:
            height, width, depth = sprite.shape
        else:
            raise ValueError(('sprite array has incorrect dimensionality. sprite shape has {} '
                            'dimensions, but must have either 2 or 3.').format(len(sprite.shape)))

        palette = []
        for row in sprite:
            for pixel in row:
                if pixel[3] != 0:
                    palette.append(tuple(pixel[:3]))

        return Counter(palette)

    def get_full_palette(self, sprites):
        palette = defaultdict(int)
        for sprite in sprites:
            p = self.get_sprite_palette(sprite)
            for k, v in p.items():
                palette[k] += v

        palette = sorted(Counter(palette).items(), key=lambda a: a[1], reverse=True)
        return {color[0]: number + 1 for number, color in enumerate(palette)}

    def first_non_trans_pixel(self, sprite):
        return zip(*np.nonzero(sprite[:, :, 3]))[0]

    def index_sprites(self):
        index = []
        for i, s in enumerate(self._sprites):
            sprite_index = {}
            p_0x, p_0y = self.first_non_trans_pixel(s)
            sprite_index['first_pixel'] = (p_0x, p_0y)
            sprite_index['first_palette'] = self._full_palette[tuple(s[p_0x][p_0y][:3])]
            index.append(sprite_index)
        return index

    def max_sprite_shape(self, arrays=None):
        arrays = self._sprites if arrays is None else arrays
        max_height = max([v.shape[0] for v in arrays])
        max_width = max([v.shape[1] for v in arrays])
        return (max_height, max_width)

    def pad_to_size(self, array, x, y):
        if array.shape[0] > x or array.shape[1] > y:
            raise ValueError('Padded array size must be larger than the given array on all dimensions')

        height = x - array.shape[0]
        width = y - array.shape[1]
        if len(array.shape) == 2:
            return np.pad(array, ((0, height), (0, width)), 'constant', constant_values=0)
        elif len(array.shape) == 3:
            return np.pad(array, ((0, height), (0, width), (0, 0)), 'constant', constant_values=0)

    def padded_array_and_mask(self, array, x, y):
        mask = np.ones(array.shape[:2])
        return self.pad_to_size(array, x, y), self.pad_to_size(mask, x, y)

    def create_tree(self, sprites, probabilities):
        # [fp_clr][sprite][x][y]
        def sorting_function(a, b):
            return abs(0.5 - prob[a[0]][a[1]]) > abs(0.5 - prob[b[0]][b[1]])

        for frst_px, sprt_list in probabilities.items():
            for i, index in enumerate(sprt_list['indices']):
                prob = sprt_list['prob'][i]
                sprite = sprites[index]
                filtered = np.logical_not(np.isclose(prob, 1))
                filtered = zip(*np.nonzero(np.where(filtered, prob, 0)))


    def stack_arrays(self, arrays):
        max_height, max_width = self.max_sprite_shape(arrays=arrays)
        arrays = arrays.copy()
        for i, a in enumerate(arrays):
            a = self.pad_to_size(a, max_height, max_width)

        return np.dstack(*arrays)

    def stack_arrays_and_masks(self, arrays):
        max_height, max_width = self.max_sprite_shape(arrays=arrays)
        arrays = [a.copy() for a in arrays]
        reshaped_arrays = []
        masks = []
        for i, a in enumerate(arrays):
            arr, m = self.padded_array_and_mask(a, max_height, max_width)
            reshaped_arrays.append(arr)
            masks.append(m)

        return np.dstack(reshaped_arrays), np.dstack(masks)

    def compute_probabilities(self):
        max_height, max_width = self.max_sprite_shape()
        stack, masks = self.stack_arrays_and_masks(self._p_sprites)

        # Group the sprites by the hashed value of their first pixels
        grouped_sprites = defaultdict(list)
        # grouped_masks = defaultdict(list)
        # grouped_names = defaultdict(list)
        grouped_indices = defaultdict(list)
        for i, h in enumerate(self._hashed):
            h = h & 0xFF
            grouped_sprites[h].append(stack[:, :, i])
            # grouped_masks[h].append(masks[:, :, i])
            # grouped_names[h].append(self._names[i])
            grouped_indices[h].append(i)

        grouped_sprites = {h: np.array(sprites) for h, sprites in grouped_sprites.items()}
        # grouped_masks = {h: np.array(masks) for h, masks in grouped_masks.items()}

        # Currently trying to figure out the probability calculation algorith
        # [fp_clr][sprite][x][y]
        probabilities = {}
        for i, k_v_pair in enumerate(grouped_sprites.items()):
            hsh, sprites = k_v_pair
            prob = np.zeros(sprites.shape)
            for j, s in enumerate(sprites):
                for x, row in enumerate(s):
                    for y, pix in enumerate(row):
                        colors_across_sprites = sprites[:, x, y]
                        counter = np.bincount(colors_across_sprites)
                        if len(counter) == 1:
                            continue
                        clrs = np.nonzero(counter)[0]
                        prob_space = np.sum(counter)
                        if s[x][y] in clrs:
                            l = s[x][y]
                            prob[j][x][y] = float(counter[l]) / float(prob_space)
            probabilities[hsh] = {
                'prob': prob,
                'indices': grouped_indices[hsh]
            }
        return probabilities


def main():
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    #sprites = {
    #    'small_run_1': sprites['small_run_1'],
    #    'big_run_1': sprites['big_run_1'],
    #    'big_run_3': sprites['big_run_3'],
    #}
    SpriteTree(sprites)


if __name__ == '__main__':
    main()
