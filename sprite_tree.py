from collections import Counter, defaultdict
import math
import sys

import cv2
import numpy as np
from termcolor import cprint

from object_detection import reduce_image, show_image
from sprite_sheet_tools import get_sprites


class SpriteTree:
    def __init__(self, sprites):
        # temp_names is used for sanity checking the most probably sprite
        # algorithm
        self.temp_names = set()
        self._sprite_dict = sprites
        sprites = sprites.items()
        self._names = [k for k, v in sprites]
        self._sprites = [v for k, v in sprites]

        self._full_palette = self.get_full_palette(self._sprites)
        self._palette_lookup = {c: i for i, c in enumerate(self._full_palette)}
        self._p_sprites = self.hash_sprites()
        self._index = self.index_sprites()
        self._hashed = self.hash_index(self._index)
        self._probabilities = self.group_and_compute_probabilities()
        self._sorted_probs = self.sort_probabilites(self._sprites, self._probabilities)
        self._color_probabilities = self.index_color_probabilities()
        self._pairwise_probabilities = self.index_pairwise_probabilities()
        self._patches = self.get_patches()
        self._patch_probabilites = self.index_patch_probabilities()

    def hash_index(self, index):
        hashed = []
        for i in index:
            hashed.append(i['first_palette']
                          + (i['first_pixel'][0] << 8)
                          + (i['first_pixel'][1] << 16))
        return hashed

    def hash_sprites(self):
        p_sprites = []
        for sprite in self._sprites:
            b, g, r, a = cv2.split(sprite)
            bs = np.multiply(b, 2 ** 16)
            gs = np.multiply(g, 2 ** 8)
            # cache the hashed image
            hashed = np.add(r, np.add(bs, gs))
            p_sprites.append(np.where(a != 0, hashed, 0))
        return p_sprites

    def get_sprite_palette(self, sprite):
        if len(sprite.shape) == 2:
            height, width = sprite.shape
        elif len(sprite.shape) == 3:
            height, width, depth = sprite.shape
        else:
            raise ValueError(('sprite array has incorrect dimensionality. sprite shape has {} '
                            'dimensions, but must have either 2 or 3.').format(len(sprite.shape)))

        b, g, r, a = cv2.split(sprite)
        bs = np.multiply(b, 2 ** 16)
        gs = np.multiply(g, 2 ** 8)
        # cache the hashed image
        hashed = np.add(r, np.add(bs, gs))
        opaque = np.where(a != 0, hashed, 0)
        opaque_colors = np.unique(opaque)
        # return it for consistency
        return opaque_colors

    def get_full_palette(self, sprites):
        palette = set()
        for sprite in sprites:
            temp = self.get_sprite_palette(sprite)
            for p in temp:
                palette.add(p)

        return np.array(sorted(list(palette)))

    def first_non_trans_pixel(self, sprite):
        return zip(*np.nonzero(sprite[:, :, 3]))[0]

    def hash_pixel(self, b, g, r, a=255):
        if a == 0:
            return 0
        return (b << 16) + (g << 8) + r

    def index_sprites(self):
        index = []
        for i, s in enumerate(self._sprites):
            sprite_index = {}
            p_0x, p_0y = self.first_non_trans_pixel(s)
            sprite_index['first_pixel'] = (p_0x, p_0y)
            sprite_index['first_palette'] = self.hash_pixel(*s[p_0x][p_0y])
            index.append(sprite_index)
        return index

    def hash_pair(self, a, b):
        try:
            iter(a)
            a = self._full_palette(tuple(a))
        except TypeError:
            # The value a is a palette value
            pass

        try:
            iter(b)
            b = self._full_palette(tuple(b))
        except TypeError:
            # The value a is a palette value
            pass
        power = int(math.ceil(math.log(max(self._full_palette.values()), 2)))
        return (a << power) + b

    def get_horizontal_pairs(self, p_sprite):
        return_pairs = []
        for i in range(2):
            pairs = np.array_split(p_sprite, np.arange(i, p_sprite.shape[1], 2), 1)
            for pair_set in pairs:
                if pair_set.shape[1] < 2:
                    continue
                for pair in pair_set:
                    return_pairs.append(pair)
        return return_pairs

    def get_vertical_pairs(self, p_sprite):
        return_pairs = []
        p_sprite = p_sprite.T
        for i in range(2):
            pairs = np.array_split(p_sprite, np.arange(i, p_sprite.shape[1], 2), 1)
            for pair_set in pairs:
                if pair_set.shape[1] < 2:
                    continue
                for pair in pair_set:
                    return_pairs.append(pair)
        return return_pairs

    def index_pairwise_probabilities(self):
        max_color_val = len(self._full_palette) + 1
        all_pairs = []
        for s in self._p_sprites:
            s_pairs = []
            s_pairs.extend(self.get_horizontal_pairs(s))
            per_sprite_totals = np.zeros((max_color_val + 1, max_color_val + 1))
            for a, b in s_pairs:
                a = self._palette_lookup[a]
                b = self._palette_lookup[b]
                per_sprite_totals[a][b] += 1
            all_pairs.append(per_sprite_totals)
        all_pairs = np.array(all_pairs)
        totals = np.sum(all_pairs, 0)

        np.seterr(invalid='ignore')
        probs = {
            'horz': np.nan_to_num(np.divide(all_pairs, totals)),
        }
        np.seterr()

        all_pairs = []
        for s in self._p_sprites:
            s_pairs = []
            s_pairs.extend(self.get_vertical_pairs(s))
            per_sprite_totals = np.zeros((max_color_val + 1, max_color_val + 1))
            for a, b in s_pairs:
                a = self._palette_lookup[a]
                b = self._palette_lookup[b]
                per_sprite_totals[a][b] += 1
            all_pairs.append(per_sprite_totals)
        all_pairs = np.array(all_pairs)
        totals = np.sum(all_pairs, 0)

        np.seterr(invalid='ignore')
        probs['vert'] = np.nan_to_num(np.divide(all_pairs, totals))
        np.seterr()

        return probs

    def index_color_probabilities(self):
        """
        Given a color, what is the probability that that pixel belongs to each sprite.
        """
        array_len = len(self._full_palette) + 1
        per_sprite_totals = []
        for i, s in enumerate(self._p_sprites):
            bins = np.unique(s.flatten())
            per_sprite_totals.append(np.pad(bins, (0, array_len - bins.shape[0]), 'constant'))

        per_sprite_totals = np.array(per_sprite_totals, dtype=float)
        totals = np.sum(per_sprite_totals, 0)
        np.seterr(all='ignore')
        prob = np.nan_to_num(np.divide(per_sprite_totals, totals.astype(float)))
        np.seterr()

        return prob

    def index_patch_probabilities(self):
        flattened = [j for i in self._patches for j in i]
        heights = {i: v for i, v in enumerate(np.bincount([f['height'] for f in flattened])) if v != 0}
        widths = {i: v for i, v in enumerate(np.bincount([f['width'] for f in flattened])) if v != 0}
        areas = {i: v for i, v in enumerate(np.bincount([f['area'] for f in flattened])) if v != 0}
        bounding_boxes = Counter([f['bounding_box'] for f in flattened])
        runs = Counter([f['runs'] for f in flattened])

        per_sprite_counts = []
        for sprite in self._patches:
            h = [patch['height'] for patch in sprite]
            w = [patch['width'] for patch in sprite]
            a = [patch['area'] for patch in sprite]
            bb = [patch['bounding_box'] for patch in sprite]
            r = [patch['runs'] for patch in sprite]

            h = {k: float(v) / float(heights[k]) for k, v in Counter(h).items()}
            w = {k: float(v) / float(widths[k]) for k, v in Counter(w).items()}
            a = {k: float(v) / float(areas[k]) for k, v in Counter(a).items()}
            bb = {k: float(v) / float(bounding_boxes[k]) for k, v in Counter(bb).items()}
            r = {k: float(v) / float(runs[k]) for k, v in Counter(r).items()}

            per_sprite_counts.append({
                'height': h,
                'width': w,
                'area': a,
                'bounding_box': bb,
                'runs': r,
            })
        return per_sprite_counts

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

    def sort_probabilites(self, sprites, probabilities):
        # [fp_clr][sprite][x][y]
        # Sort acording to distance from 50%
        def sorting_function(a, b):
            dist_a = abs(0.5 - prob[a[0]][a[1]])
            dist_b = abs(0.5 - prob[b[0]][b[1]])
            if dist_a < dist_b:
                return -1
            elif dist_a == dist_b:
                return 0
            elif dist_a > dist_b:
                return 1

        sorted_probs = defaultdict(list)
        for first_px, sprite_list in probabilities.items():
            for i, index in enumerate(sprite_list['indices']):
                prob = sprite_list['prob'][i]
                filtered = np.logical_not(np.isclose(prob, 1))
                filtered = zip(*np.nonzero(np.where(filtered, prob, 0)))
                sorted_probs[first_px].append(sorted(filtered, sorting_function))
        return sorted_probs

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

    def group_and_compute_probabilities(self):
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
        probabilities = {}
        for i, k_v_pair in enumerate(grouped_sprites.items()):
            hsh, sprites = k_v_pair
            probabilities[hsh] = self.compute_probabilites(sprites, grouped_indices[hsh], hsh)
        return probabilities

    def compute_probabilites(self, sprites, indices, first_pixel, given=[]):
        # Currently trying to figure out the probability calculation algorith
        # [fp_clr][sprite][x][y]
        prob = np.zeros(sprites.shape)
        for j, s in enumerate(sprites):
            for x, row in enumerate(s):
                for y, pix in enumerate(row):
                    colors_across_sprites = sprites[:, x, y]
                    counter = Counter(colors_across_sprites)
                    if len(counter) == 1:
                        continue
                    clrs = counter.keys()
                    prob_space = np.sum(counter.values())
                    if s[x][y] in clrs:
                        l = s[x][y]
                        prob[j][x][y] = float(counter[l]) / float(prob_space)
        return {
            'prob': prob,
            'indices': indices,
            'given': given,
        }

    def most_probable_sprite(self, hashed, image, x, y):
        if len(hashed.shape) != 2:
            raise ValueError('A hashed hashed is required')
        cprint('{}: {}'.format((x, y), hashed[x][y]), 'magenta')

        patch = self.get_target_patch_from_image(image, x, y)
        patch_probs = np.zeros((4, 85))
        cprint(patch['runs'], 'green')

        for i, s in enumerate(self._patch_probabilites):
            temp = np.array([
                s['height'].get(patch['height'], 0),
                s['width'].get(patch['width'], 0),
                s['area'].get(patch['area'], 0),
                s['runs'].get(patch['runs'], 0),
            ])

            np.copyto(patch_probs[:, i], temp)

        int_probs = np.rint(np.multiply(patch_probs[3, :], 100))
        cprint(int_probs, 'cyan')
        prob_bins = [nz[0] for nz in np.nonzero(int_probs)]
        cprint(prob_bins, 'blue')
        print
        if len(prob_bins) == 1:
            # Check if the only non-zero element is equal to 100%
            sprite_index = prob_bins[0]
            if int_probs[prob_bins[0]] == 100:
                    # A perfect match has been found, return the following information
                    # Sprite index
                    # Sprite name
                    # Sprite location
                    # Sprite bounding box
                    sprite = self._sprites[sprite_index]
                    patch_info = [p for p in self._patches[sprite_index]
                        if p['runs'] == patch['runs']]
                    if len(patch_info) == 1:
                        patch_info = patch_info[0]
                    offset = (x - patch_info['offset'][0], y - patch_info['offset'][1])
                    bounding_box = (offset,
                        (offset[0] + sprite.shape[0], offset[1] + sprite.shape[1]))

                    return {
                        'index': sprite_index,
                        'name': self._names[sprite_index],
                        'location': offset,
                        'bounding_box': bounding_box,
                    }

        patch_probs = np.prod(patch_probs, 0)

        first_px = hashed[x][y]
        # given pixel (x, y)
        # what is the joint probability of (x, y) and (x - 1, y)
        if first_px not in self._palette_lookup.keys():
            raise ValueError('Color not recognized')
        pairs = [0] * 4

        if y < hashed.shape[1] - 1:
            pairs[0] = (hashed[x][y], hashed[x][y + 1])
        if x > 0:
            pairs[1] = (hashed[x - 1][y], hashed[x][y])
        if y > 0:
            pairs[2] = (hashed[x][y - 1], hashed[x][y])
        if x < hashed.shape[0] - 1:
            pairs[3] = (hashed[x + 1][y], hashed[x][y])

        probs = np.zeros((4, len(self._p_sprites)))
        inverted_pw_probs = {k: np.nan_to_num(np.subtract(1, v))
                             for k, v in self._pairwise_probabilities.items()}
        for i, p in enumerate(pairs):
            temp_x, temp_y = self._palette_lookup[p[0]], self._palette_lookup[p[1]]
            if i % 2 == 0:
                key = 'horz'
            else:
                key = 'vert'
            np.copyto(probs[i], inverted_pw_probs[key][:, temp_x, temp_y])
        probs = np.prod(probs, 0)

        return np.argmax(patch_probs)

    def get_target_patch_from_image(self, image, x, y):
        mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype='ubyte')
        cv2.floodFill(image, mask, (y, x), (255, 0, 0), flags=cv2.FLOODFILL_MASK_ONLY)
        trimmed_mask = mask[1:-1, 1:-1]
        bb = self.get_bounding_box(trimmed_mask)
        return {
            'color': image[x][y][:3],
            'height': bb[1][0] - bb[0][0],
            'width': bb[1][1] - bb[0][1],
            'area': len(np.nonzero(trimmed_mask)[0]),
            'runs': self.get_patch_runs(trimmed_mask),
        }

    def get_color_probability(self, color):
        return self._color_probabilities[:, self._palette_lookup[color]]

    def get_horizontal_probability(self, left, right):
        if left == 0 and right == 0:
            raise ValueError("Both pixels can't be transparent.")

        return self._pairwise_probabilities['horz'][:, left, right]

    def get_vertical_probability(self, top, bottom):
        if top == 0 and bottom == 0:
            raise ValueError("Both pixels can't be transparent.")

        return self._pairwise_probabilities['vert'][:, top, bottom]

    def get_patches(self):
        # TODO: The patch runs may include runs from other patches.
        # Possible fix: flag them using negative values
        all_patches = []
        for i, s in enumerate(self._sprites):
            mask_shape = (s.shape[0] + 2, s.shape[1] + 2)
            r, g, b, a = cv2.split(s)
            s_patches = []
            for x, y in zip(*np.nonzero(a)):
                if a[x][y] == 0:
                    continue
                mask = np.zeros(mask_shape, dtype='ubyte')
                temp = s[:, :, :3].astype('ubyte')
                cv2.floodFill(temp, mask, (y, x), (0, 255, 0), flags=cv2.FLOODFILL_MASK_ONLY)
                # Trim off the 1 px wide border that is required by cv2.floodFill
                trimmed_mask = mask[1:-1, 1:-1]

                # bounding box
                bb = np.nonzero(trimmed_mask)
                x1, y1 = min(bb[0]), min(bb[1])
                x2, y2 = max(bb[0]), max(bb[1])
                bb = self.get_bounding_box(trimmed_mask)

                a = np.logical_and(np.logical_not(trimmed_mask), a).astype('ubyte')
                s_patches.append({
                    'color': s[x][y][:3],
                    'height': x2 - x1,
                    'width': y2 - y1,
                    'bounding_box': bb,
                    'area': len(np.nonzero(trimmed_mask)[0]),
                    'runs': self.get_patch_runs(trimmed_mask),
                    'offset': bb[0],
                })
            all_patches.append(s_patches)
        return all_patches

    def get_bounding_box(self, trimmed_mask):
        bb = np.nonzero(trimmed_mask)
        x1, y1 = min(bb[0]), min(bb[1])
        x2, y2 = max(bb[0]), max(bb[1])
        return ((x1, y1), (x2, y2))

    def get_patch_runs(self, patch):
        return tuple(np.trim_zeros([len(np.nonzero(row)[0]) for row in patch]))


def main():
    sprites = {k: reduce_image(v) for k, v in get_sprites('sprites').items()}
    #sprites = {
    #    'small_jump': sprites['small_jump'],
    #    'big_run_1': sprites['big_run_1'],
    #    'big_run_3': sprites['big_run_3'],
    #}
    st = SpriteTree(sprites)

    probs = st.get_horizontal_probability(3, 3)
    named_probs = sorted(zip(st._names, probs), key=lambda a: a[1])

    for n, p in named_probs:
        print('{}: {}'.format(n, p))
    print

if __name__ == '__main__':
    main()
