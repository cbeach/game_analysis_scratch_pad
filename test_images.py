import cv2
import numpy as np

from object_detection import reduce_image, show_image
from sprite_sheet_tools import get_sprites


class ImageFactory:
    def __init__(self, shape, num_sprites=1, stacking=False):
        pass


def main():
    img_factory = ImageFactory()
    show_image(img_factory.new_image())



if __name__ == '__main__':
    main()
