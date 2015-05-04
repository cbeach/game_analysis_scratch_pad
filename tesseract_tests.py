from glob import glob

from PIL import Image
from pytesseract import image_to_string

with open('analyzed/image_tests', 'wb') as fp:
    files = glob('data/*')
    text = []
    for i, f in enumerate(files):
        print('file {} of {}: {}% done'.format(i, len(files), (float(i) / float(len(files))) * 100))
        text = image_to_string(Image.open(f))
        fp.write(text)
        fp.write('\n==========\n')
