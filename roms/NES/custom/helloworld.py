import pynes

from pynes.bitbag import *

if __name__ == "__main__":
    pynes.press_start()
    exit()

palette = [
    0x22,0x29, 0x1A,0x0F, 0x22,0x36,0x17,0x0F,  0x22,0x30,0x21,0x0F,  0x22,0x27,0x17,0x0F,
    0x22,0x16,0x27,0x18,  0x22,0x1A,0x30,0x27,  0x22,0x16,0x30,0x27,  0x22,0x0F,0x36,0x17]

chr_asset = import_chr('mario.chr')

helloworld = "Hello World"

def reset():
    wait_vblank()
    clearmem()
    wait_vblank()
    load_palette(palette)
    show(helloworld, 15, 10)
