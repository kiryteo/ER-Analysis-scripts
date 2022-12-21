import os
import imageio
from PIL import Image

l = [f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_junc15_29_74/A1_decon_t0{i:02d}_ch01.png' for i in range(100)]


images = list(map(Image.open, l))
w, h = zip(*(i.size for i in images))

tw = sum(w)
mxh = max(h)

new_im = Image.new('RGB', (tw, mxh))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

new_im.save('A1_j15_sequence_mCherry.png')

