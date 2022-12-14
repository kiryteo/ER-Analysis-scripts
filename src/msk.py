import numpy as np
import imageio

l = ['01', '35', '65', '98']

home = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series013_decon_converted/'
for each in l:
    img = imageio.imread(f'{home}Series013_decon_converted_t{each}_ch00_std-annot.png')

    op = (img==255)
    imageio.imwrite(f'{home}Series013_t{each}.png', op.astype('int'))
