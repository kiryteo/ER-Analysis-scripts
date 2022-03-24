import numpy as np
import skimage.io as io
from scipy.signal import correlate
import matplotlib.pyplot as plt

imdir = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/'

l = []

for i in range(85):
    im1 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{i:02d}')
    im2 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{i+15:02d}')

    cm = np.corrcoef(im1.flat, im2.flat)
    l.append(cm[0, 1])

plt.hist(l)
plt.show()

# im1 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t04_ch00_std.png')
# im2 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t05_ch00_std.png')
#
# cm = np.corrcoef(im1.flat, im2.flat)
# sc = correlate(im1.flat, im2.flat)
#
# r = cm[0, 1]
#
# print(cm)
# print(len(sc))