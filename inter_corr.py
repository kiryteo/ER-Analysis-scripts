import numpy as np
import skimage.io as io
from scipy.signal import correlate
import matplotlib.pyplot as plt

imdir = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/'
#
# l = []
#
# im1 = io.imread(imdir + 'Series002_decon_converted_t05_ch00_std.png')
# im2 = io.imread(imdir + 'Series002_decon_converted_t08_ch00_std.png')
#
# print(np.corrcoef(im1.flat, im2.flat)[0, 1])
#
# # num1 = im1.flat - np.mean(im1.flat)
# # num2 = im2.flat - np.mean(im2.flat)
# #
# # print(np.mean(im1.flat))
# # print(num1)
#
# exit()


l = []
for i in range(2,3):
    im1 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{i:02d}')
    for j in range(i+1, 100):
        im2 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{j:02d}')
        cm = np.corrcoef(im1.flat, im2.flat)
        l.append(cm[0, 1])
        # print(j, cm[0, 1])

    #l.append(cm[0, 1])

#plt.hist(l)
plt.plot(l)
plt.show()


exit()

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