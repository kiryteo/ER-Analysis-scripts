import vigra
import skimage.io as io
import matplotlib.pyplot as plt

img = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/enh/Series002_decon_converted_t00_ch00_std_enhance.png')/255
# plt.imshow(img)
# plt.show()

img = img.astype('uint32')

cc = vigra.analysis.labelImageWithBackground(img)
skel = vigra.filters.skeletonizeImage(cc, 'DontPrune')

plt.imshow(skel)
plt.show()