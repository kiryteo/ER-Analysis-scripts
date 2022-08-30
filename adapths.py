import skimage
from skimage import exposure
import imageio
import matplotlib.pyplot as plt



for i in range(100):
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/A1_decon_t0%s_ch01_std_std_adj.png'%f'{i:02d}')
    op = exposure.equalize_adapthist(img, clip_limit=0.05)

    imageio.imwrite('/localhome/asa420/MIAL/data/confocal_movies/ATL/adapthist/' + 'A1_decon_t0%s_ch01_std_std_adj_adapthist.png'%f'{i:02d}', op)


# plt.imshow(op)
# plt.show()