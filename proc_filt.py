import skimage
import imageio

img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc/A4/A4_decon_t002_ch00_proc.png')

# op = skimage.filters.threshold_local(img)
# op = skimage.filters.sobel(img)
# op = skimage.filters.hessian(img, (1,3,1))

# op = skimage.filters.frangi(img, (1,2,1))
# op = skimage.filters.threshold_local(img, 3, method='mean')

fd, op = skimage.feature.hog(img, visualize=True)

import matplotlib.pyplot as plt

plt.imshow(op)
plt.show()