import os
import imageio
import montages as mn
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array

# l = [imageio.imread( for i in range(100))]

# l = [f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area_v3/A1_decon_t0{i:02d}_ch01.png' for i in range(100)]


l = []
for i in range(100):
    img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junction_crops/S13_j10_cc_area_v2/R13_decon_t0{i:02d}_ch01.png')
    img = img[:,:,:3]
    l.append(img)


import numpy as np
img = np.concatenate(l, axis=1)

print(img.shape)

fig = plt.gca()

plt.axis('off')
plt.imshow(img, interpolation=None)

fig = plt.gcf()
fig.set_size_inches(24, 246, forward=True)

fig.savefig('R13_j10_mch.tif', bbox_inches='tight', pad_inches=0)
# plt.show()

exit()

# l = []
# for i in range(100):
#     # img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area_v2/A1_decon_t0{i:02d}_ch01.png')
#     img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area_v2/A1_decon_t0{i:02d}_ch01.png')
#     img = img[:,:,:3]
#     # l.append(img)
#     imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area_v3/A1_decon_t0{i:02d}_ch01.png', img)
#
# exit()
#
# l = array(l)
#
# print(l.shape)



#
# op = ln.reshape(369, 36900*3)
# # print(ln.shape)
# plt.imshow(op)
# plt.show()
# exit()

# images = mn.loadImages(folder='/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area/')
#
# # print(images)
# images.setMontage(column=100, row=1)
# montageArray = mn.makeMontage(images)
# mn.saveMontage(montageArray, fileName='output_file.tif')
#
# exit()


# i = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_cc_area/A1_decon_t000_ch00.png')
#
# plt.imshow(i)
# plt.show()
# exit()

images = list(map(Image.open, l))
# images = list(map(Image.fromarray, l))
w, h = zip(*(i.size for i in images))

tw = sum(w)
mxh = max(h)

new_im = Image.new('RGB', (tw, mxh))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]

new_im.save('A1_j15_sequence_mCherry.png')

