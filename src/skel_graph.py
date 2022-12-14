import imageio
import skan
import numpy as np
from skan import draw
from skan import skeleton_to_csgraph
import matplotlib.pyplot as plt


img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t000_ch00.tif')
img = (img - img.min()) / (img.max() - img.min())

skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')


pixel_graph, coords = skeleton_to_csgraph(skel)

fig, ax = plt.subplots()
draw.overlay_skeleton_networkx(pixel_graph, np.transpose(coords), image=skel, axis=ax)

plt.show()

# fig, ax = plt.subplots()
# draw.overlay_skeleton_2d(img, skel, dilate=0, axes=ax)

# plt.show()