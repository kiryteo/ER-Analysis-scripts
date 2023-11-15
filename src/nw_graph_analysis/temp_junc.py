
from skimage.morphology import dilation
import sknw
import imageio
import numpy as np
import matplotlib.pyplot as plt

skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/atl/skel/A1_decon_t000_ch00_skel.png')
img = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/ATL/preproc/A1/A1_decon_t000_ch00_proc.png')

graph = sknw.build_sknw(skel, multi=False, iso=False)

node_set, degree_list = graph.nodes, graph.degree

node_coords = np.array([node_set[node]['o'] for node in node_set])

co = np.array([node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2])

d = np.zeros((128, 128))

plt.imshow(d, cmap='gray')
plt.plot(co[:, 1], co[:, 0], 'r.')

plt.axis('off')

plt.savefig('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/atl/junctions/A1_t0_junc.png', bbox_inches='tight', pad_inches=0)

plt.close()