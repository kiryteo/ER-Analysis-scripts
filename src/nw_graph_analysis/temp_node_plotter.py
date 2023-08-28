# import sknw
# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import copy



import skan
from skan import draw
import sknw
import imageio
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib.patches import Circle
import networkx as nx
from skimage import img_as_float, morphology
from skimage.color import gray2rgb


import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)



def _normalise_image(image, *, image_cmap=None):
    image = img_as_float(image)
    if image.ndim == 2:
        if image_cmap is None:
            image = gray2rgb(image)
        else:
            image = plt.get_cmap(image_cmap)(image)[..., :3]
    return image


def overlay_skeleton_2d(
        image,
        skeleton,
        junc,
        lab_img,
        *,
        image_cmap=None,
        color=(1, 0, 0),
        alpha=0.6,
        dilate=0,
        axes=None
        ):
    """Overlay the skeleton pixels on the input image.

    Parameters
    ----------
    image : array, shape (M, N[, 3])
        The input image. Can be grayscale or RGB.
    skeleton : array, shape (M, N)
        The input 1-pixel-wide skeleton.

    Other Parameters
    ----------------
    image_cmap : matplotlib colormap name or object, optional
        If the input image is grayscale, colormap it with this colormap.
        The default is grayscale.
    color : tuple of float in [0, 1], optional
        The RGB color for the skeleton pixels.
    alpha : float, optional
        Blend the skeleton pixels with the given alpha.
    dilate : int, optional
        Dilate the skeleton by this amount. This is useful when rendering
        large images where aliasing may cause some pixels of the skeleton
        not to be drawn.
    axes : matplotlib Axes
        The Axes on which to plot the image. If None, new ones are created.

    Returns
    -------
    axes : matplotlib Axes
        The Axis on which the image is drawn.
    """

    

    image = _normalise_image(image, image_cmap=image_cmap)
    skeleton = skeleton.astype(bool)
    junc = junc.astype(bool)
    if dilate > 0:
        selem = morphology.disk(dilate)
        skeleton = morphology.binary_dilation(skeleton, selem)
    if axes is None:
        fig, axes = plt.subplots()

    image[skeleton] = alpha * np.array(color) + (1-alpha) * image[skeleton]
    # image[junc] = alpha * np.array((1, 1, 1)) + (1-alpha) * image[junc]

    # vals = np.where(skeleton)
    # print(vals)
    # exit()

    # plt.plot(vals[1], vals[0], '.', markerfacecolor='red', markeredgecolor='red', mew=0.7, lw=0.7)

    # plt.plot(vals[1], vals[0], ',', markerfacecolor='red', markeredgecolor='red', lw=3.0, mew=3.0)

    junctions = np.where(junc)
    plt.plot(junctions[1], junctions[0], 'o', markerfacecolor='None', markeredgecolor='yellow', mew=0.7)

    cntrs = measure.find_contours(lab_img, 0.8, fully_connected='high')
    for cntr in cntrs:
        y, x = cntr.T
        plt.plot(x, y, color='cyan', mew=0.7, lw=0.7)

    axes.imshow(image)
    axes.axis('off')
    return axes


group_dict = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}

group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

def create_fuz_cc_overlay(group):

    for series in range(1, group_dict[group]+1):
        
        lab_img = get_fuz_cc_outline(group, series)

        for i in range(100):
            img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/{group_pref[group]}{series}_decon_t0{i:02d}_ch00_std.png')

            skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{series}/{group_pref[group]}{series}_decon_t0{i:02d}_ch00_skel.png')

            junc = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junctions/{group_pref[group]}{series}/{group_pref[group]}{series}_decon_t0{i:02d}_ch00_junc.png')

            ax = overlay_skeleton_2d(img, skel, junc,lab_img)
            
            plt.savefig(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/fuz_out_overlay/{group_pref[group]}{series}_decon_t0{i:02d}_ch00_skel_overlay.png', bbox_inches='tight', pad_inches=0, dpi=700)

            plt.close()


# create_fuz_cc_overlay('ATL')
# create_fuz_cc_overlay('Climp')
# create_fuz_cc_overlay('Control')
# create_fuz_cc_overlay('RTN')




# for fr in range(100):
#     skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{fr:02d}_ch00_skel.png')

#     graph = sknw.build_sknw(skel, multi=True, iso=False)


# skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')

# skel_coords = np.where(skel)

# junctions = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junctions/A1/A1_decon_t000_ch00_junc.png')


# fuz_skel_pixels = get_intersection(fuzzy_coords, skel_coords)

# print(fuz_skel_pixels)
# exit()





# data = get_skel_per_fuz_cc(lab_img)
# print(data)




graph = sknw.build_sknw(skel, multi=True, iso=False)

# node_set = graph.nodes

# node_coords = np.array([node_set[node]['o'] for node in node_set])

# print(node_coords)
# exit()

for i in range(1):
    junctions = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')

    nodes = np.where(junctions)

    skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{i:02d}_ch00_skel.png')

    graph = sknw.build_sknw(skel, multi=True, iso=False)

    degree_list = graph.degree

    fuz_nodes = get_intersection(fuzzy_coords, nodes)

    print(fuz_nodes)


lab_img = get_fuz_cc_outline('ATL', 1)
# lab_img = (lab_img>0).astype(np.uint8)

# Fuzzy CC mean intensity over time - already addressed

# data = []
# for cc_id in range(1, lab_img.max()+1):
#     cc_data = []
#     for fr in range(3):
#         er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t0{fr:02d}_ch00_std.png')
#         cc_data.append(er[lab_img==cc_id].mean())
#     data.append(cc_data)

# # data = np.array(data)
# print(len(data))
# print(data)




    