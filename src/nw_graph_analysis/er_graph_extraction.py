# Pipeline to get the ER graph starting with skeleton

import imageio
from plantcv import plantcv as pcv
import sknw


def get_skel(self, img_path):
    """
    Extracts the skeleton from a vessel-enhanced image.

    @param img_path (str): The file path of the vessel-enhanced image.
    @return: extracted skeleton (numpy.ndarray)
    """
    vess_enhanced_sample = imageio.imread(img_path)
    return pcv.morphology.skeletonize(mask=vess_enhanced_sample)


def get_graph_from_skel(skel_img_path):
    """
    @param skel_img_path:
    @return:
    """
    return sknw.build_sknw(imageio.imread(skel_img_path), multi=True, iso=False)


