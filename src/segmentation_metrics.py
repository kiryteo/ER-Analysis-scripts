import imageio
import numpy as np
import sknw
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import restoration

from PIL import Image


class SegmentationMetrics:

    def __init__(self):
        pass

    def process_erv2_output(self, erv2_output):
        erv2_output[erv2_output >= 85] = 255
        return erv2_output

    def process_analyzer_output(self, analyzer_op):
        data = analyzer_op[:,:,0]
        thr = threshold_otsu(data)
        bin_out = data > thr
        bin_out = bin_out.astype('uint8')*255

    def resize_analyzer_bin_op(self, analyzer_op):
        img = Image.fromarray(analyzer_op)
        img = img.resize((128, 128), Image.LANCZOS)
        return np.array(img)

    def process_analyzer_skel(self, analyzer_skel):
        op = analyzer_skel[:,:,0]
        mval = min(np.unique(op))
        op[np.where(op==mval)] = 0
        op[np.where(op!=mval) and np.where(op!=0)] = 255
        op = op/255


    def intersection_over_union(self, pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        return np.sum(intersection) / np.sum(union)

    def dice_coefficient(self, pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask)
        return (2. * np.sum(intersection)) / ((np.sum(pred_mask) + np.sum(true_mask)))

    