import imageio
import numpy as np
import sknw
import skimage
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
        data = data[52:1670, 52:1670]
        data = skimage.transform.resize(data, (128, 128), anti_aliasing=True)
        thr = threshold_otsu(data)
        bin_out = data > thr
        bin_out = bin_out.astype('uint8')*255
        return bin_out

    def resize_analyzer_bin_op(self, analyzer_op):
        # bin_img = self.process_analyzer_output(analyzer_op)
        # img = Image.fromarray(analyzer_op)
        # img = img.resize((128, 128), Image.LANCZOS)
        # return np.array(img)
        img = skimage.transform.resize(analyzer_op, (128, 128), anti_aliasing=True)
        return img

    def intersection_over_union(self, pred_mask, true_mask):
        if pred_mask.max() == 255:
            pred_mask = pred_mask / 255
        if true_mask.max() == 255:
            true_mask = true_mask / 255
        assert pred_mask.shape == true_mask.shape
        # assert pred_mask.max() == true_mask.max() == 1
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        return np.sum(intersection) / np.sum(union)

    def dice_coefficient(self, pred_mask, true_mask):
        if pred_mask.max() == 255:
            pred_mask = pred_mask / 255
        if true_mask.max() == 255:
            true_mask = true_mask / 255
        assert pred_mask.shape == true_mask.shape
        assert pred_mask.max() == true_mask.max() == 1
        intersection = np.logical_and(pred_mask, true_mask)
        return (2. * np.sum(intersection)) / ((np.sum(pred_mask) + np.sum(true_mask)))

    def f1_score(self, pred_mask, true_mask):
        if pred_mask.max() == 255:
            pred_mask = pred_mask / 255
        if true_mask.max() == 255:
            true_mask = true_mask / 255
        assert pred_mask.shape == true_mask.shape
        assert pred_mask.max() == true_mask.max() == 1
        intersection = np.logical_and(pred_mask, true_mask)
        precision = np.sum(intersection) / np.sum(pred_mask)
        recall = np.sum(intersection) / np.sum(true_mask)
        return 2 * ((precision * recall) / (precision + recall))
    
    def accuracy(self, pred_mask, true_mask):
        if pred_mask.max() == 255:
            pred_mask = pred_mask / 255
        if true_mask.max() == 255:
            true_mask = true_mask / 255
        assert pred_mask.shape == true_mask.shape
        assert pred_mask.max() == true_mask.max() == 1
        return np.sum(pred_mask == true_mask) / np.prod(pred_mask.shape)
    
    def jaccard_edge_similarity(g1, g2):
        jaccard_edge_similarity = []
        edge_set1 = set(g1.edges)
        edge_set2 = set(g2.edges)
        jaccard_index = len(edge_set1.intersection(edge_set2)) / len(edge_set1.union(edge_set2))
        jaccard_edge_similarity.append(jaccard_index)

        return jaccard_edge_similarity

    def jaccard_similarity(graph1, graph2):
        nodes1 = set(graph1.nodes)
        nodes2 = set(graph2.nodes)
        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        return intersection / union