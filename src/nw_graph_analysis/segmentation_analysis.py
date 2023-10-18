import imageio
from skimage.morphology import erosion

from segmentation_metrics import SegmentationMetrics
from graph_metrics import GraphMetrics

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import erosion
import contextlib

SMet = SegmentationMetrics()
GMet = GraphMetrics()

groups = ['climp', 'control', 'rtn']

prefix = '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/'

analyzer_prefix = {'control': 'ct', 'climp': 'c', 'rtn': 'r'}

def plot_subplot(subplot_num, data, title):
    plt.subplot(1, 7, subplot_num)
    plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.axis('off')

def save_comparison_plot(img, mask, analyzer, erv2, ernet, nerdy, nerdynet, filename):
    plt.figure(figsize=(20, 16))

    plot_subplot(1, img, 'Input')
    plot_subplot(2, mask, 'Mask')
    plot_subplot(3, analyzer, 'AnalyzER')
    plot_subplot(4, erv2, 'ERV2')
    plot_subplot(5, ernet, 'ERnet')
    plot_subplot(6, nerdy, 'nERdy')
    plot_subplot(7, nerdynet, 'nERdyNet')

    plt.subplots_adjust(wspace=0.02)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def qualitative_comparison():
    img = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/images/sted_climp8_er_mean.png')
    mask = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/masks/sted_climp8_er_mean_mask.png')
    analyzer = SMet.process_analyzer_output(imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/analyzer_op/ER_ce8.png'))
    ernet = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/ernet_op/sted_climp8_er_mean_out.png')
    erv2 = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/erv2_op/sted_climp8_er_mean_out_bin.png')
    nerdy = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/nerdy_op/Series008_decon_converted_mean_proc_v2_enhance.png')
    nerdynet = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/nerdynet_v2/sted_climp8_er_mean_pred.png')

    save_comparison_plot(img, mask, analyzer, erv2, ernet, nerdy, nerdynet, 'ER_segmentation_qualitative_comparison.png')





def load_all_data():
    gt_data = []
    analyzer_data = []
    erv2_data = []
    ernet_data = []
    nerdy_data = []
    nerdynet_data = []

    for group in groups:
        for num in range(1, 17):
            with contextlib.suppress(Exception):
                file = f'{prefix}{group}/masks/sted_{group}{num}_er_mean_mask.png'
                gt_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/analyzer_op/ER_{analyzer_prefix[group]}e{num}.png'
                file = f'{prefix}{group}/analyzer_op/{group}{num}_out.png'
                analyzer_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/erv2_op/sted_{group}{num}_er_mean_out_bin.png'
                erv2_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/ernet_op/sted_{group}{num}_er_mean_out.png'
                ernet_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/nerdy_op/Series0{num:02d}_decon_converted_mean_proc_v2_enhance.png'
                nerdy_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/nerdynet_op_seg/sted_{group}{num}_er_mean_pred.png'
                # file = f'{prefix}{group}/nerdynet-op/sted_{group}{num}_er_mean_pred.png' # BESTTTT

                file = f'{prefix}{group}/nerdynet_v2/sted_{group}{num}_er_mean_pred.png'
                nerdynet_data.append(imageio.imread(file))

    return gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data

# analyzer = []
# erv2 = []
# ernet = []
# nerdy = []
# nerdynet = []


# for pred, gt in zip(analyzer_data, gt_data):
#     pred = SMet.resize_analyzer_bin_op(pred)
#     pred = erosion(pred)
#     analyzer.append(SMet.intersection_over_union(pred, gt))


# for pred, gt in zip(erv2_data, gt_data):
#     pred = SMet.process_erv2_output(pred)
#     erv2.append(SMet.intersection_over_union(pred, gt))

# for pred, gt in zip(ernet_data, gt_data):
#     ernet.append(SMet.intersection_over_union(pred, gt))

# for pred, gt in zip(nerdy_data, gt_data):
#     nerdy.append(SMet.intersection_over_union(pred, gt))

# for pred, gt in zip(nerdynet_data, gt_data):
#     nerdynet.append(SMet.intersection_over_union(pred, gt))


# import numpy as np

# print(f'Analyzer: {np.mean(analyzer)}')
# print(f'ERV2: {np.mean(erv2)}')
# print(f'Ernet: {np.mean(ernet)}')
# print(f'Nerdy: {np.mean(nerdy)}')
# print(f'Nerdynet: {np.mean(nerdynet)}')


def get_graph_metrics(data_list):
    # sourcery skip: inline-immediately-returned-variable
    """
    Get graph metrics for each segmentation method
    metrics: #nodes, #edges, assortativity, clustering, #components
    """
    graphs = [GMet.seg_to_graph(seg/255.) for seg in data_list]
    graph_metrics = [GMet.simple_analysis(g) for g in graphs]
    return graph_metrics


# gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data = load_all_data()

def get_all_metrics():
    gt_graph_metrics = get_graph_metrics(gt_data)
    analyzer_graph_metrics = get_graph_metrics(analyzer_data)
    # erv2_graph_metrics = get_graph_metrics(erv2_data)
    # ernet_graph_metrics = get_graph_metrics(ernet_data)
    # nerdy_graph_metrics = get_graph_metrics(nerdy_data)
    # nerdynet_graph_metrics = get_graph_metrics(nerdynet_data)

    # return gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics

    return gt_graph_metrics, analyzer_graph_metrics

# gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics = get_all_metrics()


gt_graph_metrics, analyzer_graph_metrics = get_all_metrics()

print(gt_graph_metrics)
print(analyzer_graph_metrics)