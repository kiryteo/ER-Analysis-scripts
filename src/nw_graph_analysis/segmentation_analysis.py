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


# def load_image(file_path):
#     with contextlib.suppress(Exception):
#         return imageio.imread(file_path)
#     return None

# def load_and_erode(file_path):
#     image = load_image(file_path)
#     return erosion(image) if image is not None else None

# def load_data(data_list, file_format, group, num, operation, erode=False):
#     file_path = f'{prefix}{group}/{operation}/{file_format.format(group=group, num=num)}'
#     data_list.append(load_and_erode(file_path))

# gt_data = []
# analyzer_data = []
# erv2_data = []
# ernet_data = []
# nerdy_data = []
# nerdynet_data = []

# for group in groups:
#     for num in range(1, 17):
#         load_data(gt_data, 'sted_{group}{num}_er_mean_mask.png', group, num, 'masks')
#         load_data(analyzer_data, '{group}{num}_out.png', group, num, 'analyzer_op')
#         load_data(erv2_data, 'sted_{group}{num}_er_mean_out_bin.png', group, num, 'erv2_op')
#         load_data(ernet_data, 'sted_{group}{num}_er_mean_out.png', group, num, 'ernet_op')
#         load_data(nerdy_data, 'Series0{num:02d}_decon_converted_mean_proc_v2_enhance.png', group, num, 'nerdy_op')
#         load_data(nerdynet_data, 'sted_{group}{num}_er_mean_pred.png', group, num, 'nerdynet_v2')


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


def compute_iou_metrics(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
        if process_pred_fn:
            pred = process_pred_fn(pred)
        metrics.append(SMet.intersection_over_union(pred, gt))

    return np.mean(metrics)

def print_metric_results(metric_name, metric_value):
    print(f'{metric_name}: {metric_value}')

def get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, gt_data):
    analyzer_iou = compute_iou_metrics(analyzer_data, gt_data, SMet.resize_analyzer_bin_op)
    erv2_iou = compute_iou_metrics(erv2_data, gt_data, SMet.process_erv2_output)
    ernet_iou = compute_iou_metrics(ernet_data, gt_data)
    nerdy_iou = compute_iou_metrics(nerdy_data, gt_data)
    nerdynet_iou = compute_iou_metrics(nerdynet_data, gt_data)

    print_metric_results('Analyzer', analyzer_iou)
    print_metric_results('ERnet', ernet_iou)
    print_metric_results('ERnet_v2', erv2_iou)
    print_metric_results('Nerdy', nerdy_iou)
    print_metric_results('Nerdynet', nerdynet_iou)

# gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data = load_all_data()

# Assuming the function is called with the required arguments
# get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, gt_data)


def get_graph_metrics(data_list, method):
    # sourcery skip: inline-immediately-returned-variable
    """
    Get graph metrics for each segmentation method
    metrics: #nodes, #edges, assortativity, clustering, #components
    """
    if method in ['analyzer', 'ernet', 'erv2']:
        graphs = [GMet.seg_to_graph(erosion(seg/255.)) for seg in data_list]
    else:
        graphs = [GMet.seg_to_graph(seg/255.) for seg in data_list]
    graph_metrics = [GMet.simple_analysis(g) for g in graphs]
    return graph_metrics


def get_all_metrics():
    gt_graph_metrics = get_graph_metrics(gt_data)
    analyzer_graph_metrics = get_graph_metrics(analyzer_data)
    erv2_graph_metrics = get_graph_metrics(erv2_data)
    ernet_graph_metrics = get_graph_metrics(ernet_data)
    nerdy_graph_metrics = get_graph_metrics(nerdy_data)
    nerdynet_graph_metrics = get_graph_metrics(nerdynet_data)

    return gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics

    # return gt_graph_metrics, analyzer_graph_metrics

# gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics = get_all_metrics()




# print(gt_graph_metrics)
# print(analyzer_graph_metrics)

method = ['AnalyzER', 'ERnet', 'ERnet-v2', 'nERdy', 'nERdy+']
data = [0.302, 0.524, 0.458, 0.687, 0.745]

# Analyzer: 0.3027548722647748
# ERnet: 0.5249320744721783
# ERnet_v2: 0.4587372268867602
# # Nerdy: 0.6877634174096008
# Nerdy: 0.7454739562927578

plt.bar(method, data)
plt.xlabel('Segmentation Method', fontsize=14)
plt.ylabel('mean IoU', fontsize=14)
plt.title('Segmentation Performance', fontsize=16)
plt.savefig('segmentation_method_vs_iou.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()
plt.close()