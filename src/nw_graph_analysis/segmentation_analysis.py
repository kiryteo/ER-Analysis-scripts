import imageio
from skimage.morphology import erosion

from segmentation_metrics import SegmentationMetrics
from graph_metrics import GraphMetrics

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import erosion
import contextlib
import pickle

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
    # analyzer_data = []
    # erv2_data = []
    # ernet_data = []
    # nerdy_data = []
    # nerdynet_data = []
    # nerdy_adapt_data = []
    # eq_nerdy = []
    # eq_p4m = []

    nerdy150_data = []

    for group in groups:
        for num in range(1, 17):
            with contextlib.suppress(Exception):
                file = f'{prefix}{group}/updated_masks/sted_{group}{num}_er_mean_mask.png'
                # file = f'{prefix}{group}/adaptive_mask/{group}{num}_er_mean_mask.png'
                gt_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/analyzer_op/ER_{analyzer_prefix[group]}e{num}.png'
                
                # file = f'{prefix}{group}/analyzer_op/{group}{num}_out.png'
                # analyzer_data.append(erosion(imageio.imread(file)))

                # file = f'{prefix}{group}/erv2_op/sted_{group}{num}_er_mean_out_bin.png'
                # erv2_data.append(erosion(imageio.imread(file)))

                # file = f'{prefix}{group}/ernet_op/sted_{group}{num}_er_mean_out.png'
                # ernet_data.append(erosion(imageio.imread(file)))

                # file = f'{prefix}{group}/nerdy_op/Series0{num:02d}_decon_converted_mean_proc_v2_enhance.png'
                # nerdy_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/nerdynet_op_seg/sted_{group}{num}_er_mean_pred.png'
                # file = f'{prefix}{group}/nerdynet-op/sted_{group}{num}_er_mean_pred.png' # BESTTTT

                # file = f'{prefix}{group}/nerdynet_v2/sted_{group}{num}_er_mean_pred.png'
                # nerdynet_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/nerdy_adaptive_mask/sted_{group}{num}_er_mean_pred.png'
                # nerdy_adapt_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/equi-nnet/sted_{group}{num}_er_mean_pred.png'
                # eq_nerdy.append(imageio.imread(file))

                # file = f'{prefix}{group}/p4m_v2/sted_{group}{num}_er_mean_pred.png'
                # eq_p4m.append(imageio.imread(file))

                file = f'{prefix}{group}/nerdy_150/sted_{group}{num}_er_mean_pred.png'
                nerdy150_data.append(imageio.imread(file))



    # return gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, nerdy_adapt_data
    # return gt_data, nerdy_data, nerdynet_data, eq_nerdy
    # return gt_data, nerdynet_data, eq_nerdy, eq_p4m
    return gt_data, nerdy150_data


def compute_iou_metrics(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
        if process_pred_fn:
            pred = process_pred_fn(pred)
        metrics.append(SMet.intersection_over_union(pred, gt))

    return np.mean(metrics)

def compute_f1_score(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
        if process_pred_fn:
            pred = process_pred_fn(pred)
        metrics.append(SMet.f1_score(pred, gt))

    return np.mean(metrics)

def compute_dice_coefficient(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
        if process_pred_fn:
            pred = process_pred_fn(pred)
        metrics.append(SMet.dice_coefficient(pred, gt))

    return np.mean(metrics)

def compute_accuracy(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
        if process_pred_fn:
            pred = process_pred_fn(pred)
        metrics.append(SMet.accuracy(pred, gt))

    return np.mean(metrics)

def print_metric_results(metric_name, metric_value):
    print(f'{metric_name}: {metric_value}')

# def get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, nerdy_adapt_data, gt_data):

def get_segmentation_metrics(nerdy150_data, gt_data):
    # analyzer_iou = compute_iou_metrics(analyzer_data, gt_data, SMet.resize_analyzer_bin_op)
    # erv2_iou = compute_iou_metrics(erv2_data, gt_data, SMet.process_erv2_output)
    # ernet_iou = compute_iou_metrics(ernet_data, gt_data)
    # nerdy_iou = compute_iou_metrics(nerdy_data, gt_data)
    # nerdynet_iou = compute_iou_metrics(nerdynet_data, gt_data)
    # # nerdy_adapt_iou = compute_iou_metrics(nerdy_adapt_data, gt_data)
    # equi_nerdy_iou = compute_iou_metrics(equi_nerdy, gt_data)
    # eq_p4m_iou = compute_iou_metrics(eq_p4m, gt_data)

    nerdy150_iou = compute_iou_metrics(nerdy150_data, gt_data)

    # analyzer_f1 = compute_f1_score(analyzer_data, gt_data, SMet.resize_analyzer_bin_op)
    # erv2_f1 = compute_f1_score(erv2_data, gt_data, SMet.process_erv2_output)
    # ernet_f1 = compute_f1_score(ernet_data, gt_data)
    # nerdy_f1 = compute_f1_score(nerdy_data, gt_data)
    # nerdynet_f1 = compute_f1_score(nerdynet_data, gt_data)
    # # nerdy_adapt_f1 = compute_f1_score(nerdy_adapt_data, gt_data)
    # equi_nerdy_f1 = compute_f1_score(equi_nerdy, gt_data)
    # eq_p4m_f1 = compute_f1_score(eq_p4m, gt_data)

    nerdy150_f1 = compute_f1_score(nerdy150_data, gt_data)

    # analyzer_dice = compute_dice_coefficient(analyzer_data, gt_data, SMet.resize_analyzer_bin_op)
    # erv2_dice = compute_dice_coefficient(erv2_data, gt_data, SMet.process_erv2_output)
    # ernet_dice = compute_dice_coefficient(ernet_data, gt_data)
    # nerdy_dice = compute_dice_coefficient(nerdy_data, gt_data)
    # nerdynet_dice = compute_dice_coefficient(nerdynet_data, gt_data)
    # # nerdy_adapt_dice = compute_dice_coefficient(nerdy_adapt_data, gt_data)
    # equi_nerdy_dice = compute_dice_coefficient(equi_nerdy, gt_data)
    # eq_p4m_dice = compute_dice_coefficient(eq_p4m, gt_data)

    nerdy150_dice = compute_dice_coefficient(nerdy150_data, gt_data)

    # analyzer_acc = compute_accuracy(analyzer_data, gt_data, SMet.resize_analyzer_bin_op)
    # erv2_acc = compute_accuracy(erv2_data, gt_data, SMet.process_erv2_output)
    # ernet_acc = compute_accuracy(ernet_data, gt_data)
    # nerdy_acc = compute_accuracy(nerdy_data, gt_data)
    # nerdynet_acc = compute_accuracy(nerdynet_data, gt_data)
    # # nerdy_adapt_acc = compute_accuracy(nerdy_adapt_data, gt_data)
    # equi_nerdy_acc = compute_accuracy(equi_nerdy, gt_data)
    # eq_p4m_acc = compute_accuracy(eq_p4m, gt_data)

    nerdy150_acc = compute_accuracy(nerdy150_data, gt_data)

    # print_metric_results('Analyzer f1', analyzer_f1)
    # print_metric_results('ERnet', ernet_f1)
    # print_metric_results('ERnet_v2', erv2_f1)
    # print_metric_results('Nerdy', nerdy_f1)
    # print_metric_results('Nerdynet', nerdynet_f1)
    # # print_metric_results('Nerdy Adaptive f1', nerdy_adapt_f1)
    # print_metric_results('Equi-Nerdy f1', equi_nerdy_f1)
    # print_metric_results('P4M v1 f1', eq_p4m_f1)

    print_metric_results('Nerdy 150 f1', nerdy150_f1)

    # print_metric_results('Analyzer dice', analyzer_dice)
    # print_metric_results('ERnet', ernet_dice)
    # print_metric_results('ERnet_v2', erv2_dice)
    # print_metric_results('Nerdy', nerdy_dice)
    # print_metric_results('Nerdynet', nerdynet_dice)
    # # print_metric_results('Nerdy Adaptive dice', nerdy_adapt_dice)
    # print_metric_results('Equi-Nerdy dice', equi_nerdy_dice)
    # print_metric_results('P4M v1 dice', eq_p4m_dice)

    print_metric_results('Nerdy 150 dice', nerdy150_dice)

    # print_metric_results('Analyzer acc', analyzer_acc)
    # print_metric_results('ERnet', ernet_acc)
    # print_metric_results('ERnet_v2', erv2_acc)
    # print_metric_results('Nerdy', nerdy_acc)
    # print_metric_results('Nerdynet', nerdynet_acc)
    # print_metric_results('Nerdy Adaptive acc', nerdy_adapt_acc)
    # print_metric_results('Equi-Nerdy acc', equi_nerdy_acc)
    # print_metric_results('P4M v1 acc', eq_p4m_acc)

    print_metric_results('Nerdy 150 acc', nerdy150_acc)


    # print_metric_results('Analyzer iou', analyzer_iou)
    # print_metric_results('ERnet', ernet_iou)
    # print_metric_results('ERnet_v2', erv2_iou)
    # print_metric_results('Nerdy', nerdy_iou)
    # print_metric_results('Nerdynet', nerdynet_iou)
    # print_metric_results('Nerdy Adaptive iou', nerdy_adapt_iou)
    # print_metric_results('Equi-Nerdy iou', equi_nerdy_iou)
    # print_metric_results('P4M v1 iou', eq_p4m_iou)

    print_metric_results('Nerdy 150 iou', nerdy150_iou)

# gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, nerdy_adapt_data = load_all_data()

# gt_data, nerdy_data, nerdynet_data, eq_nerdy = load_all_data()

# gt_data, nerdynet_data, eq_nerdy, eq_p4m = load_all_data()

gt_data, nerdy150_data = load_all_data()

# Assuming the function is called with the required arguments
# get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, nerdy_adapt_data, gt_data)

# get_segmentation_metrics(nerdy_data, nerdynet_data, eq_nerdy, gt_data)

# get_segmentation_metrics(nerdynet_data, eq_nerdy, eq_p4m, gt_data)

# get_segmentation_metrics(nerdy150_data, gt_data)


# exit()

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
    gt_graph_metrics = get_graph_metrics(gt_data, None)
    # analyzer_graph_metrics = get_graph_metrics(analyzer_data, 'analyzer')
    # erv2_graph_metrics = get_graph_metrics(erv2_data, 'erv2')
    # ernet_graph_metrics = get_graph_metrics(ernet_data, 'ernet')
    # nerdy_graph_metrics = get_graph_metrics(nerdy_data, None)
    # nerdynet_graph_metrics = get_graph_metrics(nerdynet_data, None)
    # adaptive_metrics = get_graph_metrics(nerdy_adapt_data, None)
    # nerdy_p4m_v2_metrics = get_graph_metrics(eq_p4m, None)

    nerdy_150_metrics = get_graph_metrics(nerdy150_data, None)

    # return gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics, adaptive_metrics

    # return gt_graph_metrics, analyzer_graph_metrics
    return gt_graph_metrics, nerdy_150_metrics

# gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, nerdynet_graph_metrics, adaptive_metrics = get_all_metrics()

gt_graph_metrics, nerdy_150_metrics = get_all_metrics()

def get_mean(nparr):
    return [np.mean(lt) for lt in nparr]

gt_graph_metrics = get_all_metrics()

def get_error(gt, method):
    err = []
    for i in range(5):
        val = abs(gt[i] - method[i]) / abs(gt[i])
        err.append(val)
    return err

gt_graph_metrics = get_mean(np.array(gt_graph_metrics).T)
# analyzer_graph_metrics = get_mean(np.array(analyzer_graph_metrics).T)
# erv2_graph_metrics = get_mean(np.array(erv2_graph_metrics).T)
# ernet_graph_metrics = get_mean(np.array(ernet_graph_metrics).T)
# nerdy_graph_metrics = get_mean(np.array(nerdy_graph_metrics).T)
# nerdynet_graph_metrics = get_mean(np.array(nerdynet_graph_metrics).T)
# adaptive_metrics = get_mean(np.array(adaptive_metrics).T)

# nerdy_p4m_v2_metrics = get_mean(np.array(nerdy_p4m_v2_metrics).T)

nerdy_150_metrics = get_mean(np.array(nerdy_150_metrics).T)

# with open('sted_gt_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(gt_graph_metrics, f)

# with open('sted_analyzer_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(analyzer_graph_metrics, f)

# with open('sted_erv2_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(erv2_graph_metrics, f)

# with open('sted_ernet_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(ernet_graph_metrics, f)

# with open('sted_nerdy_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(nerdy_graph_metrics, f)

# with open('sted_nerdynet_graph_metrics.pkl', 'wb') as f:
#     pickle.dump(nerdynet_graph_metrics, f)

# with open('sted_adaptive_metrics.pkl', 'wb') as f:
    # pickle.dump(adaptive_metrics, f)

# with open('sted_nerdy_p4m_v2_metrics.pkl', 'wb') as f:
    # pickle.dump(nerdy_p4m_v2_metrics, f)


with open('sted_nerdy_150_metrics.pkl', 'wb') as f:
    pickle.dump(nerdy_150_metrics, f)

# analyzer_error = get_error(gt_graph_metrics, analyzer_graph_metrics)
# erv2_error = get_error(gt_graph_metrics, erv2_graph_metrics)
# ernet_error = get_error(gt_graph_metrics, ernet_graph_metrics)
# nerdy_error = get_error(gt_graph_metrics, nerdy_graph_metrics)
# nerdynet_error = get_error(gt_graph_metrics, nerdynet_graph_metrics)
# adaptive_error = get_error(gt_graph_metrics, adaptive_metrics)

# nerdy_p4m_v2_error = get_error(gt_graph_metrics, nerdy_p4m_v2_metrics)

nerdy150_error = get_error(gt_graph_metrics, nerdy_150_metrics)

# print(analyzer_error)
# print(erv2_error)
# print(ernet_error)
# print(nerdy_error)
# print(nerdynet_error)
# print(adaptive_error)

# print(nerdy_p4m_v2_error)

print(nerdy150_error)


# method = ['AnalyzER', 'ERnet', 'ERnet-v2', 'nERdy', 'nERdy+']
# data = [0.302, 0.524, 0.458, 0.687, 0.745]

# # Analyzer: 0.3027548722647748
# # ERnet: 0.5249320744721783
# # ERnet_v2: 0.4587372268867602
# # # Nerdy: 0.6877634174096008
# # Nerdy: 0.7454739562927578

# plt.bar(method, data)
# plt.xlabel('Segmentation Method', fontsize=14)
# plt.ylabel('mean IoU', fontsize=14)
# plt.title('Segmentation Performance', fontsize=16)
# plt.savefig('segmentation_method_vs_iou.png', bbox_inches='tight', pad_inches=0.1)
# # plt.show()
# plt.close()