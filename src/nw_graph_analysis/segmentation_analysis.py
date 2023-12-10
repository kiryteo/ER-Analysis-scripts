import imageio
from skimage.morphology import erosion

from segmentation_metrics import SegmentationMetrics
from graph_metrics import GraphMetrics

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, threshold_mean
from skimage.morphology import erosion
from skimage import restoration
import copy
from PIL import Image
import contextlib
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import torchmetrics

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

SMet = SegmentationMetrics()
GMet = GraphMetrics()

groups = ['control', 'rtn', 'climp']

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



class NNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NNet, self).__init__()
        

        self.conv1 = P4MConvZ2(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = P4MConvP4M(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = P4MConvP4M(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = P4MConvP4M(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv5 = P4MConvP4M(128, 64, kernel_size=3, padding=1)
        self.conv6 = P4MConvP4M(64, 32, kernel_size=3, padding=1)

        self.convt = nn.ConvTranspose2d(32*8, out_channels, kernel_size=2, stride=2, output_padding=0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        xs = x.size()
        x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
#         x = F.avg_pool2d(x, 2, 2)
        x = self.convt(x)

        return x


def load_models():
    p4m = NNet(1, 1)
    p4m.load_state_dict(torch.load('nw_graph_analysis/NNet_groupy_p4m_v2_VecAdam.pth'))

    # p4m_adaptive = NNet(1, 1)
    # p4m_adaptive.load_state_dict(torch.load('NNet_groupy_p4m_v2_VecAdam_adaptive.pth'))

    # return p4m, p4m_adaptive
    return p4m

def process_op(imgpath, model):
    image = Image.open(imgpath)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Convert the output to probabilities by applying the sigmoid activation
    output_probs = torch.sigmoid(output)

    # Convert tensors to numpy arrays for visualization
    output_probs_np = output_probs.cpu().squeeze().numpy()

    norm = (output_probs_np - output_probs_np.min()) / (output_probs_np.max() - output_probs_np.min())

    rb = restoration.rolling_ball(norm)
    subt = norm - rb

    # thr = threshold_otsu(subt) # for STED data
    thr = threshold_mean(subt) # in case of confocal data

    subt2 = copy.deepcopy(subt)

    subt2[subt < thr] = 0.
    subt2[subt >= thr] = 255.

    return subt2

# from skimage.morphology import skeletonize
# import skimage.io as io

# p4m = load_models()

# imgpath = '/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean/climp12_er_mean.png'
# op = process_op(imgpath, p4m) / 255
# io.imsave('climp12_er_mean_pred_p4m.png', op)
# skel = (skeletonize(op) * 255).astype('uint8')
# io.imsave('climp12_er_mean_pred_skel_p4m.png', skel)

# exit()

# file = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/images/sted_rtn16_er_mean.png')

# gt_mask = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/updated_masks/sted_rtn16_er_mean_mask.png')

# analyzer_op = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/analyzer_op/resized_op/rtn16_out.png')

# ernet_op = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/ernet_op/sted_rtn16_er_mean_out.png')

# erv2_op = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/erv2_op/sted_rtn16_er_mean_out_bin.png')

# nerdy_op = imageio.imread('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/rtn/nerdy_op/Series016_decon_converted_mean_proc_v2_enhance.png')

# p4m = load_models()
# nerdy_plus_op = process_op('/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/control/images/sted_control8_er_mean.png', p4m)

# imageio.imsave('STED_control8_er_mean_pred_nerdy+.png', nerdy_plus_op)


# exit()

# plt.subplot(1, 7, 1)
# plt.axis('off')
# plt.imshow(file, cmap='gray')
# # plt.title('Input')

# plt.subplot(1, 7, 2)
# plt.axis('off')
# plt.imshow(gt_mask, cmap='gray')
# # plt.title('Mask')

# plt.subplot(1, 7, 3)
# plt.axis('off')
# plt.imshow(analyzer_op, cmap='gray')
# # plt.title('AnalyzER')

# plt.subplot(1, 7, 4)
# plt.axis('off')
# plt.imshow(ernet_op, cmap='gray')
# # plt.title('ERnet')

# plt.subplot(1, 7, 5)
# plt.axis('off')
# plt.imshow(erv2_op, cmap='gray')
# # plt.title('ERnet-v2')

# plt.subplot(1, 7, 6)
# plt.axis('off')
# plt.imshow(nerdy_op, cmap='gray')
# # plt.title('nERdy')

# plt.subplot(1, 7, 7)
# plt.axis('off')
# plt.imshow(nerdy_plus_op, cmap='gray')
# # plt.title('nERdy+')

# plt.subplots_adjust(wspace=0.02)

# plt.savefig('rtn16_segmentation_comparison.png', bbox_inches='tight', pad_inches=0)

# # plt.show()
# plt.close()

# exit()
# get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data, gt_data)

def load_all_data():

    p4m = load_models()

    gt_data = []
    analyzer_data = []
    erv2_data = []
    ernet_data = []
    nerdy_data = []


    p4m_vecadam_data = []
    # p4m_vecadam_adaptive_data = []

    for group in groups:
        for num in range(1, 17):
            with contextlib.suppress(Exception):
                file = f'{prefix}{group}/updated_masks/sted_{group}{num}_er_mean_mask.png'
                # file = f'{prefix}{group}/blur_masks/sted_{group}{num}_er_mean_blur_mask.png'

                # file = f'{prefix}{group}/adaptive_mask/{group}{num}_er_mean_mask.png'
                gt_data.append(imageio.imread(file))

                # file = f'{prefix}{group}/analyzer_op/ER_{analyzer_prefix[group]}e{num}.png'
                
                file = f'{prefix}{group}/analyzer_op/resized_op/{group}{num}_out.png'
                # analyzer_data.append(SMet.resize_analyzer_bin_op(imageio.imread(file)))
                
                analyzer_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/erv2_op/sted_{group}{num}_er_mean_out_bin.png'
                erv2_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/ernet_op/sted_{group}{num}_er_mean_out.png'
                ernet_data.append(erosion(imageio.imread(file)))

                file = f'{prefix}{group}/nerdy_op/Series0{num:02d}_decon_converted_mean_proc_v2_enhance.png'
                nerdy_data.append(imageio.imread(file))

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

                # file = f'{prefix}{group}/nerdy_150/sted_{group}{num}_er_mean_pred.png'
                # nerdy150_data.append(imageio.imread(file))

                # p4m_adaptive = f'{prefix}{group}/p4m_adaptive/sted_{group}{num}_er_mean_pred.png'
                # p4m_adaptive = f'{prefix}{group}/p4m_v2/sted_{group}{num}_er_mean_pred.png'
                # p4m_v2_blur = f'{prefix}{group}/blur_p4m_v2/sted_{group}{num}_er_mean_pred.png'
                
                # p4m_adaptive_data.append(imageio.imread(p4m_adaptive))
                # p4m_v2_blur_data.append(imageio.imread(p4m_v2_blur))

                # file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/er_mean_blur/{group}{num}_er_mean_blur.png'

                file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/images/sted_{group}{num}_er_mean.png'

                # print(process_op(file, p4m))

                p4m_vecadam_data.append(process_op(file, p4m))


            # p4m_vecadam_adaptive_data.append(process_op(file, p4m_adaptive))


    # return gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, nerdynet_data, nerdy_adapt_data
    # return gt_data, nerdy_data, nerdynet_data, eq_nerdy
    # return gt_data, nerdynet_data, eq_nerdy, eq_p4m
    # return gt_data, p4m_adaptive_data
    # return gt_data, p4m_v2_blur_data
    # return gt_data, p4m_vecadam_data, p4m_vecadam_adaptive_data

    return gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data
    # return gt_data, p4m_vecadam_data


def compute_iou_metrics(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred, gt in zip(pred_data, gt_data):
    #     # if process_pred_fn:
    #     #     pred = process_pred_fn(pred)
    #     # metrics.append(SMet.intersection_over_union(pred, gt))
        if pred.max() == 255:
            pred = pred / 255
        if gt.max() == 255:
            gt = gt / 255
        metrics.append(torchmetrics.classification.BinaryJaccardIndex()(torch.tensor(pred), torch.tensor(gt)).item())

    return metrics
    # return np.mean(metrics), np.std(metrics)
    # pred_data = np.array(pred_data)
    # gt_data = np.array(gt_data)

    # assert pred_data.shape == gt_data.shape
    # assert pred_data.max() == gt_data.max()

    # return torchmetrics.classification.BinaryJaccardIndex()(torch.tensor(pred_data/255.), torch.tensor(gt_data/255.))


from sklearn.metrics import f1_score

def compute_f1_score(pred_data, gt_data):
    metrics = []

    for pred_mask, true_mask in zip(pred_data, gt_data):
        # pred = pred.flatten()/255
        # gt = gt.flatten()/255
        # metrics.append(f1_score(gt, pred))
        if pred_mask.max() == 255:
            pred_mask = pred_mask / 255
        if true_mask.max() == 255:
            true_mask = true_mask / 255
        # metrics.append(SMet.f1_score(pred, gt))
        metrics.append(torchmetrics.classification.BinaryF1Score()(torch.tensor(pred_mask), torch.tensor(true_mask)).item())

    return metrics
    # return np.mean(metrics), np.std(metrics)


# def compute_f1_score(pred_data, gt_data, process_pred_fn=None):
    # metrics = []

    # for pred, gt in zip(pred_data, gt_data):
    #     if process_pred_fn:
    #         pred = process_pred_fn(pred)
    #     metrics.append(SMet.f1_score(pred, gt))

    # return np.mean(metrics)

def compute_dice_coefficient(pred_data, gt_data, process_pred_fn=None):
    metrics = []

    for pred_mask, true_mask in zip(pred_data, gt_data):
        # if process_pred_fn:
        #     pred = process_pred_fn(pred)

        if pred_mask.max() == 255.:
            pred_mask = pred_mask / 255.
        if true_mask.max() == 255.:
            true_mask = true_mask / 255.
        # metrics.append(SMet.dice_coefficient(pred, gt))
        metrics.append(torchmetrics.classification.Dice()(torch.tensor(pred_mask.astype('int')), torch.tensor(true_mask.astype('int'))).item())

    return metrics
    # return np.mean(metrics), np.std(metrics)

# def compute_accuracy(pred_data, gt_data, process_pred_fn=None):
#     metrics = []

#     for pred, gt in zip(pred_data, gt_data):
#         # if process_pred_fn:
#         #     pred = process_pred_fn(pred)
#         if pred.max() == 255:
#             pred = pred / 255
#         if gt.max() == 255:
#             gt = gt / 255
#         metrics.append(SMet.accuracy(pred, gt))

#     return np.mean(metrics)

def print_metric_results(metric_name, metric_value):
    print(f'{metric_name}: {metric_value}')

def get_segmentation_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data, gt_data):
# def get_segmentation_metrics(p4m_vecadam_data, gt_data):
    analyzer_iou, analyzer_std = compute_iou_metrics(analyzer_data, gt_data)
    erv2_iou, erv2_std = compute_iou_metrics(erv2_data, gt_data)
    ernet_iou, ernet_std = compute_iou_metrics(ernet_data, gt_data)
    nerdy_iou, nerdy_std = compute_iou_metrics(nerdy_data, gt_data)
    p4m_vecadam_iou, p4m_std = compute_iou_metrics(p4m_vecadam_data, gt_data)

    analyzer_f1, an_f1_std = compute_f1_score(analyzer_data, gt_data)
    erv2_f1, erv2_f1_std = compute_f1_score(erv2_data, gt_data)
    ernet_f1, ernet_f1_std = compute_f1_score(ernet_data, gt_data)
    nerdy_f1, nerdy_f1_std = compute_f1_score(nerdy_data, gt_data)
    p4m_vecadam_f1, p4m_f1_std = compute_f1_score(p4m_vecadam_data, gt_data)

    analyzer_dice, an_dice_std = compute_dice_coefficient(analyzer_data, gt_data)
    erv2_dice, erv2_dice_std = compute_dice_coefficient(erv2_data, gt_data)
    ernet_dice, ernet_dice_std = compute_dice_coefficient(ernet_data, gt_data)
    nerdy_dice, nerdy_dice_std = compute_dice_coefficient(nerdy_data, gt_data)
    p4m_vecadam_dice, p4m_dice_std = compute_dice_coefficient(p4m_vecadam_data, gt_data)


    print_metric_results('Analyzer f1', analyzer_f1)
    print(f'Analyzer f1 std: {an_f1_std}')
    print_metric_results('ERnet', ernet_f1)
    print(f'ERnet f1 std: {ernet_f1_std}')
    print_metric_results('ERnet_v2', erv2_f1)
    print(f'ERnet_v2 f1 std: {erv2_f1_std}')
    print_metric_results('Nerdy', nerdy_f1)
    print(f'Nerdy f1 std: {nerdy_f1_std}')
    print_metric_results('P4M v2 VecAdam f1',
     p4m_vecadam_f1)
    print(f'P4M v2 VecAdam f1 std: {p4m_f1_std}')

    print_metric_results('Analyzer dice', analyzer_dice)
    print(f'Analyzer dice std: {an_dice_std}')
    print_metric_results('ERnet', ernet_dice)
    print(f'ERnet dice std: {ernet_dice_std}')
    print_metric_results('ERnet_v2', erv2_dice)
    print(f'ERnet_v2 dice std: {erv2_dice_std}')
    print_metric_results('Nerdy', nerdy_dice)
    print(f'Nerdy dice std: {nerdy_dice_std}')
    print_metric_results('P4M v2 VecAdam dice', p4m_vecadam_dice)
    print(f'P4M v2 VecAdam dice std: {p4m_dice_std}')

    print_metric_results('Analyzer iou', analyzer_iou)
    print(f'Analyzer iou std: {analyzer_std}')
    print_metric_results('ERnet', ernet_iou)
    print(f'ERnet iou std: {ernet_std}')
    print_metric_results('ERnet_v2', erv2_iou)
    print(f'ERnet_v2 iou std: {erv2_std}')
    print_metric_results('Nerdy', nerdy_iou)
    print(f'Nerdy iou std: {nerdy_std}')
    print_metric_results('P4M v2 VecAdam iou', p4m_vecadam_iou)
    print(f'P4M v2 VecAdam iou std: {p4m_std}')


def get_seg_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data, gt_data):
# def get_segmentation_metrics(p4m_vecadam_data, gt_data):
    analyzer_iou = compute_iou_metrics(analyzer_data, gt_data)
    erv2_iou = compute_iou_metrics(erv2_data, gt_data)
    ernet_iou = compute_iou_metrics(ernet_data, gt_data)
    nerdy_iou = compute_iou_metrics(nerdy_data, gt_data)
    p4m_vecadam_iou = compute_iou_metrics(p4m_vecadam_data, gt_data)

    analyzer_f1 = compute_f1_score(analyzer_data, gt_data)
    erv2_f1 = compute_f1_score(erv2_data, gt_data)
    ernet_f1 = compute_f1_score(ernet_data, gt_data)
    nerdy_f1 = compute_f1_score(nerdy_data, gt_data)
    p4m_vecadam_f1 = compute_f1_score(p4m_vecadam_data, gt_data)

    analyzer_dice = compute_dice_coefficient(analyzer_data, gt_data)
    erv2_dice = compute_dice_coefficient(erv2_data, gt_data)
    ernet_dice = compute_dice_coefficient(ernet_data, gt_data)
    nerdy_dice = compute_dice_coefficient(nerdy_data, gt_data)
    p4m_vecadam_dice = compute_dice_coefficient(p4m_vecadam_data, gt_data)

    return analyzer_iou, ernet_iou, erv2_iou, nerdy_iou, p4m_vecadam_iou, analyzer_f1, ernet_f1, erv2_f1, nerdy_f1, p4m_vecadam_f1, analyzer_dice, ernet_dice, erv2_dice, nerdy_dice, p4m_vecadam_dice


# gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data = load_all_data()

# analyzer_iou, ernet_iou, erv2_iou, nerdy_iou, p4m_vecadam_iou, analyzer_f1, ernet_f1, erv2_f1, nerdy_f1, p4m_vecadam_f1, analyzer_dice, ernet_dice, erv2_dice, nerdy_dice, p4m_vecadam_dice = get_seg_metrics(analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data, gt_data)

# with open('sted_f1.pkl', 'wb') as f:
#     pickle.dump([analyzer_f1, ernet_f1, erv2_f1, nerdy_f1, p4m_vecadam_f1], f)

# with open('sted_dice.pkl', 'wb') as f:
#     pickle.dump([analyzer_dice, ernet_dice, erv2_dice, nerdy_dice, p4m_vecadam_dice], f)

# with open('sted_iou.pkl', 'wb') as f:
#     pickle.dump([analyzer_iou, ernet_iou, erv2_iou, nerdy_iou, p4m_vecadam_iou], f)

# with open('sted_rtn_f1.pkl', 'wb') as f:
#     pickle.dump([analyzer_f1, erv2_f1, ernet_f1, nerdy_f1, p4m_vecadam_f1], f)

# with open('sted_rtn_dice.pkl', 'wb') as f:
#     pickle.dump([analyzer_dice, erv2_dice, ernet_dice, nerdy_dice, p4m_vecadam_dice], f)

# with open('sted_rtn_iou.pkl', 'wb') as f:
#     pickle.dump([analyzer_iou, erv2_iou, ernet_iou, nerdy_iou, p4m_vecadam_iou], f)


# gt_data, p4m_data = load_all_data()
# get_segmentation_metrics(p4m_data, gt_data)

# exit()


# gt_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/climp/gt_skel/climp1_proc_skel.png')

# # skel to graph
# gt_graph = GMet.get_graph(gt_skel)

# analyzer_mask = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/analyzer_masks/climp/ER_climp1.png')
# # get resized skeleton
# analyzer_seg = SMet.process_analyzer_output(analyzer_mask)
# # skel to graph
# analyzer_graph = GMet.seg_to_graph(analyzer_seg/255.)


# ernet_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/ernet_masks/climp/climp1_er_mean_out.png')
# # ernet_skel = GMet.seg_to_graph(erosion(ernet_skel/255.))
# ernet_graph = GMet.seg_to_graph(ernet_skel/255.)

# erv2_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/erv2_new_masks/climp/climp1_er_mean_out_0000.png')
# # erv2_skel = GMet.seg_to_graph(erosion(erv2_skel/255.))
# erv2_graph = GMet.seg_to_graph(erv2_skel/255.)

# nerdy_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean_proc/climp1_er_mean_proc_enhance.png')
# nerdy_graph = GMet.seg_to_graph(nerdy_skel/255.)

# p4m_vecadam_file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/p4m_vecadam_op/sted_{group}{seq}_er_mean_pred.png'

# input_file = '/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean/climp1_er_mean.png'

# p4m = load_models()
# p4m_vecadam_seg = process_op(input_file, p4m)

# p4m_vecadam_graph = GMet.seg_to_graph(p4m_vecadam_seg/255.)


def plot_graph(graph, input_file):
    plt.imshow(imageio.imread(input_file), cmap='gray')

    degree_list = graph.degree

    tgraph = copy.deepcopy(graph)

    # for i, val in enumerate(degree_list):
    #     print(val)
    #     if val[1] < 3:
    #         tgraph.remove_node(i)


    node_set = tgraph.nodes

    degree_list = tgraph.degree
    node_coords = np.array([node_set[node]['o'] for node in node_set])
    nps = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]
    nps = np.array(nps)

    for (s,e) in graph.edges():
        ps = graph[s][e][0]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

    plt.plot(nps[:,1], nps[:,0], '.', markerfacecolor='red', markeredgecolor='red', mew=1)

    # plt.show()
    plt.axis('off')
    # plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/er_mean/rtn4_overlay.png', bbox_inches='tight', pad_inches=0.0)
    plt.savefig('climp1_analyzer_graph_overlay.png', bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


# plot_graph(analyzer_graph, input_file)
# exit()

import networkx as nx

def get_graphs():

    p4m = load_models()

    groups = ['climp', 'control', 'rtn']

    sym = {'climp':'c', 'control':'ct', 'rtn':'r'}

    gt_graphs = []
    analyzer_graphs = []
    erv2_graphs = []
    ernet_graphs = []
    nerdy_graphs = []
    p4m_vecadam_graphs = []

    for group in groups:
        for seq in range(1, 17):
            with contextlib.suppress(Exception):
                gt_graph = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/gt_skel/sted_{group}{seq}_proc_skel.png')

                # skel to graph
                gt_graph = GMet.get_graph(gt_graph)
                gt_graphs.append(GMet.simple_analysis(gt_graph))
                # gt_graphs.append(gt_graph)

                analyzer_skel = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/analyzer_skel/ER_{sym[group]}{seq}.png')
                # get resized skeleton
                analyzer_skel = GMet.process_analyzer_skel(analyzer_skel)

                # skel to graph
                analyzer_skel = GMet.get_graph(analyzer_skel)
                # analyzer_graphs.append(analyzer_skel)
                analyzer_graphs.append(GMet.simple_analysis(analyzer_skel))

                ernet_skel = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/ernet_op/sted_{group}{seq}_er_mean_out.png')

                # ernet_skel = GMet.seg_to_graph(erosion(ernet_skel/255.))
                ernet_skel = GMet.seg_to_graph(ernet_skel/255.)
                # ernet_graphs.append(ernet_skel)
                ernet_graphs.append(GMet.simple_analysis(ernet_skel))

                erv2_skel = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/erv2_op/sted_{group}{seq}_er_mean_out_bin.png')

                # erv2_skel = GMet.seg_to_graph(erosion(erv2_skel/255.))
                erv2_skel = GMet.seg_to_graph(erv2_skel/255.)
                # erv2_graphs.append(erv2_skel)
                erv2_graphs.append(GMet.simple_analysis(erv2_skel))

                nerdy_skel = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/nerdy_op/Series0{seq:02d}_decon_converted_mean_proc_v2_enhance.png')

                nerdy_skel = GMet.seg_to_graph(nerdy_skel/255.)
                # nerdy_graphs.append(nerdy_skel)
                nerdy_graphs.append(GMet.simple_analysis(nerdy_skel))

                # p4m_vecadam_file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/p4m_vecadam_op/sted_{group}{seq}_er_mean_pred.png'
                input_file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/images/sted_{group}{seq}_er_mean.png'

                p4m_vecadam_seg = process_op(input_file, p4m)

                p4m_vecadam_skel = GMet.seg_to_graph(p4m_vecadam_seg/255.)
                # p4m_vecadam_graphs.append(p4m_vecadam_skel)
                p4m_vecadam_graphs.append(GMet.simple_analysis(p4m_vecadam_skel))

    return gt_graphs, analyzer_graphs, erv2_graphs, ernet_graphs, nerdy_graphs, p4m_vecadam_graphs

    # return gt_graphs, p4m_vecadam_graphs


# def get_graph_metrics(data_list, method):
#     # sourcery skip: inline-immediately-returned-variable
#     """
#     Get graph metrics for each segmentation method
#     metrics: #nodes, #edges, assortativity, clustering, #components
#     """
#     if method in ['analyzer', 'ernet', 'erv2']:
#         graphs = [GMet.seg_to_graph(erosion(seg/255.)) for seg in data_list]
#     else:
#         graphs = [GMet.seg_to_graph(seg/255.) for seg in data_list]
#     graph_metrics = [GMet.simple_analysis(g) for g in graphs]
#     return graph_metrics


def get_graph_metrics(data_list):
    graph_metrics = [GMet.simple_analysis(g) for g in data_list]
    return graph_metrics

# gt_data, analyzer_data, erv2_data, ernet_data, nerdy_data, p4m_vecadam_data = get_graphs()

# gt_graphs, analyzer_graphs, erv2_graphs, ernet_graphs, nerdy_graphs, p4m_graphs = get_graphs()


# gt_graph_metrics = np.array(gt_graphs).T
# analyzer_graph_metrics = np.array(analyzer_graphs).T
# erv2_graph_metrics = np.array(erv2_graphs).T
# ernet_graph_metrics = np.array(ernet_graphs).T
# nerdy_graph_metrics = np.array(nerdy_graphs).T
# p4m_graph_metrics = np.array(p4m_graphs).T


def get_rel_error(gt, method):
    op = []
    for gt_metric, method_metric in zip(gt, method):
        data = []    
        for v1, v2 in zip(gt_metric, method_metric):
            if v1 != 0:
                val = abs(v1 - v2) / abs(v1)
                data.append(val)
        op.append(data)
    return op


def find_outliers(data, threshold=1.5):
    """
    Find outliers in a list of values using the IQR method.

    Parameters:
    - data: 1D array or list, the input data
    - threshold: float, a multiplier to determine the outlier range

    Returns:
    - outliers: 1D array, the values identified as outliers
    """
    # Convert data to a numpy array
    data = np.array(data)

    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers


def get_variation(op):
    data = []
    for num, metric in enumerate(op):
        if num == 2:
            outl = find_outliers(metric)
            for e in outl:
                metric.remove(e)
        # mn = np.mean(metric)
        # st = np.std(metric)
        # print(f'{mn:.2f} +/- {st:.2f}')
        data.append(metric)
    return data


gt_graphs, analyzer_graphs, erv2_graphs, ernet_graphs, nerdy_graphs, p4m_vecadam_graphs = get_graphs()

gt_graph_metrics = np.array(gt_graphs).T
analyzer_graph_metrics = np.array(analyzer_graphs).T
erv2_graph_metrics = np.array(erv2_graphs).T
ernet_graph_metrics = np.array(ernet_graphs).T
nerdy_graph_metrics = np.array(nerdy_graphs).T
p4m_vecadam_metrics = np.array(p4m_vecadam_graphs).T

analyzer_error = get_rel_error(gt_graph_metrics, analyzer_graph_metrics)
ernet_error = get_rel_error(gt_graph_metrics, ernet_graph_metrics)
erv2_error = get_rel_error(gt_graph_metrics, erv2_graph_metrics)
nerdy_error = get_rel_error(gt_graph_metrics, nerdy_graph_metrics)
p4m_error = get_rel_error(gt_graph_metrics, p4m_vecadam_metrics)

# with open('sted_rtn_analyzer_graph_err.pkl', 'wb') as f:
#     pickle.dump(analyzer_error, f)

# with open('sted_rtn_ernet_graph_err.pkl', 'wb') as f:
#     pickle.dump(ernet_error, f)

# with open('sted_rtn_erv2_graph_err.pkl', 'wb') as f:
#     pickle.dump(erv2_error, f)

# with open('sted_rtn_nerdy_graph_err.pkl', 'wb') as f:
#     pickle.dump(nerdy_error, f)

# with open('sted_rtn_p4m_graph_err.pkl', 'wb') as f:
#     pickle.dump(p4m_error, f)

with open('sted_analyzer_graph_err.pkl', 'wb') as f:
    pickle.dump(analyzer_error, f)

with open('sted_ernet_graph_err.pkl', 'wb') as f:
    pickle.dump(ernet_error, f)

with open('sted_erv2_graph_err.pkl', 'wb') as f:
    pickle.dump(erv2_error, f)

with open('sted_nerdy_graph_err.pkl', 'wb') as f:
    pickle.dump(nerdy_error, f)

with open('sted_p4m_graph_err.pkl', 'wb') as f:
    pickle.dump(p4m_error, f)

exit()


# analyzer_error = get_rel_error(gt_graph_metrics, analyzer_graph_metrics)
# # print(analyzer_error)
# analyzer = get_variation(analyzer_error)
# with open('sted_analyzer.pkl', 'wb') as f:
#     pickle.dump(analyzer, f)

# print('----------------------')
# ernet_error = get_rel_error(gt_graph_metrics, ernet_graph_metrics)
# ernet = get_variation(ernet_error)

# with open('sted_ernet.pkl', 'wb') as f:
#     pickle.dump(ernet, f)

# print('----------------------')
# erv2_error = get_rel_error(gt_graph_metrics, erv2_graph_metrics)
# erv2 = get_variation(erv2_error)

# with open('sted_erv2.pkl', 'wb') as f:
#     pickle.dump(erv2, f)

# print('----------------------')
# nerdy_error = get_rel_error(gt_graph_metrics, nerdy_graph_metrics)
# nerdy = get_variation(nerdy_error)

# with open('sted_nerdy.pkl', 'wb') as f:
#     pickle.dump(nerdy, f)

# print('----------------------')
# p4m_error = get_rel_error(gt_graph_metrics, p4m_graph_metrics)
# p4m = get_variation(p4m_error)

# with open('sted_p4m.pkl', 'wb') as f:
#     pickle.dump(p4m, f)

# print('----------------------')

# exit()

# gt_data, p4m_vecadam_data = get_graphs()


# def get_all_metrics():
#     gt_graph_metrics = get_graph_metrics(gt_data)
#     analyzer_graph_metrics = get_graph_metrics(analyzer_data)
#     erv2_graph_metrics = get_graph_metrics(erv2_data)
#     ernet_graph_metrics = get_graph_metrics(ernet_data)
#     nerdy_graph_metrics = get_graph_metrics(nerdy_data)
#     p4m_vecadam_metrics = get_graph_metrics(p4m_vecadam_data)

#     return gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, p4m_vecadam_metrics
    # return gt_graph_metrics, p4m_vecadam_metrics

# gt_graph_metrics = np.array(get_graph_metrics(gt_graphs)).T
# analyzer_graph_metrics = np.array(get_graph_metrics(analyzer_graphs)).T
# erv2_graph_metrics = np.array(get_graph_metrics(erv2_graphs)).T
# ernet_graph_metrics = np.array(get_graph_metrics(ernet_graphs)).T
# nerdy_graph_metrics = np.array(get_graph_metrics(nerdy_graphs)).T
# p4m_vecadam_metrics = np.array(get_graph_metrics(p4m_vecadam_graphs)).T


def get_rel_error(gt, method):
    op = []
    for gt_metric, method_metric in zip(gt, method):
        data = []    
        for v1, v2 in zip(gt_metric, method_metric):
            if v1 != 0:
                val = abs(v1 - v2) / abs(v1)
                data.append(val)
        op.append(data)
    return op

def get_variation(op):
    for metric in op:
        mn = np.nanmean(metric)
        st = np.nanstd(metric)
        print(f'{mn} +/- {st}')

# def get_mean(op):
#     for metric in op:
#         print(np.nanmean(metric))

# def get_std(op):
#     for metric in op:
#         print(np.nanstd(metric))

# op = get_rel_error(gt_graph_metrics, analyzer_graph_metrics)
# get_variation(op)

# print('----------------------')

# op = get_rel_error(gt_graph_metrics, erv2_graph_metrics)
# get_variation(op)
# print('----------------------')

# op = get_rel_error(gt_graph_metrics, ernet_graph_metrics)
# get_variation(op)
# print('----------------------')

# op = get_rel_error(gt_graph_metrics, nerdy_graph_metrics)
# get_variation(op)
# print('----------------------')

# op = get_rel_error(gt_graph_metrics, p4m_vecadam_metrics)
# get_variation(op)
# print('----------------------')

# exit()

# gt_graph_metrics, analyzer_graph_metrics, erv2_graph_metrics, ernet_graph_metrics, nerdy_graph_metrics, p4m_vecadam_metrics = get_all_metrics()

# gt_graph_metrics, p4m_vecadam_metrics = get_all_metrics()


# def get_mean(nparr):
#     return [np.mean(lt) for lt in nparr]

# def get_error(gt, method):
#     err = []
#     for i in range(len(gt)):
#         val = abs(gt[i] - method[i]) / abs(gt[i])
#         err.append(val)
#     return err

# gt_graph_metrics = get_mean(np.array(gt_graph_metrics).T)
# analyzer_graph_metrics = get_mean(np.array(analyzer_graph_metrics).T)
# erv2_graph_metrics = get_mean(np.array(erv2_graph_metrics).T)
# ernet_graph_metrics = get_mean(np.array(ernet_graph_metrics).T)
# nerdy_graph_metrics = get_mean(np.array(nerdy_graph_metrics).T)
# p4m_vecadam_metrics = get_mean(np.array(p4m_vecadam_metrics).T)


# print(gt_graph_metrics)
# print(analyzer_graph_metrics)
# print(erv2_graph_metrics)
# print(ernet_graph_metrics)
# print(nerdy_graph_metrics)
# print(p4m_vecadam_metrics)

# exit()

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


# analyzer_error = get_error(gt_graph_metrics, analyzer_graph_metrics)
# ernet_error = get_error(gt_graph_metrics, ernet_graph_metrics)
# erv2_error = get_error(gt_graph_metrics, erv2_graph_metrics)
# nerdy_error = get_error(gt_graph_metrics, nerdy_graph_metrics)
# p4m_vecadam_error = get_error(gt_graph_metrics, p4m_vecadam_metrics)


# print(analyzer_error)
# print(ernet_error)
# print(erv2_error)
# print(nerdy_error)
# print(p4m_vecadam_error)


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