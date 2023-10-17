import imageio
import glob
from skimage.morphology import erosion

from segmentation_metrics import SegmentationMetrics
from graph_metrics import GraphMetrics


groups = ['climp', 'control', 'rtn']

prefix = '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/'

# climp/analyzer_op/ER_ce1.png

seg = SegmentationMetrics()
gm = GraphMetrics()


def get_seg_op(method):
    method_name = {'gt': 'masks','analyzer': 'analyzer_op', 'erv2': 'erv2_op', 'ernet': 'ernet_op', 'nerdy': 'nerdy_op', 'nerdynet': 'nerdynet_op'}
    data = []
    # for group in groups:
    #     try:
    #         for num in range(1, 17):
    #             file = f'{prefix}{group}/{method_name[method]}/*{}*.png'
    #     # files = glob.glob(f'{prefix}{group}/{method_name[method]}/*')
    #     data.extend(imageio.imread(file) for file in files)
    # return data


import contextlib
# gt_data = get_seg_op('gt')
# analyzer_data = get_seg_op('analyzer')
# erv2_data = get_seg_op('erv2')
# ernet_data = get_seg_op('ernet')
# nerdy_data = get_seg_op('nerdy')
# nerdynet_data = get_seg_op('nerdynet')

gt_data = []
analyzer_data = []
erv2_data = []
ernet_data = []
nerdy_data = []
nerdynet_data = []

analyzer_prefix = {'control': 'ct', 'climp': 'c', 'rtn': 'r'}

for group in groups:
    for num in range(1, 17):
        with contextlib.suppress(Exception):
            file = f'{prefix}{group}/masks/sted_{group}{num}_er_mean_mask.png'
            gt_data.append(imageio.imread(file))
            file = f'{prefix}{group}/analyzer_op/ER_{analyzer_prefix[group]}e{num}.png'
            analyzer_data.append(imageio.imread(file))
            file = f'{prefix}{group}/erv2_op/sted_{group}{num}_er_mean_out_bin.png'
            erv2_data.append(imageio.imread(file))
            file = f'{prefix}{group}/ernet_op/sted_{group}{num}_er_mean_out.png'
            ernet_data.append(imageio.imread(file))
            file = f'{prefix}{group}/nerdy_op/Series0{num:02d}_decon_converted_mean_proc_v2_enhance.png'
            nerdy_data.append(imageio.imread(file))
            file = f'{prefix}{group}/nerdynet_op_seg/sted_{group}{num}_er_mean_pred.png'
            nerdynet_data.append(imageio.imread(file))

analyzer = []
erv2 = []
ernet = []
nerdy = []
nerdynet = []


for pred, gt in zip(analyzer_data, gt_data):
    pred = seg.resize_analyzer_bin_op(pred)
    analyzer.append(seg.intersection_over_union(pred, gt))


for pred, gt in zip(erv2_data, gt_data):
    pred = seg.process_erv2_output(pred)
    erv2.append(seg.intersection_over_union(pred, gt))

for pred, gt in zip(ernet_data, gt_data):
    ernet.append(seg.intersection_over_union(pred, gt))

for pred, gt in zip(nerdy_data, gt_data):
    nerdy.append(seg.intersection_over_union(pred, gt))

for pred, gt in zip(nerdynet_data, gt_data):
    nerdynet.append(seg.intersection_over_union(pred, gt))


import numpy as np

print(f'Analyzer: {np.mean(analyzer)}')
print(f'ERV2: {np.mean(erv2)}')
print(f'Ernet: {np.mean(ernet)}')
print(f'Nerdy: {np.mean(nerdy)}')
print(f'Nerdynet: {np.mean(nerdynet)}')



def get_graphs():
    erv2_graphs = []
    ernet_graphs = []
    nerdy_graphs = []
    nerdynet_graphs = []

    for group in groups:
        

