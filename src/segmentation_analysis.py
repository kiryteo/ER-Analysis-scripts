import os
from pathlib import Path
import copy
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from skimage import restoration
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt
from torchvision import transforms
import contextlib

from segmentation_metrics import SegmentationMetrics
from graph_metrics import GraphMetrics


from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

SMet = SegmentationMetrics()
GMet = GraphMetrics()

groups = ['climp', 'control', 'rtn']

prefix = '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/'

analyzer_prefix = {'control': 'ct', 'climp': 'c', 'rtn': 'r'}



class Plotter:
    @staticmethod
    def plot_subplot(subplot_num, data, title):
        plt.subplot(1, 7, subplot_num)
        plt.imshow(data, cmap='gray')
        plt.title(title)
        plt.axis('off')

    @staticmethod
    def save_comparison_plot(img, mask, analyzer, erv2, ernet, nerdy, nerdynet, filename):
        plt.figure(figsize=(20, 16))

        Plotter.plot_subplot(1, img, 'Input')
        Plotter.plot_subplot(2, mask, 'Mask')
        Plotter.plot_subplot(3, analyzer, 'AnalyzER')
        Plotter.plot_subplot(4, erv2, 'ERV2')
        Plotter.plot_subplot(5, ernet, 'ERnet')
        Plotter.plot_subplot(6, nerdy, 'nERdy')
        Plotter.plot_subplot(7, nerdynet, 'nERdyNet')

        plt.subplots_adjust(wspace=0.02)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
        plt.close()

    @staticmethod
    def save_comparison_plot_v2(img, mask, analyzer, ernet, erv2, nerdy, nerdynet, output_filename):
        images = [img, mask, analyzer, ernet, erv2, nerdy, nerdynet]
        titles = ['Input', 'Mask', 'AnalyzER', 'ERnet', 'ERnet-v2', 'nERdy', 'nERdy+']

        plt.figure(figsize=(15, 5))
        for i, (image, title) in enumerate(zip(images, titles), 1):
            plt.subplot(1, 7, i)
            plt.axis('off')
            plt.imshow(image, cmap='gray')
            plt.title(title)

        plt.subplots_adjust(wspace=0.02)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def plot_graph(graph, input_file):
        plt.imshow(imageio.imread(input_file), cmap='gray')

        degree_list = graph.degree
        tgraph = copy.deepcopy(graph)
        node_set = tgraph.nodes
        degree_list = tgraph.degree
        node_coords = np.array([node_set[node]['o'] for node in node_set])
        nps = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]
        nps = np.array(nps)

        for (s, e) in graph.edges():
            ps = graph[s][e][0]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        plt.plot(nps[:, 1], nps[:, 0], '.', markerfacecolor='red', markeredgecolor='red', mew=1)
        plt.axis('off')
        plt.savefig('climp1_analyzer_graph_overlay.png', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

class DataLoader:
    @staticmethod
    def qualitative_comparison(base_path):
        file_paths = {
            "img": "images/sted_climp8_er_mean.png",
            "mask": "masks/sted_climp8_er_mean_mask.png",
            "analyzer": "analyzer_op/ER_ce8.png",
            "ernet": "ernet_op/sted_climp8_er_mean_out.png",
            "erv2": "erv2_op/sted_climp8_er_mean_out_bin.png",
            "nerdy": "nerdy_op/Series008_decon_converted_mean_proc_v2_enhance.png",
            "nerdynet": "nerdynet_v2/sted_climp8_er_mean_pred.png"
        }

        images = {}
        for key, relative_path in file_paths.items():
            try:
                images[key] = imageio.imread(Path(base_path) / relative_path)
            except FileNotFoundError:
                print(f"File not found: {relative_path}")
            except Exception as e:
                print(f"Error reading {relative_path}: {e}")

        return images

    @staticmethod
    def load_all_data(groups, prefix):
        p4m = ModelLoader.load_models()
        gt_data = []
        p4m_vecadam_data = []

        for group in groups:
            for num in range(1, 17):
                with contextlib.suppress(Exception):
                    gt_file = f'{prefix}{group}/updated_masks/sted_{group}{num}_er_mean_mask.png'
                    gt_data.append(imageio.imread(gt_file))

                    p4m_file = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/images/sted_{group}{num}_er_mean.png'
                    p4m_vecadam_data.append(ModelProcessor.process_op(p4m_file, p4m))

        return gt_data, p4m_vecadam_data

class ModelLoader:
    @staticmethod
    def load_models():
        p4m = NNet(1, 1)
        p4m.load_state_dict(torch.load('NNet_groupy_p4m_v2_VecAdam.pth'))
        return p4m

class ModelProcessor:
    @staticmethod
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
        thr = threshold_mean(subt)  # in case of confocal data

        subt2 = copy.deepcopy(subt)
        subt2[subt < thr] = 0.
        subt2[subt >= thr] = 255.

        return subt2

class Metrics:
    @staticmethod
    def compute_iou_metrics(pred_data, gt_data):
        metrics = []
        for pred, gt in zip(pred_data, gt_data):
            if pred.max() == 255:
                pred = pred / 255
            if gt.max() == 255:
                gt = gt / 255
            metrics.append(torchmetrics.classification.BinaryJaccardIndex()(torch.tensor(pred), torch.tensor(gt)))
        return np.mean(metrics)

    @staticmethod
    def compute_f1_score(pred_data, gt_data):
        metrics = []
        for pred_mask, true_mask in zip(pred_data, gt_data):
            if pred_mask.max() == 255:
                pred_mask = pred_mask / 255
            if true_mask.max() == 255:
                true_mask = true_mask / 255
            metrics.append(torchmetrics.classification.BinaryF1Score()(torch.tensor(pred_mask), torch.tensor(true_mask)))
        return np.mean(metrics)

    @staticmethod
    def compute_dice_coefficient(pred_data, gt_data):
        metrics = []
        for pred_mask, true_mask in zip(pred_data, gt_data):
            if pred_mask.max() == 255.:
                pred_mask = pred_mask / 255.
            if true_mask.max() == 255.:
                true_mask = true_mask / 255.
            metrics.append(torchmetrics.classification.Dice()(torch.tensor(pred_mask.astype('int')), torch.tensor(true_mask.astype('int'))))
        return np.mean(metrics)

    @staticmethod
    def print_metric_results(metric_name, metric_value):
        print(f'{metric_name}: {metric_value}')

    @staticmethod
    def get_segmentation_metrics(methods_data, gt_data):
        for method_name, method_data in methods_data.items():
            iou = Metrics.compute_iou_metrics(method_data, gt_data)
            f1 = Metrics.compute_f1_score(method_data, gt_data)
            dice = Metrics.compute_dice_coefficient(method_data, gt_data)

            Metrics.print_metric_results(f'{method_name} f1', f1)
            Metrics.print_metric_results(f'{method_name} dice', dice)
            Metrics.print_metric_results(f'{method_name} iou', iou)

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
        x = self.convt(x)
        return x

class SegmentationAnalysis:
    def __init__(self):
        self.p4m = ModelLoader.load_models()
        self.sym = {'climp': 'c', 'control': 'ct', 'rtn': 'r'}
        base_path = f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/'
        self.methods = {
            'gt': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/gt_skel/sted_{group}{seq}_proc_skel.png', 'process': GMet.get_graph},
            'analyzer': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/analyzer_skel/ER_{sym[group]}{seq}.png', 'process': GMet.process_analyzer_skel},
            'ernet': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/ernet_op/sted_{group}{seq}_er_mean_out.png', 'process': lambda x: GMet.seg_to_graph(x / 255.)},
            'erv2': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/erv2_op/sted_{group}{seq}_er_mean_out_bin.png', 'process': lambda x: GMet.seg_to_graph(x / 255.)},
            'nerdy': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/nerdy_op/Series0{seq:02d}_decon_converted_mean_proc_v2_enhance.png', 'process': lambda x: GMet.seg_to_graph(x / 255.)},
            'p4m_vecadam': {'path': '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/images/sted_{group}{seq}_er_mean.png', 'process': lambda x: GMet.seg_to_graph(ModelProcessor.process_op(x, self.p4m) / 255.)}
        }

    def get_graphs(self):
        graphs = {method: [] for method in self.methods}
        for group in groups:
            for seq in range(1, 17):
                for method, details in self.methods.items():
                    with contextlib.suppress(Exception):
                        file_path = details['path'].format(group=group, seq=seq, sym=self.sym)
                        image = imageio.imread(file_path)
                        processed_image = details['process'](image)
                        graphs[method].append(processed_image)
        return tuple(graphs[method] for method in self.methods)

    def get_graph_metrics(self, data_list):
        graph_metrics = [GMet.simple_analysis(g) for g in data_list]
        return graph_metrics

    def get_mean(self, nparr):
        return [np.mean(lt) for lt in nparr]

    def get_error(self, gt, method):
        err = []
        for i in range(len(gt)):
            val = abs(gt[i] - method[i]) / abs(gt[i])
            err.append(val)
        return err

if __name__ == "__main__":
    analysis = SegmentationAnalysis()

    groups = ["climp", "control", "reticulon", "atlastin"]
    prefix = '/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/'

    # Load data
    images = DataLoader.qualitative_comparison(prefix)
    gt_data, p4m_vecadam_data = DataLoader.load_all_data(groups, prefix)

    # Process and plot
    Plotter.save_comparison_plot_v2(images['img'], images['mask'], images['analyzer'], images['ernet'], images['erv2'], images['nerdy'], images['nerdynet'], "output_filename.png")

    # Compute metrics
    methods_data = {
        "p4m_vecadam": p4m_vecadam_data
    }
    Metrics.get_segmentation_metrics(methods_data, gt_data)

    gt_graphs, analyzer_graphs, erv2_graphs, ernet_graphs, nerdy_graphs, p4m_vecadam_graphs = analysis.get_graphs()

    gt_graph_metrics = analysis.get_graph_metrics(gt_graphs)
    analyzer_graph_metrics = analysis.get_graph_metrics(analyzer_graphs)
    erv2_graph_metrics = analysis.get_graph_metrics(erv2_graphs)
    ernet_graph_metrics = analysis.get_graph_metrics(ernet_graphs)
    nerdy_graph_metrics = analysis.get_graph_metrics(nerdy_graphs)
    p4m_vecadam_metrics = analysis.get_graph_metrics(p4m_vecadam_graphs)

    gt_graph_metrics = analysis.get_mean(np.array(gt_graph_metrics).T)
    analyzer_graph_metrics = analysis.get_mean(np.array(analyzer_graph_metrics).T)
    erv2_graph_metrics = analysis.get_mean(np.array(erv2_graph_metrics).T)
    ernet_graph_metrics = analysis.get_mean(np.array(ernet_graph_metrics).T)
    nerdy_graph_metrics = analysis.get_mean(np.array(nerdy_graph_metrics).T)
    p4m_vecadam_metrics = analysis.get_mean(np.array(p4m_vecadam_metrics).T)

    analyzer_error = analysis.get_error(gt_graph_metrics, analyzer_graph_metrics)
    ernet_error = analysis.get_error(gt_graph_metrics, ernet_graph_metrics)
    erv2_error = analysis.get_error(gt_graph_metrics, erv2_graph_metrics)
    nerdy_error = analysis.get_error(gt_graph_metrics, nerdy_graph_metrics)
    p4m_vecadam_error = analysis.get_error(gt_graph_metrics, p4m_vecadam_metrics)
