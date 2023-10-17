import imageio
import numpy as np
import sknw
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx


class GraphMetrics:

    def __init__(self, data_path):
        self.data_path = data_path

    def get_graph(self, skel):
        return sknw.build_sknw(skel, multi=False, iso=False)

    def process_analyzer_skel(self, analyzer_skel):
        """
        Process the skeletonized output from the analyzer
        """
        op = analyzer_skel[:,:,0]
        op = op[52:1670, 52:1670]
        mval = min(np.unique(op))
        op[np.where(op==mval)] = 0
        op[np.where(op!=mval) and np.where(op!=0)] = 255
        op = op/255

        return skeletonize(op).astype(np.uint16)

    def resize_analyzer_skel(self, analyzer_skel):
        """
        Resize the skeletonized output from the analyzer
        """
        img = Image.fromarray(self.process_analyzer_skel(analyzer_skel))
        img = img.resize((128, 128), Image.LANCZOS)
        return np.array(img)
    
    def jaccard_similarity(self, G1, G2):
        nodes1 = set(G1.nodes)
        nodes2 = set(G2.nodes)
        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        return intersection / union

    def jaccard_edge_similarity(self, g1, g2):
        # sourcery skip: inline-immediately-returned-variable
        edge_set1 = set(g1.edges)
        edge_set2 = set(g2.edges)
        jaccard_index = len(edge_set1.intersection(edge_set2)) / len(edge_set1.union(edge_set2))
        return jaccard_index
        

    def simple_analysis(self, G):
        no_nodes = G.number_of_nodes()
        no_edges = G.number_of_edges()
        assortativity = nx.degree_assortativity_coefficient(G)
        clustering = nx.average_clustering(G)
        compo = nx.number_connected_components(G)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        size_G0_edges = G0.number_of_edges()
        size_G0_nodes = G0.number_of_nodes()
        # ratio_nodes = size_G0_nodes / no_nodes
        # ratio_edges = size_G0_edges / no_edges
        return [
            no_nodes,
            no_edges,
            assortativity,
            clustering,
            compo
        ]
    
    def seg_to_graph(self, data_path):
        """
        Convert segmentation output to a graph per frame
        """
        ernet_enh = imageio.imread(data_path)
        ernet_skel = skeletonize(ernet_enh).astype(np.uint16)
        return self.get_graph(ernet_skel)

    # ernet: /localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/ernet_op/sted_climp1_er_mean_out.png
    # erv2: /localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/erv2_op/sted_climp1_er_mean_out_0033_bin.png
    # nerdy: /localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/nerdy_op/Series001_decon_converted_mean_proc_v2_enhance.png
    # nerdynet: /localhome/asa420/MIAL/data/sted-data/vess_enh_unet/climp/nerdynet_op_seg/sted_climp1_er_mean_pred.png

    def get_metrics(self, data_path):
        G = self.seg_to_graph(data_path)
        return self.simple_analysis(G)
    
    def analyze_metrics(self):
        """
        Analyze the metrics
        """
        gt_data = []
        erv2_data = []
        ernet_data = []
        analyzer_data = []
        nerdy_data = []
        nerdynet_data = []

        