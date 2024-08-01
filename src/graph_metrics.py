import imageio
import numpy as np
import sknw
from PIL import Image
import skimage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx


class GraphMetrics:

    def __init__(self):
        pass

    def get_graph(self, skel):
        return sknw.build_sknw(skel, multi=True, iso=False)

    def process_analyzer_skel(self, analyzer_skel):
        """
        Process the skeletonized output from the analyzer
        """
        op = analyzer_skel[:,:,0]
        op = op[52:1670, 52:1670]
        op = skimage.transform.resize(op, (128, 128), anti_aliasing=True)
        mval = min(np.unique(op))
        op[np.where(op==mval)] = 0
        op[np.where(op!=mval) and np.where(op!=0)] = 255
        op = op/255

        plt.imshow(op)
        plt.show()

        return skeletonize(op).astype(np.uint16)

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
        ratio_nodes = size_G0_nodes / no_nodes
        ratio_edges = size_G0_edges / no_edges
        return [
            no_nodes,
            no_edges,
            assortativity,
            clustering,
            compo,
            ratio_nodes,
            ratio_edges
        ]
    
    def seg_to_graph(self, seg_img):
        """
        Convert segmentation output to a graph per frame
        """
        # ernet_enh = imageio.imread(data_path)
        ernet_skel = skeletonize(seg_img).astype(np.uint16)
        return self.get_graph(ernet_skel)

    def get_metrics(self, data_path):
        G = self.seg_to_graph(data_path)
        return self.simple_analysis(G)