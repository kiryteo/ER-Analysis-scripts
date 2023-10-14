import imageio
import numpy as np
import sknw
import matplotlib.pyplot as plt



class GraphMetrics:

    def __init__(self, img):
        self.img = img

    def get_graph(self):
        return sknw.build_sknw(self.img)

    def process_analyzer_skel(self):
        # op = analyzer_skel[:,:,0]
        op = self.img[:,:,0]
        mval = min(np.unique(op))
        op[np.where(op==mval)] = 0
        op[np.where(op!=mval) and np.where(op!=0)] = 255
        op = op/255
        return skeletonize(op).astype(np.uint16)
