### topology loss class for PyTorch
from topologylayer.nn import AlphaLayer, LevelSetLayer2D, BarcodePolyFeature, PartialSumBarcodeLengths, SumBarcodeLengths, TopKBarcodeLengths
import torch, numpy as np, matplotlib.pyplot as plt
from torch.nn.functional import sigmoid, relu

## Symmetric Topology Loss based on Gabrielsson's work
## DS Sep 2023

class TopologyLoss(torch.nn.Module):
    def __init__(self, k=49, size=(32, 32)):
        super(TopologyLoss, self).__init__()
        self.pd_feat = LevelSetLayer2D(size=size,  sublevel=False)
        self.f1 = PartialSumBarcodeLengths(dim=0, skip=k)
        self.f2 = TopKBarcodeLengths(dim=0, k=k)
        self.f3 = TopKBarcodeLengths(dim=1, k=k)
        self.f4 = SumBarcodeLengths(dim=0)
        self.size = size
        self.k = k

    def forward(self, x):
        total_loss = 0

        for j in range(x.shape[0]):
            xx = x[j].squeeze() # this loss is for grayscale images only!
            pd = self.pd_feat(xx)
            pd_ = self.pd_feat(1-xx)
            loss = self.f1(pd) + torch.mean((1 - self.f2(pd))**2) + torch.mean((1 - self.f3(pd_))**2) + self.f4(pd_)  + torch.mean(sigmoid(xx)**2)
            total_loss += loss
        
        return total_loss/(j + 1) # average over batch



class TopologyLossBaseline(torch.nn.Module):
    def __init__(self, k=49, size=(32, 32)):
        super(TopologyLossBaseline, self).__init__()
        self.pd_feat = LevelSetLayer2D(size=size,  sublevel=False)
        self.f1 = PartialSumBarcodeLengths(dim=0, skip=k)
        self.f2 = TopKBarcodeLengths(dim=0, k=k)
        self.f3 = TopKBarcodeLengths(dim=1, k=k)
        self.f4 = SumBarcodeLengths(dim=0)
        self.size = size

    def forward(self, x):
        total_loss = 0

        for j in range(x.shape[0]):
            xx = x[j].squeeze() # this loss is for grayscale images only!
            pd = self.pd_feat(xx)
            # pd_ = self.pd_feat(1-xx)
            loss = self.f1(pd) + torch.mean((1 - self.f2(pd))**2)  + torch.mean(sigmoid(xx)**0.5)
            total_loss += loss
        
        return total_loss/(j + 1) # average over batch
    
