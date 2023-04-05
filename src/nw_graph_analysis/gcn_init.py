import torch
import torch.nn as nn



class GroupBase:
    def __init__(self, dim, identity):
        super(GroupBase, self).__init__()
        self.dim = dim
        self.register_buffer('identity', torch.Tensor(identity))

    def elements(self):
        pass

    def product(self, h, h_p):
        pass

    def inverse(self, h):
        pass

    def left_action_on_R2(self, h_batch, x_batch):
        pass

    def left_action_on_H(self, h_batch, h_p_batch):
        pass

    def matrix_representation(self, h):
        pass

    def determinant(self, h):
        pass

    def normalize_group_elements(self, h):
        pass

class CyclicGroup(GroupBase):
