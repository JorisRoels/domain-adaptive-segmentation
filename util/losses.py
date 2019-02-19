
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion

class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=False):
        super(CrossEntropyLoss, self).__init__()

        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):

        # input: (n, c, h, w), target: (n, h, w)
        input_size = input.size()
        if len(input_size) == 2: # 1D
            n, c = input.size()
        elif len(input_size) == 4: # 2D
            n, c, h, w = input.size()
        else: # 3D
            n, c, h, w, d = input.size()
        # log_p: (n, c, h, w)
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            # ==0.2.X
            log_p = F.log_softmax(input)
        else:
            # >=0.3
            log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        if len(input_size) == 2: # 1D
            log_p = log_p.contiguous()
            log_p = log_p[target.view(n, 1).repeat(1, c) >= 0]
        elif len(input_size) == 4: # 2D
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
            log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        else: # 3D
            log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()
            log_p = log_p[target.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=self.weight)
        if self.size_average:
            loss /= mask.data.sum()
        return loss

class MSELoss(nn.Module):

    def forward(self, input, target):

        return torch.mean((input-target)**2)