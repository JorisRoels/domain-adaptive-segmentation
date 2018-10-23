
import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(nn.Module):

    def forward(self, input, target):

        eps = 1e-10

        predicted_probabilities = F.softmax(input, dim=1)[:, 1:2, :, :]
        target = target.float()

        intersection = (predicted_probabilities * target).sum()
        union = predicted_probabilities.sum() + target.sum() - intersection

        return - (intersection+eps) / (union+eps)

class MSELoss(nn.Module):

    def forward(self, input, target):

        return torch.mean((input-target)**2)