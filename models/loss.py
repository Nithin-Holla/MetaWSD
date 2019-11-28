import torch

from torch import nn


class BCEWithLogitsLossAndIgnoreIndex(nn.Module):
    def __init__(self, ignore_index=-1):
        super(BCEWithLogitsLossAndIgnoreIndex, self).__init__()
        self.ignore_index = ignore_index
        self.loss_module = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        retain_indices = torch.nonzero(target != self.ignore_index).view(-1)
        output = output[retain_indices].view(-1)
        target = target[retain_indices].float()
        loss = self.loss_module(output, target)
        return loss


class AdaptiveLogSoftmaxWithLossAndIgnoreIndex(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, ignore_index=-1,
                 include_output=False):
        super(AdaptiveLogSoftmaxWithLossAndIgnoreIndex, self).__init__()
        self.ignore_index = ignore_index
        self.include_output = include_output
        self.loss_module = nn.AdaptiveLogSoftmaxWithLoss(in_features=in_features,
                                                         n_classes=n_classes,
                                                         cutoffs=cutoffs,
                                                         div_value=div_value,
                                                         head_bias=head_bias)

    def forward(self, output, target):
        retain_indices = torch.nonzero(target != self.ignore_index).view(-1)
        output = output[retain_indices]
        target = target[retain_indices]
        log_prob, loss = self.loss_module(output, target)
        if self.include_output:
            return log_prob, loss
        else:
            return loss
