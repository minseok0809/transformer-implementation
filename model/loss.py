import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    Computer loss at one time step.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """Label Smoothing module
        args:
            size: vocab_size
            padding_idx: index for symbol `padding`
            smoothing: smoothing ratio
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (*, n_classes)
        # target: (*)
        assert x.size(1) == self.size
        with torch.no_grad():
            tgt_dist = torch.zeros_like(x, dtype=torch.float)
            tgt_dist.fill_(
                self.smoothing / (self.size - 2)
            )  # one for padding, another for label
            tgt_dist[:, self.padding_idx] = 0
            tgt_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

            mask = torch.nonzero(target == self.padding_idx)
            if mask.shape[0] > 0:
                tgt_dist.index_fill_(0, mask.squeeze(), 0)

        return self.criterion(x, tgt_dist)