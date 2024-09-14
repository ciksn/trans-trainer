import torch
import torch.nn as nn

class giou_loss(nn.Module):
    def __init__(self,eps=1e-7, reduction='mean') -> None:
        super().__init__()
        self.eps=eps
        self.reduction=reduction
    
    def forward(self, preds, bbox):
        ix1 = torch.max(preds[:, 0], bbox[:, 0])
        iy1 = torch.max(preds[:, 1], bbox[:, 1])
        ix2 = torch.min(preds[:, 2], bbox[:, 2])
        iy2 = torch.min(preds[:, 3], bbox[:, 3])

        iw = (ix2 - ix1 + 1.0).clamp(0.)
        ih = (iy2 - iy1 + 1.0).clamp(0.)

        # overlap
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters + self.eps

        # ious
        ious = inters / uni

        ex1 = torch.min(preds[:, 0], bbox[:, 0])
        ey1 = torch.min(preds[:, 1], bbox[:, 1])
        ex2 = torch.max(preds[:, 2], bbox[:, 2])
        ey2 = torch.max(preds[:, 3], bbox[:, 3])
        ew = (ex2 - ex1 + 1.0).clamp(min=0.)
        eh = (ey2 - ey1 + 1.0).clamp(min=0.)

        # enclose erea
        enclose = ew * eh + self.eps

        giou = ious - (enclose - uni) / enclose

        loss = 1 - giou

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError
        return loss
