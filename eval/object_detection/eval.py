import torch
import numpy

def get_iou(pred, bbox):
    """
    Compute IoU for two sets of boxes in batch.
    :param pred: (batch_size, 4), each box is in the format [x_min, y_min, x_max, y_max]
    :param bbox: (batch_size, 4), each box is in the format [x_min, y_min, x_max, y_max]
    :return: IoU for each pair in the batch (batch_size,)
    """
    if isinstance(pred,numpy.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(bbox,numpy.ndarray):
        bbox = torch.from_numpy(bbox)
    # Get the coordinates of the intersection rectangle
    x1_inter = torch.max(pred[:, 0], bbox[:, 0])
    y1_inter = torch.max(pred[:, 1], bbox[:, 1])
    x2_inter = torch.min(pred[:, 2], bbox[:, 2])
    y2_inter = torch.min(pred[:, 3], bbox[:, 3])

    # Compute the area of intersection
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

    # Compute the area of both sets of boxes
    box1_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    box2_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Compute the union area
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou.mean()