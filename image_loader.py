import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch


# define transforms
resize_pil = transforms.Resize((256, 512))  # use 8, 16 to test
to_t = transforms.ToTensor()
to_resized_tensor = transforms.Compose([resize_pil, to_t])
# to_pil = transforms.ToPILImage()


def one_hot_transform(seg_tensor):
    seg_tensor[:, :, :] *= 255  # seg array is floats = category/255
    # seg_tensor[seg_tensor[:, :, :] == -1] = 34
    # assume -1 is fitered out? I think this is handled by dataloader
    height = list(seg_tensor.shape)[1]
    width = list(seg_tensor.shape)[2]
    # 35 classes: 0-33 and -1
    ret_tensor = torch.zeros(35, height, width)
    for chan in range(0, 34):
        ret_tensor[chan, :, :] = seg_tensor[0, :, :] == chan
    # ret_tensor[34, :, :] = ((seg_tensor[0, :, :] == -1) or (seg_tensor[0, :, :] == 34))
    return ret_tensor


class CityscapesLoader(torch.utils.data.Dataset):
    # split must be 'train', 'test', or 'val'
    def __init__(self, split):
        super(CityscapesLoader, self).__init__()
        self.tensors_dataset = Cityscapes(root='./data/cityscapes', split=split, mode='fine', target_type='semantic',
                                          transform=to_resized_tensor, target_transform=to_resized_tensor)

    def __getitem__(self, item):
        img, seg = self.tensors_dataset[item]
        seg = one_hot_transform(seg)  # .squeeze(0)  # need to remove classes dimension for CrossEntropyLoss
        # seg = seg.long()
        return img, seg

    def __len__(self):
        return len(self.tensors_dataset)


# train_dataset = CityscapesLoader('train')
# for i in range(0, 2):
#     img, seg = train_dataset[i]
#     print(img)
#     print(seg[7])


# the higher-order representation of edges captured by network is different than simple L1Loss
def get_edges(pred_seg, gt_seg):
    edge_tensor = gt_seg - pred_seg
    # a mismatched edge will have 1s in the channel that is the gt, and -1s in the channel that was predicted
    # softmax predicts one channel only to put positive 1 in
    # remove negative values only
    edge_tensor = torch.nn.functional.relu(edge_tensor, inplace=False)  # inplace=True would be faster? default False
    return edge_tensor


# need to change models' channel dimensions to use this instead
def get_edges2(pred_seg, gt_seg):
    edge_tensor = gt_seg - pred_seg
    # remove negative values only
    missed_edge_tensor = torch.nn.functional.relu(edge_tensor)
    # flip negatives to positive, then get those values only
    edge_tensor[:, :, :] *= -1
    wrong_edge_tensor = torch.nn.functional.relu(edge_tensor)
    ret_tensor = torch.cat([missed_edge_tensor, wrong_edge_tensor], 0)
    return ret_tensor


# intersection over union assuming both tensors are [BATCH_SIZE, classes, h, w]
def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.round().int()  # convert to 0s and 1s instead of probabilities
    labels = labels.int()
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # credit to https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    SMOOTH = 1e-6
    intersection = (outputs & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((2, 3))  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    del(outputs)
    del(labels)
    # iou is score for every class in every batch
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    ret = iou.mean().item()
    del iou
    return ret  # Or thresholded.mean() if you are interested in average across the batch

