import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch


# define transforms
resize_pil = transforms.Resize((8, 16))
to_t = transforms.ToTensor()
to_resized_tensor = transforms.Compose([resize_pil, to_t])
# to_pil = transforms.ToPILImage()


def one_hot_transform(seg_tensor):
    seg_tensor[:, :, :] *= 255  # seg array is floats = category/255
    height = list(seg_tensor.shape)[1]
    width = list(seg_tensor.shape)[2]
    # 35 classes: 0-33 and -1
    ret_tensor = torch.zeros(35, height, width)
    for chan in range(0, 33):
        ret_tensor[chan, :, :] = seg_tensor[0, :, :] == chan
    ret_tensor[34, :, :] = seg_tensor[0, :, :] == -1
    return ret_tensor


class CityscapesLoader(torch.utils.data.Dataset):
    # split must be 'train', 'test', or 'val'
    def __init__(self, split):
        super(CityscapesLoader, self).__init__()
        self.tensors_dataset = Cityscapes(root='./data/cityscapes', split=split, mode='fine', target_type='semantic',
                                          transform=to_resized_tensor, target_transform=to_resized_tensor)

    def __getitem__(self, item):
        img, seg = self.tensors_dataset[item]
        seg = one_hot_transform(seg)
        seg = seg.long()
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

