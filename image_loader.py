import glob
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch


# def make_dataset():
#     dataset = []
#     original_img_rpath = './prepped/images_prepped_train'
#     seg_mask_rpath = './prepped/annotations_prepped_train'
#     for img_path in glob.glob(os.path.join(original_img_rpath, '*.jpg')):
#         basename = os.path.basename(img_path)
#         original_img_path = os.path.join(original_img_rpath, basename)
#         basename = basename[:-3] + 'png'
#         seg_mask_path = os.path.join(seg_mask_rpath, basename)
#         # print(original_img_path, seg_mask_path)
#         dataset.append([original_img_path, seg_mask_path])
#     # print(dataset)
#     return dataset


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
        return img, seg

    def __len__(self):
        return len(self.tensors_dataset)


# train_dataset = CityscapesLoader('train')
# for i in range(0, 2):
#     img, seg = train_dataset[i]
#     print(img)
#     print(seg[7])

