from image_loader import CityscapesLoader, iou
from Stcgan_net import *
import torch
import torch.utils.data as Data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load from path
G1_from_G1D1 = Generator_first().to(device)
G1_from_G1D1.load_state_dict(torch.load('/work/LAS/jannesar-lab/mburke/SegEdgeGAN/saved/2020-04-15-11:16:33.560820/generator1_26.pkl'))  # 'G1D1G2D2_ave_e26_2020-04-15-11:16:33.560820.csv'
G1_from_G1D1.eval()
G1_from_G1D1G2D2 = Generator_first().to(device)
G1_from_G1D1G2D2.load_state_dict(torch.load('/work/LAS/jannesar-lab/mburke/SegEdgeGAN/saved/2020-04-14-23:34:27.841495/generator1_26.pkl'))  # 'G1D1G2D2_ave_e26_2020-04-14-23:34:27.841495.csv'
G1_from_G1D1G2D2.eval()

print("models loaded")

# keep track of average
ave_from_G1D1 = 0
ave_from_G1D1G2D2 = 0

# get test set dataloader
test_dataset = CityscapesLoader('test')
test_data_loader = Data.DataLoader(test_dataset, batch_size=1)

with torch.no_grad():
    for i, data in enumerate(test_data_loader):
        # get batch of data
        original_image, seg_gt = data
        original_image = original_image.to(device)
        seg_gt = seg_gt.to(device)
        seg_gt_flat = seg_gt.argmax(axis=1).long().to(device)  # needed for NLLLoss

        g1_output_from_1 = G1_from_G1D1(original_image)
        g1_output_from_2 = G1_from_G1D1G2D2(original_image)

        iou_score_1 = iou(g1_output_from_1, seg_gt)
        iou_score_2 = iou(g1_output_from_2, seg_gt)

        ave_from_G1D1 += iou_score_1
        ave_from_G1D1G2D2 += iou_score_2

    print('G1D1', ave_from_G1D1)
    print(ave_from_G1D1 / len(test_data_loader))
    print('G1D1G2D2', ave_from_G1D1G2D2)
    print(ave_from_G1D1G2D2 / len(test_data_loader))

