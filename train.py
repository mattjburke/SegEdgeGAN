from image_loader import CityscapesLoader
from image_loader import get_edges
from Stcgan_net import *
import torch
import torch.utils.data as Data
import os
import pandas as pd
from datetime import datetime
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor
REAL = 1
FAKE = 0

#hyperparam
BATCH_SIZE = 1
lambda1 = 5  #high due to L1Loss instead of BCELoss?
lambda2 = 0.1
lambda3 = 0.1


def single_gpu_train():
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)
    G2 = Generator_second().to(device)
    D1 = Discriminator_first().to(device)
    D2 = Discriminator_second().to(device)


    criterion_d = torch.nn.BCELoss()
    # criterion_g = torch.nn.L1Loss()
    # criterion_g_data = torch.nn.CrossEntropyLoss()
    criterion_g_data = torch.nn.NLLLoss()  # since CrossEntopyLoss includes softmax
    # criterion_g_adv = torch.nn.BCELoss()  # need separate criterion to carry loss only through Gs? No
    optimizer_d = torch.optim.Adam([
        {'params': D1.parameters()},
        {'params': D2.parameters()}], lr=0.001)  # does sharing parameters make sense?
    optimizer_g = torch.optim.Adam([
        {'params': G1.parameters()},
        {'params': G2.parameters()}], lr=0.001)  # does sharing parameters make sense?

    # Lists to keep track of progress
    img_seg_list = []
    iters = 0
    epochs = []
    total_losses = []
    L_data1_losses = []
    L_data2_losses = []
    L_cgan1_losses = []
    D1_losses = []
    G1_adv_losses = []
    L_cgan2_losses = []
    D2_losses = []
    G2_adv_losses = []
    time_begin = str(datetime.now()).replace(' ', '-')
    path = '/work/LAS/jannesar-lab/mburke/SegEdgeGAN/'

    for epoch in range(100000):
        for i, data in enumerate(train_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)
            seg_gt_flat = seg_gt.argmax(axis=1).squeeze(1).long()
            # print('seg_gt', seg_gt.dtype, seg_gt.shape)
            # print('seg_gt_flat', seg_gt_flat.dtype, seg_gt_flat.shape)


            g1_output = G1(original_image)
            G1_loss = criterion_g_data(g1_output, seg_gt_flat)  # L_data1(G1)  # needs longs for seg_gt
            L_data1 = G1_loss

            # print('original_image', original_image.dtype, original_image.shape)
            # print('g1_output', g1_output.dtype, g1_output.shape)
            # print('seg_gt', seg_gt.dtype, seg_gt.shape)
            g1_pred_cat = torch.cat((original_image, g1_output), 1)
            g1_gt_cat = torch.cat((original_image, seg_gt), 1)  # seg_gt.float()

            # prob_g1_gt = D1(g1_gt_cat).detach()  # what does detatch do? Prevents backpropogation occuring through new variable
            # print('g1_pred_cat', g1_pred_cat.dtype, g1_pred_cat.shape)
            # print('g1_gt_cat', g1_gt_cat.dtype, g1_gt_cat.shape)
            prob_g1_gt = D1(g1_gt_cat)
            prob_g1_pred = D1(g1_pred_cat)
            # print('prob_g1_gt', prob_g1_gt.dtype, prob_g1_gt.shape)
            # print('prob_g1_pred', prob_g1_pred.dtype, prob_g1_pred.shape)
            # D1_loss = criterion_d(prob_g1_pred, prob_g1_gt)  # correct? loss b/w predictions != loss b/w pred and correct label

            REAL_t = torch.full((prob_g1_gt.shape), REAL, device=device)  # tensor of REAL labels
            FAKE_t = torch.full((prob_g1_gt.shape), FAKE, device=device)  # tensor of FAKE labels
            # print('REAL_t', REAL_t.dtype, REAL_t.shape)
            # print('FAKE_t', FAKE_t.dtype, FAKE_t.shape)

            D1_loss = criterion_d(prob_g1_pred, FAKE_t) + criterion_d(prob_g1_gt, REAL_t)  # call loss.backward on both individually? or together is ok?
            G1_adv_loss = criterion_d(prob_g1_pred, REAL_t)
            # L_cgan(G1, D1) = D1_loss + G1_adv_loss
            L_cgan1 = D1_loss + G1_adv_loss

            # g2_input = torch.cat((original_image, shadow_mask), 1)
            g2_input = g1_gt_cat
            # print('g2_input', g2_input.dtype, g2_input.shape)
            g2_output = G2(g2_input)
            # print('g1_output', g1_output.dtype, g1_output.shape)
            # print('seg_gt', seg_gt.dtype, seg_gt.shape)
            seg_edges_gt = get_edges(g1_output, seg_gt)
            seg_edges_gt_flat = seg_edges_gt.argmax(dim=1).squeeze(1).long()
            G2_loss = criterion_g_data(g2_output, seg_edges_gt_flat)  #L_data2(G2|G1)
            L_data2 = G2_loss

            # edges_if_seg_perfect = get_edges(seg_gt, seg_gt)  # torch.zeroes faster, but don't know h, w, BATCHSIZE
            # we want the d2 loss to capture the higher-order properties of the edges (that they are connected units)
            # this is done by comparing real (coherent parts) edges and fake (possibly fuzzy or scattered/inconsistent)
            # the g1_output and original_image are just extra inputs to help line up edges, but the focus is on the edges.
            # as in, given input, does the output look real? NOT given different inputs, does the output look real?

            g2_gt_cat = torch.cat((original_image, g1_output, seg_edges_gt), 1)
            g2_pred_cat = torch.cat((original_image, g1_output, g2_output), 1)

            # prob_g2_gt = D2(g2_gt_cat).detach()
            prob_g2_gt = D2(g2_gt_cat)
            prob_g2_pred = D2(g2_pred_cat)
            # D2_loss = criterion_d(prob_g2_pred, prob_g2_gt)
            D2_loss = criterion_d(prob_g2_pred, FAKE_t) + criterion_d(prob_g2_gt, REAL_t)
            G2_adv_loss = criterion_d(prob_g2_pred, REAL_t)
            L_cgan2 = D2_loss + G2_adv_loss

            # loss = G1_loss + lambda1 * G2_loss + lambda2 * D1_loss + lambda3 * D2_loss
            loss = L_data1 + lambda1 * L_data2 + lambda2 * L_cgan1 + lambda3 * L_cgan2

            if epoch % 6 < 3:
                optimizer_d.zero_grad()
                loss.backward()  # only uses D1_loss and D2_loss?
                optimizer_d.step()
            else:
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

            if iters % 10 == 0:
                print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
                epochs.append(epoch)
                total_losses.append(loss.item())
                L_data1_losses.append(L_data1.item())
                L_data2_losses.append(L_data2.item())
                L_cgan1_losses.append(L_cgan1.item())
                D1_losses.append(D1_loss.item())
                G1_adv_losses.append(G1_adv_loss.item())
                L_cgan2_losses.append(L_cgan2.item())
                D2_losses.append(D2_loss.item())
                G2_adv_losses.append(G2_adv_loss.item())
            iters += 1

        if epoch % 2 == 0:
            generator1_model = os.path.join(path, "saved_models/generator1_%d.pkl" % epoch)
            generator2_model = os.path.join(path, "saved_models/generator2_%d.pkl" % epoch)
            discriminator1_model = os.path.join(path, "saved_models/discriminator1_%d.pkl" % epoch)
            discriminator2_model = os.path.join(path, "saved_models/discriminator2_%d.pkl" % epoch)
            torch.save(G1.state_dict(), generator1_model)
            torch.save(G2.state_dict(), generator2_model)
            torch.save(D1.state_dict(), discriminator1_model)
            torch.save(D2.state_dict(), discriminator2_model)

        if epoch % 5 == 0:
            df = pd.DataFrame(list(zip(*[epochs, total_losses, L_data1_losses, L_data2_losses, L_cgan1_losses, D1_losses, G1_adv_losses, L_cgan2_losses, D2_losses, G2_adv_losses]))).add_prefix('Col')
            # [total_losses, L_data1_losses, L_data2_losses, L_cgan1_losses, D1_losses, G1_adv_losses, L_cgan2_losses, D2_losses, G2_adv_losses]
            filename = path + 'saved_losses/G1D1G2D2_e' + epoch + '_' + time_begin + '.csv'
            print('saving to', filename)
            df.to_csv(filename, index=False)

            print(df)


single_gpu_train()


def train_G1():
    # trains only G1 with CE loss, no discriminators
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataset = CityscapesLoader('val')
    val_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)

    # criterion_g = torch.nn.L1Loss(size_average=False)
    criterion_g = torch.nn.CrossEntropyLoss()

    optimizer_g = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
    for epoch in range(0, 100):
        for i, data in enumerate(train_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)

            g1_output = G1(original_image)
            G1_loss = criterion_g(g1_output, seg_gt)
            optimizer_g.zero_grad()
            G1_loss.backward()
            optimizer_g.step()
            if i % 10 == 0:
                print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(G1_loss)))

        # if epoch % 100 == 99:
        # save every epoch
        generator1_model = os.path.join("model/generator1_%d.pkl" % epoch)
        torch.save(G1.state_dict(), generator1_model)

        running_loss = 0
        for i, data in enumerate(val_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)
            g1_output = G1(original_image)
            G1_loss = criterion_g(g1_output, seg_gt)
            running_loss += G1_loss.item()
            if i % 10 == 0:
                print('Epoch: %d | iter: %d | val loss: %.10f | running_loss: %.10f' % (epoch, i, float(G1_loss), float(running_loss)))


train_G1()

def train_G1D1():
    # trains G1 along with D1
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)
    D1 = Discriminator_first().to(device)

    criterion_g = torch.nn.CELoss(size_average=False)
    criterion_d = torch.nn.BCELoss(size_average=False)
    optimizer_g = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
    optimizer_d = torch.optim.Adam([{'params': D1.parameters()}], lr=0.001)

    for epoch in range(100000):
        for i, data in enumerate(train_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)

            g1_output = G1(original_image)
            g1 = torch.cat((original_image, g1_output), 1)
            gt1 = torch.cat((original_image, seg_gt), 1)

            G1_loss = criterion_g(g1_output, seg_gt)
            D1_loss = criterion_d(prob_g1, prob_gt1)




def train_G1D1_G2D2():
    # trains all
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    # unnecessary method since single_gpu_train() trains all


