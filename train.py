from image_loader import CityscapesLoader
from image_loader import get_edges
from Stcgan_net import *
import torch
import torch.utils.data as Data
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL = 1
FAKE = 0

#hyperparam
BATCH_SIZE = 5
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
    criterion_g_data = torch.nn.CrossEntropyLoss()
    # criterion_g_adv = torch.nn.BCELoss()  # need separate criterion to carry loss only through Gs? No
    optimizer_d = torch.optim.Adam([
        {'params': D1.parameters()},
        {'params': D2.parameters()}], lr=0.001)  # does sharing parameters make sense?
    optimizer_g = torch.optim.Adam([
        {'params': G1.parameters()},
        {'params': G2.parameters()}], lr=0.001)  # does sharing parameters make sense?

    for epoch in range(100000):
        for i, data in enumerate(train_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)


            g1_output = G1(original_image)
            G1_loss = criterion_g_data(g1_output, seg_gt)  # L_data1(G1)
            L_data1 = G1_loss

            g1_pred_cat = torch.cat((original_image, g1_output), 1)
            g1_gt_cat = torch.cat((original_image, seg_gt), 1)

            # prob_g1_gt = D1(g1_gt_cat).detach()  # what does detatch do? Prevents backpropogation occuring through new variable
            prob_g1_gt = D1(g1_gt_cat)
            prob_g1_pred = D1(g1_pred_cat)
            # D1_loss = criterion_d(prob_g1_pred, prob_g1_gt)  # correct? loss b/w predictions != loss b/w pred and correct label

            REAL_t = torch.full((BATCH_SIZE,), REAL, device=device)  # tensor of REAL labels
            FAKE_t = torch.full((BATCH_SIZE,), FAKE, device=device)  # tensor of FAKE labels

            D1_loss = criterion_d(prob_g1_pred, FAKE_t) + criterion_d(prob_g1_gt, REAL_t)  # call loss.backward on both individually? or together is ok?
            G1_adv_loss = criterion_d(prob_g1_pred, REAL_t)
            # L_cgan(G1, D1) = D1_loss + G1_adv_loss
            L_cgan1 = D1_loss + G1_adv_loss

            # g2_input = torch.cat((original_image, shadow_mask), 1)
            g2_input = g1_gt_cat
            g2_output = G2(g2_input)
            seg_edges_gt = get_edges(g1_output, seg_gt)
            G2_loss = criterion_g_data(g2_output, seg_edges_gt)  #L_data2(G2|G1)
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
            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
            if epoch % 6 < 3:
                optimizer_d.zero_grad()
                loss.backward()  # only uses D1_loss and D2_loss?
                optimizer_d.step()
            else:
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

        if epoch % 100 == 99:
            generator1_model = os.path.join("model/generator1_%d.pkl" % epoch)
            generator2_model = os.path.join("model/generator2_%d.pkl" % epoch)
            discriminator1_model = os.path.join("model/discriminator1_%d.pkl" % epoch)
            discriminator2_model = os.path.join("model/discriminator2_%d.pkl" % epoch)
            torch.save(G1.state_dict(), generator1_model)
            torch.save(G2.state_dict(), generator2_model)
            torch.save(D1.state_dict(), discriminator1_model)
            torch.save(D2.state_dict(), discriminator2_model)


# single_gpu_train()


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


