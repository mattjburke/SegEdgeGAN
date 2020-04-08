from image_loader import CityscapesLoader
from Stcgan_net import *
import torch
import torch.utils.data as Data
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparam
BATCH_SIZE = 5
lambda1 = 5
lambda2 = 0.1
lambda3 = 0.1


def single_gpu_train():
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)
    G2 = Generator_second().to(device)
    D1 = Discriminator_first().to(device)
    D2 = Discriminator_second().to(device)


    criterion1 = torch.nn.BCELoss(size_average=False)
    criterion2 = torch.nn.L1Loss()
    optimizerd = torch.optim.Adam([
        {'params': D1.parameters()},
        {'params': D2.parameters()}], lr=0.001)
    optimizerg = torch.optim.Adam([
        {'params': G1.parameters()},
        {'params': G2.parameters()}], lr=0.001)

    for epoch in range(100000):
        for i, data in enumerate(train_data_loader):
            original_image, shadow_mask, shadow_free_image = data
            original_image = original_image.to(device)
            shadow_mask = shadow_mask.to(device)
            shadow_free_image = shadow_free_image.to(device)

            g1_output = G1(original_image)
            g1 = torch.cat((original_image, g1_output), 1)
            gt1 = torch.cat((original_image, shadow_mask), 1)

            prob_gt1 = D1(gt1).detach()
            prob_g1 = D1(g1)

            #D1_loss = -torch.mean(torch.log(prob_gt1) +  torch.log(1 - prob_g1))
            #G1_loss = torch.mean(torch.log(shadow_mask - g1_output))
            D1_loss = criterion1(prob_g1, prob_gt1)
            G1_loss = criterion2(g1_output, shadow_mask)

            g2_input = torch.cat((original_image, shadow_mask), 1)
            g2_output = G2(g2_input)

            gt2 = torch.cat((original_image, shadow_mask, shadow_free_image), 1)
            g2 = torch.cat((original_image, g1_output, g2_output), 1)

            prob_gt2 = D2(gt2).detach()
            prob_g2 = D2(g2)

            #D2_loss = -torch.mean(torch.log(prob_gt2) + torch.log(1 - prob_g2))
            #G2_loss = torch.mean(torch.log(shadow_free_image, g2_output))
            D2_loss = criterion1(prob_g2, prob_gt2)
            G2_loss = criterion2(g2_output, shadow_free_image)

            loss = G1_loss + lambda1 * G2_loss + lambda2 * D1_loss + lambda3 * D2_loss
            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
            if epoch % 2000 < 1000:
                optimizerd.zero_grad()
                loss.backward()
                optimizerd.step()
            else:
                optimizerg.zero_grad()
                loss.backward()
                optimizerg.step()

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

    criterion_g = torch.nn.CELoss(size_average=False)

    optimizerg = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
    for epoch in range(0, 100):
        for i, data in enumerate(train_data_loader):
            original_image, seg_gt = data
            original_image = original_image.to(device)
            seg_gt = seg_gt.to(device)

            g1_output = G1(original_image)
            G1_loss = criterion_g(g1_output, seg_gt)
            optimizerg.zero_grad()
            G1_loss.backward()
            optimizerg.step()
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



def train_G1D1():
    # trains G1 along with D1
    train_dataset = CityscapesLoader('train')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)
    D1 = Discriminator_first().to(device)

    criterion_g = torch.nn.CELoss(size_average=False)
    criterion_d = torch.nn.BCELoss(size_average=False)
    optimizerg = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
    optimizerd = torch.optim.Adam([{'params': D1.parameters()}], lr=0.001)

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


