from image_loader import CityscapesLoader, get_edges, iou
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

    val_dataset = CityscapesLoader('val')
    val_data_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    G1 = Generator_first().to(device)
    G2 = Generator_second().to(device)
    D1 = Discriminator_first().to(device)
    D2 = Discriminator_second().to(device)

    G1.train()
    G2.train()
    D1.train()
    D2.train()


    criterion_d = torch.nn.BCELoss()
    criterion_g_data = torch.nn.NLLLoss()  # since CrossEntopyLoss includes softmax
    # 2 optimizers used to update discriminators and generators in alternating fashion
    optimizer_d = torch.optim.Adam([
        {'params': D1.parameters()},
        {'params': D2.parameters()}], lr=0.001)  # optimizer updates these parameters with step()
    optimizer_g = torch.optim.Adam([
        {'params': G1.parameters()},
        {'params': G2.parameters()}], lr=0.001)

    # Lists to keep track of progress
    img_seg_list = []  # store some samples to visually inspect progress
    iters = 0
    # stores sample every 25 iterations during training
    epochs = []
    iou_scores = []
    total_losses = []
    L_data1_losses = []
    L_data2_losses = []
    L_cgan1_losses = []
    D1_losses = []
    G1_adv_losses = []
    L_cgan2_losses = []
    D2_losses = []
    G2_adv_losses = []

    # stores average of metrics for each epoch during training
    ave_epochs = []  # stores epoch # once, where epochs[] stores same epoch several times since 119 measuerements taken each epoch
    ave_iou_scores = []
    ave_total_losses = []
    ave_L_data1_losses = []
    ave_L_data2_losses = []
    ave_L_cgan1_losses = []
    ave_D1_losses = []
    ave_G1_adv_losses = []
    ave_L_cgan2_losses = []
    ave_D2_losses = []
    ave_G2_adv_losses = []

    # stores average of metrics during each validation epoch
    val_epochs = []  # stores epoch # once, where epochs[] stores same epoch several times since 119 measuerements taken each epoch
    val_iou_scores = []
    val_total_losses = []
    val_L_data1_losses = []
    val_L_data2_losses = []
    val_L_cgan1_losses = []
    val_D1_losses = []
    val_G1_adv_losses = []
    val_L_cgan2_losses = []
    val_D2_losses = []
    val_G2_adv_losses = []

    time_begin = str(datetime.now()).replace(' ', '-')
    path = '/work/LAS/jannesar-lab/mburke/SegEdgeGAN/saved/' + time_begin + '/'
    os.mkdir(path)
    print("beginning training at " + time_begin)

    for epoch in range(100000):
        for mode in ['train', 'val']:
            if mode == 'train':
                data_loader = train_data_loader
                G1.train()
                G2.train()
                D1.train()
                D2.train()
            elif mode == 'val':
                data_loader = val_data_loader
                G1.eval()
                G2.eval()
                D1.eval()
                D2.eval()

            run_iou_score = 0
            run_loss = 0
            run_L_data1 = 0
            run_L_data2 = 0
            run_L_cgan1 = 0
            run_D1_loss = 0
            run_G1_adv_loss = 0
            run_L_cgan2 = 0
            run_D2_loss = 0
            run_G2_adv_loss = 0

            for i, data in enumerate(data_loader):
                # get batch of data
                original_image, seg_gt = data
                original_image = original_image.to(device)
                seg_gt = seg_gt.to(device)
                seg_gt_flat = seg_gt.argmax(axis=1).squeeze(1).long()  # needed for NLLLoss

                # predict segmentation map with G1
                g1_output = G1(original_image)
                # measures how well G1 predicted segmentation map
                L_data1 = criterion_g_data(g1_output, seg_gt_flat)

                # Intersection over Union is measure of segmentation map accuracy
                iou_score = iou(g1_output, seg_gt)  # what we ultimately want to improve

                # prepare FAKE/generated and REAL/ground truth input for D1
                g1_pred_cat = torch.cat((original_image, g1_output), 1)  # FAKE
                g1_gt_cat = torch.cat((original_image, seg_gt), 1)  # REAL

                # predict probability that input is FAKE or REAL with D1
                prob_g1_gt = D1(g1_gt_cat)  # good D1 would predict REAL
                prob_g1_pred = D1(g1_pred_cat)  # good D1 would predict FAKE

                # get tensors of labels to compute loss for D1
                REAL_t = torch.full((prob_g1_gt.shape), REAL, device=device)  # tensor of REAL labels
                FAKE_t = torch.full((prob_g1_gt.shape), FAKE, device=device)  # tensor of FAKE labels

                # D1 tries to accurately predict FAKE or REAL (ing + seg), but this gets harder as G1 improves
                D1_loss = criterion_d(prob_g1_pred, FAKE_t) + criterion_d(prob_g1_gt, REAL_t)

                # G1 should produce output that D1 thinks is REAL. G1_adv_loss updates G1 params to improve G1
                G1_adv_loss = criterion_d(prob_g1_pred, REAL_t)
                # L_cgan(G1, D1) = D1_loss + G1_adv_loss
                L_cgan1 = D1_loss + G1_adv_loss

                # g2_input = g1_gt_cat  # found mistake! Can't predict edges from gt since no overlap pattern
                # g2_output = G2(g2_input)

                # G2 predicts edges where G1 prediction does not match with ground truth
                # As G1 improves, edges get smaller, changing input to G2, but patterns to recognize are similar
                g2_output = G2(g1_pred_cat)

                # find edges where G1 prediction does not match with ground truth
                seg_edges_gt = get_edges(g1_output, seg_gt)
                # reformat for BCELoss input
                seg_edges_gt_flat = seg_edges_gt.argmax(dim=1).squeeze(1).long()

                # measure how well G2 predicted edges
                L_data2 = criterion_g_data(g2_output, seg_edges_gt_flat)  #L_data2(G2|G1)

                # we want the d2 loss to capture the higher-order properties of the edges (that they are connected units)
                # this is done by comparing real (coherent parts) edges and fake (possibly fuzzy or scattered/inconsistent)
                # since L_data_2 loss alone may produce fuzzy/inconsistent parts

                # prepare FAKE/(predicted by G2) and REAL/(as if G2 were perfect) input for D2
                g2_gt_cat = torch.cat((g1_pred_cat, seg_edges_gt), 1)  # REAL
                g2_pred_cat = torch.cat((g1_pred_cat, g2_output), 1)  # FAKE

                # predict the probability that input is REAL or FAKE with D2
                prob_g2_gt = D2(g2_gt_cat)  # good D2 would predict REAL
                prob_g2_pred = D2(g2_pred_cat)  # good D2 would predict FAKE

                # measure how well D2 predicts REAL or FAKE
                D2_loss = criterion_d(prob_g2_pred, FAKE_t) + criterion_d(prob_g2_gt, REAL_t)
                # we want G2 to produce output that D2 thinks is REAL
                G2_adv_loss = criterion_d(prob_g2_pred, REAL_t)
                L_cgan2 = D2_loss + G2_adv_loss

                # lambda1 = 5, lambda2 = 0.1, lambda3 = 0.1 in paper (set at top)
                loss = L_data1 + lambda1 * L_data2 + lambda2 * L_cgan1 + lambda3 * L_cgan2

                #val_epoch = epoch
                run_iou_score += iou_score
                run_loss += loss.item()  # or need to sum all loss.items in epoch / len(data_loader) ?
                run_L_data1 += L_data1.item()
                run_L_data2 += L_data2.item()
                run_L_cgan1 += L_cgan1.item()
                run_D1_loss += D1_loss.item()
                run_G1_adv_loss += G1_adv_loss.item()
                run_L_cgan2 += L_cgan2.item()
                run_D2_loss += D2_loss.item()
                run_G2_adv_loss += G2_adv_loss.item()

                if mode == 'train':
                    if epoch % 6 < 3:
                        optimizer_d.zero_grad()  # clears previous gradients (from previous loss.backward() calls)
                        loss.backward()  # computes derivatives of loss (aka gradients)
                        optimizer_d.step()  # adjusts model parameters based on gradients
                    else:
                        optimizer_g.zero_grad()
                        loss.backward()
                        optimizer_g.step()

                    # store finer points for graphing training
                    if iters % 25 == 0:
                        # loss on each item is good enough sample to graph, but could also add average loss for epoch
                        print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
                        epochs.append(epoch)
                        iou_scores.append(iou_score)
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

            # we've completed one epoch
            # save losses and iou every epoch for graphing
            if mode == 'train':
                # save every 25 iterations points
                df = pd.DataFrame(list(zip(*[epochs, iou_scores, total_losses, L_data1_losses, L_data2_losses, L_cgan1_losses, D1_losses, G1_adv_losses, L_cgan2_losses, D2_losses, G2_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1G2D2_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)

                # save averages per epoch
                ave_epochs.append(epoch)
                ave_iou_scores.append(run_iou_score / len(train_data_loader))
                ave_total_losses.append(run_loss / len(train_data_loader))
                ave_L_data1_losses.append(run_L_data1 / len(train_data_loader))
                ave_L_data2_losses.append(run_L_data2 / len(train_data_loader))
                ave_L_cgan1_losses.append(run_L_cgan1 / len(train_data_loader))
                ave_D1_losses.append(run_D1_loss / len(train_data_loader))
                ave_G1_adv_losses.append(run_G1_adv_loss / len(train_data_loader))
                ave_L_cgan2_losses.append(run_L_cgan2 / len(train_data_loader))
                ave_D2_losses.append(run_D2_loss / len(train_data_loader))
                ave_G2_adv_losses.append(run_G2_adv_loss / len(train_data_loader))

                # saves lists of average (per epoch) losses
                df = pd.DataFrame(list(zip(
                    *[ave_epochs, ave_iou_scores, ave_total_losses, ave_L_data1_losses, ave_L_data2_losses,
                      ave_L_cgan1_losses, ave_D1_losses, ave_G1_adv_losses, ave_L_cgan2_losses, ave_D2_losses,
                      ave_G2_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1G2D2_ave_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)

            elif mode == 'val':
                # not important to gather samples every 25 iterations since networks not updating
                # save averages per epoch
                val_epochs.append(epoch)
                val_iou_scores.append(run_iou_score / len(val_data_loader))
                val_total_losses.append(run_loss / len(val_data_loader))
                val_L_data1_losses.append(run_L_data1 / len(val_data_loader))
                val_L_data2_losses.append(run_L_data2 / len(val_data_loader))
                val_L_cgan1_losses.append(run_L_cgan1 / len(val_data_loader))
                val_D1_losses.append(run_D1_loss / len(val_data_loader))
                val_G1_adv_losses.append(run_G1_adv_loss / len(val_data_loader))
                val_L_cgan2_losses.append(run_L_cgan2 / len(val_data_loader))
                val_D2_losses.append(run_D2_loss / len(val_data_loader))
                val_G2_adv_losses.append(run_G2_adv_loss / len(val_data_loader))

                # saves lists of average (per epoch) losses
                df = pd.DataFrame(list(zip(
                    *[val_epochs, val_iou_scores, val_total_losses, val_L_data1_losses, val_L_data2_losses,
                      val_L_cgan1_losses, val_D1_losses, val_G1_adv_losses, val_L_cgan2_losses, val_D2_losses,
                      val_G2_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1G2D2_val_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)

        # will use best model for test set
        # outside of modes since only needs to be done once per epoch
        if epoch > 5:
            generator1_model = os.path.join(path, "generator1_%d.pkl" % epoch)
            generator2_model = os.path.join(path, "generator2_%d.pkl" % epoch)
            discriminator1_model = os.path.join(path, "discriminator1_%d.pkl" % epoch)
            discriminator2_model = os.path.join(path, "discriminator2_%d.pkl" % epoch)
            torch.save(G1.state_dict(), generator1_model)
            torch.save(G2.state_dict(), generator2_model)
            torch.save(D1.state_dict(), discriminator1_model)
            torch.save(D2.state_dict(), discriminator2_model)

        print("finished epoch " + str(epoch) + " at " + str(datetime.now()))

    print("finished all epochs")


single_gpu_train()

#
# def train_G1():
#     # trains only G1 with CE loss, no discriminators
#     train_dataset = CityscapesLoader('train')
#     train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
#     val_dataset = CityscapesLoader('val')
#     val_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
#
#     G1 = Generator_first().to(device)
#
#     # criterion_g = torch.nn.L1Loss(size_average=False)
#     criterion_g = torch.nn.CrossEntropyLoss()
#
#     optimizer_g = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
#     for epoch in range(0, 100):
#         for i, data in enumerate(train_data_loader):
#             original_image, seg_gt = data
#             original_image = original_image.to(device)
#             seg_gt = seg_gt.to(device)
#
#             g1_output = G1(original_image)
#             G1_loss = criterion_g(g1_output, seg_gt)
#             optimizer_g.zero_grad()
#             G1_loss.backward()
#             optimizer_g.step()
#             if i % 10 == 0:
#                 print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(G1_loss)))
#
#         # if epoch % 100 == 99:
#         # save every epoch
#         generator1_model = os.path.join("model/generator1_%d.pkl" % epoch)
#         torch.save(G1.state_dict(), generator1_model)
#
#         running_loss = 0
#         for i, data in enumerate(val_data_loader):
#             original_image, seg_gt = data
#             original_image = original_image.to(device)
#             seg_gt = seg_gt.to(device)
#             g1_output = G1(original_image)
#             G1_loss = criterion_g(g1_output, seg_gt)
#             running_loss += G1_loss.item()
#             if i % 10 == 0:
#                 print('Epoch: %d | iter: %d | val loss: %.10f | running_loss: %.10f' % (epoch, i, float(G1_loss), float(running_loss)))
#
#
# # train_G1()
#
# def train_G1D1():
#     # trains G1 along with D1
#     train_dataset = CityscapesLoader('train')
#     train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
#
#     G1 = Generator_first().to(device)
#     D1 = Discriminator_first().to(device)
#
#     criterion_g = torch.nn.CELoss(size_average=False)
#     criterion_d = torch.nn.BCELoss(size_average=False)
#     optimizer_g = torch.optim.Adam([{'params': G1.parameters()}], lr=0.001)
#     optimizer_d = torch.optim.Adam([{'params': D1.parameters()}], lr=0.001)
#
#     for epoch in range(100000):
#         for i, data in enumerate(train_data_loader):
#             original_image, seg_gt = data
#             original_image = original_image.to(device)
#             seg_gt = seg_gt.to(device)
#
#             g1_output = G1(original_image)
#             g1 = torch.cat((original_image, g1_output), 1)
#             gt1 = torch.cat((original_image, seg_gt), 1)
#
#             G1_loss = criterion_g(g1_output, seg_gt)
#             D1_loss = criterion_d(prob_g1, prob_gt1)
#
#
#
#
# def train_G1D1_G2D2():
#     # trains all
#     train_dataset = CityscapesLoader('train')
#     train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
#     # unnecessary method since single_gpu_train() trains all


