from image_loader import CityscapesLoader, get_edges, iou
from Stcgan_net import *
import torch
import torch.utils.data as Data
import os
import pandas as pd
from datetime import datetime
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:1")
# dtype = torch.cuda.FloatTensor
REAL = 1
FAKE = 0

#hyperparam
BATCH_SIZE = 1  # 4 gpus to use in parallel: can't since need synchronous execution? fix slurm script to request 4
lambda1 = 1  # make higher due to L1Loss being small when most are 0s?
lambda2 = 1
lambda3 = 1


def single_gpu_train_G1D1():
    train_dataset = CityscapesLoader('train')
    # train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    sampler595 = torch.utils.data.SubsetRandomSampler(range(0, 595))  # 1/5 the 2975 train images
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler595)


    val_dataset = CityscapesLoader('val')
    # val_sampler = torch.utils.data.RandomSampler(val_dataset)
    # val_data_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    sampler100 = torch.utils.data.SubsetRandomSampler(range(0, 100))  # 1/5 the 500 val images
    val_data_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=sampler100)

    G1 = Generator_first().to(device)  # .cuda(0)  # nn.DataParallel? use parllel.DistributedDataParallel?
    # G2 = Generator_second().to(device)  # .cuda(0)
    D1 = Discriminator_first().to(device)  # .cuda(1)
    # D2 = Discriminator_second().to(device)  # .cuda(1)
    criterion_d = torch.nn.BCELoss()
    criterion_g1_data = torch.nn.NLLLoss()  # since CrossEntopyLoss includes log_softmax
    # criterion_g2_data = torch.nn.L1Loss()
    # 2 optimizers used to update discriminators and generators in alternating fashion
    optimizer_d = torch.optim.Adam([
        {'params': D1.parameters()}], lr=0.001)  # optimizer updates these parameters with step()
    optimizer_g = torch.optim.Adam([
        {'params': G1.parameters()}], lr=0.001)

    # Lists to keep track of progress
    # img_seg_list = []  # store some samples to visually inspect progress

    # store sample every 25 iterations
    # iters = 0
    epochs = []
    iou_scores = []
    total_losses = []
    D_losses = []
    G_losses = []
    L_data1_losses = []
    # L_data2_losses = []
    L_cgan1_losses = []
    D1_losses = []
    G1_adv_losses = []
    # L_cgan2_losses = []
    # D2_losses = []
    # G2_adv_losses = []

    val_epochs = []
    val_iou_scores = []
    val_total_losses = []
    val_D_losses = []
    val_G_losses = []
    val_L_data1_losses = []
    # val_L_data2_losses = []
    val_L_cgan1_losses = []
    val_D1_losses = []
    val_G1_adv_losses = []
    # val_L_cgan2_losses = []
    # val_D2_losses = []
    # val_G2_adv_losses = []

    # stores average of metrics for each epoch during training
    ave_epochs = []  # stores epoch # once, where epochs[] stores same epoch several times
    ave_iou_scores = []
    ave_total_losses = []
    ave_D_losses = []
    ave_G_losses = []
    ave_L_data1_losses = []
    # ave_L_data2_losses = []
    ave_L_cgan1_losses = []
    ave_D1_losses = []
    ave_G1_adv_losses = []
    # ave_L_cgan2_losses = []
    # ave_D2_losses = []
    # ave_G2_adv_losses = []

    val_ave_epochs = []
    val_ave_iou_scores = []
    val_ave_total_losses = []
    val_ave_D_losses = []
    val_ave_G_losses = []
    val_ave_L_data1_losses = []
    # val_ave_L_data2_losses = []
    val_ave_L_cgan1_losses = []
    val_ave_D1_losses = []
    val_ave_G1_adv_losses = []
    # val_ave_L_cgan2_losses = []
    # val_ave_D2_losses = []
    # val_ave_G2_adv_losses = []


    time_begin = str(datetime.now()).replace(' ', '-')
    path = '/work/LAS/jannesar-lab/mburke/SegEdgeGAN/saved/' + time_begin + '/'
    os.mkdir(path)
    print("beginning training at " + time_begin)

    for epoch in range(100000):
        for mode in ['train', 'val']:
            if mode == 'train':
                torch.set_grad_enabled(True)
                data_loader = train_data_loader
                G1.train()
                # G2.train()
                D1.train()
                # D2.train()
            elif mode == 'val':
                torch.set_grad_enabled(False)
                data_loader = val_data_loader
                G1.eval()
                # G2.eval()
                D1.eval()
                # D2.eval()

            run_iou_score = 0
            run_loss = 0
            run_D_loss = 0
            run_G_loss = 0
            run_L_data1 = 0
            # run_L_data2 = 0
            run_L_cgan1 = 0
            run_D1_loss = 0
            run_G1_adv_loss = 0
            # run_L_cgan2 = 0
            # run_D2_loss = 0
            # run_G2_adv_loss = 0

            for i, data in enumerate(data_loader):
                # get batch of data
                original_image, seg_gt = data
                original_image = original_image.to(device)
                seg_gt = seg_gt.to(device)
                # seg_gt_flat = seg_gt.argmax(axis=1).squeeze(1).long().to(device)  # squeeze redundant?
                seg_gt_flat = seg_gt.argmax(axis=1).long().to(device)  # needed for NLLLoss

                # predict segmentation map with G1
                g1_output = G1(original_image)
                # measures how well G1 predicted segmentation map
                L_data1 = criterion_g1_data(torch.log(g1_output), seg_gt_flat)  # log of softmax values produces input of [-inf, 0] for NLLoss
                # del(seg_gt_flat)

                # Intersection over Union is measure of segmentation map accuracy
                iou_score = iou(g1_output, seg_gt)  # what we ultimately want to improve

                # prepare FAKE/generated and REAL/ground truth input for D1
                g1_pred_cat = torch.cat((original_image, g1_output), 1).to(device)  # FAKE
                g1_gt_cat = torch.cat((original_image, seg_gt), 1).to(device)  # REAL
                # del(original_image)

                # predict probability that input is FAKE or REAL with D1
                prob_g1_gt = D1(g1_gt_cat.detach())  # good D1 would predict REAL
                prob_g1_pred = D1(g1_pred_cat.detach())  # good D1 would predict FAKE
                prob_g1_pred_adv = D1(g1_pred_cat)  # good D1 would predict FAKE, NOT detached

                # get tensors of labels to compute loss for D1
                REAL_t = torch.full((prob_g1_gt.shape), REAL, device=device)  # tensor of REAL labels
                FAKE_t = torch.full((prob_g1_gt.shape), FAKE, device=device)  # tensor of FAKE labels

                # D1 tries to accurately predict FAKE or REAL (ing + seg), but this gets harder as G1 improves
                D1_loss = criterion_d(prob_g1_pred, FAKE_t) + criterion_d(prob_g1_gt, REAL_t)
                # del(prob_g1_gt)

                # G1 should produce output that D1 thinks is REAL. G1_adv_loss updates G1 params to improve G1
                G1_adv_loss = criterion_d(prob_g1_pred_adv, REAL_t)
                # del(prob_g1_pred)
                # L_cgan(G1, D1) = D1_loss + G1_adv_loss
                L_cgan1 = D1_loss + G1_adv_loss

                # g2_input = g1_gt_cat  # found mistake! Can't predict edges from gt since no overlap pattern
                # g2_output = G2(g2_input)

                # G2 predicts edges where G1 prediction does not match with ground truth
                # As G1 improves, edges get smaller, changing input to G2, but patterns to recognize are similar
                # g2_output = G2(g1_pred_cat.to(device))

                # find edges where G1 prediction does not match with ground truth
                # seg_edges_gt = get_edges(g1_output, seg_gt).to(device)  # to(device)? new tensor created outside of model
                # del (seg_gt)
                # del (g1_output)
                # reformat for BCELoss input
                # seg_edges_gt_flat = seg_edges_gt.argmax(dim=1).long().to(device)  # squeeze doesn't do anything

                # measure how well G2 predicted edges
                # L_data2 = criterion_g2_data(g2_output, seg_edges_gt)  #L_data2(G2|G1)
                # del(seg_edges_gt_flat)

                # we want the d2 loss to capture the higher-order properties of the edges (that they are connected units)
                # this is done by comparing real (coherent parts) edges and fake (possibly fuzzy or scattered/inconsistent)
                # since L_data_2 loss alone may produce fuzzy/inconsistent parts

                # prepare FAKE/(predicted by G2) and REAL/(as if G2 were perfect) input for D2
                # g2_gt_cat = torch.cat((g1_pred_cat, seg_edges_gt), 1).to(device)  # REAL
                # g2_pred_cat = torch.cat((g1_pred_cat, g2_output), 1).to(device)  # FAKE

                # predict the probability that input is REAL or FAKE with D2
                # prob_g2_gt = D2(g2_gt_cat.detach())  # good D2 would predict REAL
                # prob_g2_pred = D2(g2_pred_cat.detach())  # good D2 would predict FAKE
                # prob_g2_pred_adv = D2(g2_pred_cat)  # good D2 would predict FAKE, NOT detached to carry gradient to G2

                # measure how well D2 predicts REAL or FAKE
                # D2_loss = criterion_d(prob_g2_pred, FAKE_t) + criterion_d(prob_g2_gt, REAL_t)
                # we want G2 to produce output that D2 thinks is REAL
                # G2_adv_loss = criterion_d(prob_g2_pred_adv, REAL_t)
                # del(prob_g2_pred)
                # del(prob_g2_gt)
                # del(FAKE_t)
                # del(REAL_t)
                # L_cgan2 = D2_loss + G2_adv_loss

                # lambda1 = 5, lambda2 = 0.1, lambda3 = 0.1 in paper (set at top)
                # loss = L_data1 + lambda1 * L_data2 + lambda2 * L_cgan1 + lambda3 * L_cgan2
                loss = L_data1 + lambda2 * L_cgan1

                # G_loss = L_data1 + lambda1 * L_data2 + lambda2 * G1_adv_loss + lambda3 * G2_adv_loss
                # D_loss = lambda2 * D1_loss + lambda3 * D2_loss
                G_loss = L_data1 + lambda2 * G1_adv_loss
                D_loss = lambda2 * D1_loss

                #val_epoch = epoch
                run_iou_score += iou_score
                run_loss += loss.item()  # or need to sum all loss.items in epoch / len(data_loader) ?
                run_D_loss += D_loss.item()
                run_G_loss += G_loss.item()
                run_L_data1 += L_data1.item()
                # run_L_data2 += L_data2.item()
                run_L_cgan1 += L_cgan1.item()
                run_D1_loss += D1_loss.item()
                run_G1_adv_loss += G1_adv_loss.item()
                # run_L_cgan2 += L_cgan2.item()
                # run_D2_loss += D2_loss.item()
                # run_G2_adv_loss += G2_adv_loss.item()

                if mode == 'train':
                    if epoch % 2 < 1:
                        # optimizer_g.zero_grad()
                        optimizer_d.zero_grad()  # clears previous gradients (from previous loss.backward() calls)
                        loss.backward()  # computes derivatives of loss (aka gradients)
                        # D_loss.backward()
                        optimizer_d.step()  # adjusts model parameters based on gradients
                    else:
                        # optimizer_d.zero_grad()  # clear both to prevent buildup? as long as zero_grad before step, doesn't matter
                        optimizer_g.zero_grad()
                        loss.backward()  # clears graph for all computations to make loss
                        # G_loss.backward()
                        optimizer_g.step()

                    # store finer points for graphing training
                    if i % 25 == 0:
                        # loss on each item is good enough sample to graph, but could also add average loss for epoch
                        print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
                        epochs.append(epoch)
                        iou_scores.append(iou_score)
                        total_losses.append(loss.item())
                        D_losses.append(D_loss.item())
                        G_losses.append(G_loss.item())
                        L_data1_losses.append(L_data1.item())
                        # L_data2_losses.append(L_data2.item())
                        L_cgan1_losses.append(L_cgan1.item())
                        D1_losses.append(D1_loss.item())
                        G1_adv_losses.append(G1_adv_loss.item())
                        # L_cgan2_losses.append(L_cgan2.item())
                        # D2_losses.append(D2_loss.item())
                        # G2_adv_losses.append(G2_adv_loss.item())

                elif mode == 'val':
                    if i % 25 == 0:
                        # loss on each item is good enough sample to graph, but could also add average loss for epoch
                        print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
                        val_epochs.append(epoch)
                        val_iou_scores.append(iou_score)
                        val_total_losses.append(loss.item())
                        val_D_losses.append(D_loss.item())
                        val_G_losses.append(G_loss.item())
                        val_L_data1_losses.append(L_data1.item())
                        # val_L_data2_losses.append(L_data2.item())
                        val_L_cgan1_losses.append(L_cgan1.item())
                        val_D1_losses.append(D1_loss.item())
                        val_G1_adv_losses.append(G1_adv_loss.item())
                        # val_L_cgan2_losses.append(L_cgan2.item())
                        # val_D2_losses.append(D2_loss.item())
                        # val_G2_adv_losses.append(G2_adv_loss.item())

                # iters += 1

            # we've completed one epoch
            # save losses and iou every epoch for graphing
            if mode == 'train':
                # save every 25 iterations points
                df = pd.DataFrame(list(zip(*[epochs, iou_scores, total_losses, D_losses, G_losses, L_data1_losses,
                                             L_cgan1_losses, D1_losses, G1_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)

                # save averages per epoch
                ave_epochs.append(epoch)
                ave_iou_scores.append(run_iou_score / len(train_data_loader))
                ave_total_losses.append(run_loss / len(train_data_loader))
                ave_D_losses.append(run_D_loss / len(train_data_loader))
                ave_G_losses.append(run_G_loss / len(train_data_loader))
                ave_L_data1_losses.append(run_L_data1 / len(train_data_loader))
                # ave_L_data2_losses.append(run_L_data2 / len(train_data_loader))
                ave_L_cgan1_losses.append(run_L_cgan1 / len(train_data_loader))
                ave_D1_losses.append(run_D1_loss / len(train_data_loader))
                ave_G1_adv_losses.append(run_G1_adv_loss / len(train_data_loader))
                # ave_L_cgan2_losses.append(run_L_cgan2 / len(train_data_loader))
                # ave_D2_losses.append(run_D2_loss / len(train_data_loader))
                # ave_G2_adv_losses.append(run_G2_adv_loss / len(train_data_loader))

                # saves lists of average (per epoch) losses
                df = pd.DataFrame(list(zip(
                    *[ave_epochs, ave_iou_scores, ave_total_losses, ave_D_losses, ave_G_losses, ave_L_data1_losses,
                      ave_L_cgan1_losses, ave_D1_losses, ave_G1_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1_ave_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)
                del(df)

            elif mode == 'val':
                # not important to gather samples every 25 iterations since networks not updating
                # save every 25 iterations points
                df = pd.DataFrame(list(zip(*[val_epochs, val_iou_scores, val_total_losses, val_D_losses, val_G_losses,
                                             val_L_data1_losses, val_L_cgan1_losses, val_D1_losses,
                                             val_G1_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1_val_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)

                # save averages per epoch
                val_ave_epochs.append(epoch)
                val_ave_iou_scores.append(run_iou_score / len(val_data_loader))
                val_ave_total_losses.append(run_loss / len(val_data_loader))
                val_ave_D_losses.append(run_D_loss / len(val_data_loader))
                val_ave_G_losses.append(run_G_loss / len(val_data_loader))
                val_ave_L_data1_losses.append(run_L_data1 / len(val_data_loader))
                # val_ave_L_data2_losses.append(run_L_data2 / len(val_data_loader))
                val_ave_L_cgan1_losses.append(run_L_cgan1 / len(val_data_loader))
                val_ave_D1_losses.append(run_D1_loss / len(val_data_loader))
                val_ave_G1_adv_losses.append(run_G1_adv_loss / len(val_data_loader))
                # val_ave_L_cgan2_losses.append(run_L_cgan2 / len(val_data_loader))
                # val_ave_D2_losses.append(run_D2_loss / len(val_data_loader))
                # val_ave_G2_adv_losses.append(run_G2_adv_loss / len(val_data_loader))

                # saves lists of average (per epoch) losses
                df = pd.DataFrame(list(zip(
                    *[val_ave_epochs, val_ave_iou_scores, val_ave_total_losses, val_ave_D_losses, val_ave_G_losses,
                      val_ave_L_data1_losses, val_ave_L_cgan1_losses, val_ave_D1_losses,
                      val_ave_G1_adv_losses]))).add_prefix('Col')
                filename = path + 'G1D1_val_ave_e' + str(epoch) + '_' + time_begin + '.csv'
                print('saving to', filename)
                df.to_csv(filename, index=False)
                del(df)

        # will use best model for test set
        # outside of modes since only needs to be done once per epoch
        if epoch > 0:
            generator1_model = os.path.join(path, "generator1_%d.pkl" % epoch)
            # generator2_model = os.path.join(path, "generator2_%d.pkl" % epoch)
            discriminator1_model = os.path.join(path, "discriminator1_%d.pkl" % epoch)
            # discriminator2_model = os.path.join(path, "discriminator2_%d.pkl" % epoch)
            torch.save(G1.state_dict(), generator1_model)
            # torch.save(G2.state_dict(), generator2_model)
            torch.save(D1.state_dict(), discriminator1_model)
            # torch.save(D2.state_dict(), discriminator2_model)

        print("finished epoch " + str(epoch) + " at " + str(datetime.now()))

    print("finished all epochs")


single_gpu_train_G1D1()

