import matplotlib.pyplot as plt
import csv

FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-10-14:36:28.987422/'
FILEPATH_TRAIN = FILE_BASE + 'G1D1G2D2_e9_2020-04-10-14:36:28.987422.csv'
FILEPATH_VAL = FILE_BASE + 'G1D1G2D2_val_e9_2020-04-10-14:36:28.987422.csv'

# lists in training files
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
iters = []  # stores epochs as float for each data point
# sample gathered every 25 iterations = 119 per epoch (2975 training files)

with open(FILEPATH_TRAIN, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    i = 0
    for row in plots:
        epochs.append(int(row['Col0']))
        iou_scores.append(float(row['Col1']))
        total_losses.append(float(row['Col2']))
        L_data1_losses.append(float(row['Col3']))
        L_data2_losses.append(float(row['Col4']))
        L_cgan1_losses.append(float(row['Col5']))
        D1_losses.append(float(row['Col6']))
        G1_adv_losses.append(float(row['Col7']))
        L_cgan2_losses.append(float(row['Col8']))
        D2_losses.append(float(row['Col9']))
        G2_adv_losses.append(float(row['Col10']))
        iters.append(float(i / 119))  # gets epoch
        i += 1

plt.figure()

ax1 = plt.subplot(211)
ax1.set_ylim([-.5, .5])
plt.title('Train Metrics')
# plt.plot(iters, epochs, marker=',')
plt.plot(iters, iou_scores, marker=',')
plt.plot(iters, total_losses, marker=',')
# plt.plot(iters, L_data1_losses, marker=',')
# plt.plot(iters, L_data2_losses, marker=',')
# plt.plot(iters, L_cgan1_losses, marker=',')
# plt.plot(iters, D1_losses, marker=',')
# plt.plot(iters, G1_adv_losses, marker=',')
# plt.plot(iters, L_cgan2_losses, marker=',')
# plt.plot(iters, D2_losses, marker=',')
# plt.plot(iters, G2_adv_losses, marker=',')

plt.ylabel('Train Metric')
plt.xlabel('Epoch')
plt.legend(['iou_scores', 'total_losses'], loc='lower right')

# plt.legend(['epochs', 'iou_scores', 'total_losses', 'L_data1_losses', 'L_data2_losses', 'L_cgan1_losses',
#             'D1_losses', 'G1_adv_losses', 'L_cgan2_losses', 'D2_losses', 'G2_adv_losses'], loc='lower right')

# plt.show()


# lists in validation files
val_epochs = []
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
# only average per epoch is gathered for val set
iters = []  # used to represent epoch
num_points_per_epoch = 119  # use to match graph with training data gathered

with open(FILEPATH_VAL, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    e = 0
    for row in plots:
        val_epochs.append(int(row['Col0']))
        val_iou_scores.append(float(row['Col1']))
        val_total_losses.append(float(row['Col2']))
        val_L_data1_losses.append(float(row['Col3']))
        val_L_data2_losses.append(float(row['Col4']))
        val_L_cgan1_losses.append(float(row['Col5']))
        val_D1_losses.append(float(row['Col6']))
        val_G1_adv_losses.append(float(row['Col7']))
        val_L_cgan2_losses.append(float(row['Col8']))
        val_D2_losses.append(float(row['Col9']))
        val_G2_adv_losses.append(float(row['Col10']))
        iters.append(int(e))  # gets epoch
        e +=1

# plt.figure()


ax2 = plt.subplot(212)
ax2.set_ylim([-.5, .5])
plt.title('Val Metrics')
# plt.plot(iters, val_epochs, marker='o')
plt.plot(iters, val_iou_scores, marker='o')
plt.plot(iters, val_total_losses, marker='o')
# plt.plot(iters, val_L_data1_losses, marker='o')
# plt.plot(iters, val_L_data2_losses, marker='o')
# plt.plot(iters, val_L_cgan1_losses, marker='o')
# plt.plot(iters, val_D1_losses, marker='o')
# plt.plot(iters, val_G1_adv_losses, marker='o')
# plt.plot(iters, val_L_cgan2_losses, marker='o')
# plt.plot(iters, val_D2_losses, marker='o')
# plt.plot(iters, val_G2_adv_losses, marker='o')

plt.ylabel('Val Metric')
plt.xlabel('Epoch')
plt.legend(['val_iou_scores', 'val_total_losses'], loc='lower right')
# plt.legend(['val_iou_scores', 'val_total_losses', 'val_L_data1_losses', 'val_L_data2_losses',
#             'val_L_cgan1_losses', 'val_L_cgan2_losses'], loc='lower right')
# plt.legend(['epochs', 'val_iou_scores', 'val_total_losses', 'val_L_data1_losses', 'val_L_data2_losses',
#             'val_L_cgan1_losses', 'val_D1_losses', 'val_G1_adv_losses', 'val_L_cgan2_losses', 'val_D2_losses',
#             'val_G2_adv_losses'], loc='lower right')

plt.show()












#
# if gan:
#     FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-05/gan_stacked_segnet-2020-04-05-18:41:12.353349/model_history_log.csv'
#     # epoch,accuracy,auc_2,loss,val_accuracy,val_auc_2,val_loss
#     with open(FILEPATH, 'r') as csvfile:
#         plots = csv.DictReader(csvfile)
#         i = 0
#         for row in plots:
#             epoch.append(int(row['epoch']))
#             accuracy.append(float(row['accuracy']))
#             auc_2.append(float(row['auc_2']))
#             loss.append(float(row['loss']))
#             # real_acc.append(float(row['sensitivity_at_specificity']))  # sensitivity_at_specificity
#             # fake_acc.append(float(row['specificity_at_sensitivity']))  # specificity_at_sensitivity
#             val_accuracy.append(float(row['val_accuracy']))
#             val_auc_2.append(float(row['val_auc_2']))
#             val_loss.append(float(row['val_loss']))
#             # val_real_acc.append(float(row['val_sensitivity_at_specificity']))  # sensitivity_at_specificity
#             # val_fake_acc.append(float(row['val_specificity_at_sensitivity']))  # specificity_at_sensitivity
#             cum_epochs.append(i)
#             i += 1
#
#     plt.figure()
#
#     plt.subplot(211)
#     plt.title('GAN Accuracy')
#     plt.plot(cum_epochs, epoch, marker='o')
#     plt.plot(cum_epochs, accuracy, marker='o')
#     plt.plot(cum_epochs, val_accuracy, marker='o')
#     plt.plot(cum_epochs, auc_2, marker='o')
#     # plt.plot(cum_epochs, real_acc, marker='o')
#     # plt.plot(cum_epochs, fake_acc, marker='o')
#     plt.plot(cum_epochs, val_auc_2, marker='o')
#     # plt.plot(cum_epochs, val_real_acc, marker='o')
#     # plt.plot(cum_epochs, val_fake_acc, marker='o')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['epoch', 'Train Accuracy', 'Validation Accuracy', 'auc', 'val_auc'], loc='lower right')
#
#     ax2 = plt.subplot(212)
#     plt.title('GAN Loss')
#     plt.plot(cum_epochs, loss, marker='o')
#     plt.plot(cum_epochs, val_loss, marker='o')
#     plt.ylabel('Loss')
#     ax2.set_ylim([0, 300])
#     plt.xlabel('Epoch')
#     plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
#
# plt.show()

