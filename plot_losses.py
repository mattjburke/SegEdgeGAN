import matplotlib.pyplot as plt
import csv

# FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-10-14:36:28.987422/'
# FILEPATH_TRAIN = FILE_BASE + 'G1D1G2D2_e9_2020-04-10-14:36:28.987422.csv'
# FILEPATH_VAL = FILE_BASE + 'G1D1G2D2_val_e9_2020-04-10-14:36:28.987422.csv'

# FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-12-12:19:02.479705/'
# FILEPATH_TRAIN = FILE_BASE + 'G1D1G2D2_ave_e9_2020-04-12-12:19:02.479705.csv'
# FILEPATH_VAL = FILE_BASE + 'G1D1G2D2_val_e9_2020-04-12-12:19:02.479705.csv'

# FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-14-01:10:45.123745/'
# FILEPATH_TRAIN = FILE_BASE + 'G1D1G2D2_ave_e6_2020-04-14-01:10:45.123745.csv'
# FILEPATH_VAL = FILE_BASE + 'G1D1G2D2_val_ave_e6_2020-04-14-01:10:45.123745.csv'
# FILEPATH_TRAIN = FILE_BASE + 'G1D1G2D2_e6_2020-04-14-01:10:45.123745.csv'
# FILEPATH_VAL = FILE_BASE + 'G1D1G2D2_val_e6_2020-04-14-01:10:45.123745.csv'

# 21 epochs of G1D1G2D2
FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-14-23:34:27.841495/'
FILEPATH_TRAIN_both = FILE_BASE + 'G1D1G2D2_ave_e26_2020-04-14-23:34:27.841495.csv'
FILEPATH_VAL_both = FILE_BASE + 'G1D1G2D2_val_ave_e26_2020-04-14-23:34:27.841495.csv'

# 7 epochs of G1D1 only
FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-15-11:16:33.560820/'
FILEPATH_TRAIN_1only = FILE_BASE + 'G1D1G2D2_ave_e26_2020-04-15-11:16:33.560820.csv'
FILEPATH_VAL_1only = FILE_BASE + 'G1D1G2D2_val_ave_e26_2020-04-15-11:16:33.560820.csv'

# with 10 l_data2 and 5 cgan_2 (.5 on discrim)
FILE_BASE = '/Users/MatthewBurke/PycharmProjects/SegEdgeGAN/saved/2020-04-16-01:13:54.794910/'
FILEPATH_TRAIN_3 = FILE_BASE + 'G1D1G2D2_ave_e13_2020-04-16-01:13:54.794910.csv'
FILEPATH_VAL_3 = FILE_BASE + 'G1D1G2D2_val_ave_e13_2020-04-16-01:13:54.794910.csv'

epochs_2 = []
iou_2 = []
epochs_1 = []
iou_1 = []
val_epochs_2 = []
val_iou_2 = []
val_epochs_1 = []
val_iou_1 = []

epochs_3 = []
iou_3 = []
val_epochs_3 = []
val_iou_3 = []

with open(FILEPATH_TRAIN_both, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        epochs_2.append(int(row['Col0']))
        iou_2.append(float(row['Col1']))

with open(FILEPATH_TRAIN_1only, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        epochs_1.append(int(row['Col0']))
        iou_1.append(float(row['Col1']))

with open(FILEPATH_VAL_both, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        val_epochs_2.append(int(row['Col0']))
        val_iou_2.append(float(row['Col1']))

with open(FILEPATH_VAL_1only, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        val_epochs_1.append(int(row['Col0']))
        val_iou_1.append(float(row['Col1']))

with open(FILEPATH_TRAIN_3, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        epochs_3.append(int(row['Col0']))
        iou_3.append(float(row['Col1']))

with open(FILEPATH_VAL_3, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        val_epochs_3.append(int(row['Col0']))
        val_iou_3.append(float(row['Col1']))

plt.figure()
ax = plt.subplot()
plt.ylabel('IOU Score')
plt.xlabel('Epoch')
# ax.set_ylim([.2, .35])
plt.title('Intersection Over Union Scores')
# plt.plot(iters, epochs, marker=',')
plt.plot(epochs_2, iou_2, marker='o')
plt.plot(epochs_1, iou_1, marker='o')
plt.plot(val_epochs_2, val_iou_2, marker='o')
plt.plot(val_epochs_1, val_iou_1, marker='o')
plt.plot(epochs_3, iou_3, marker='o')
plt.plot(val_epochs_3, val_iou_3, marker='o')
plt.legend(['train_G1D1G2D2', 'train_G1D1', 'val_G1D1G2D2', 'val_G1D1', 'train_high_loss', 'val_high_loss'], loc='lower right')
plt.show()


'''
# lists in training files
epochs = []
iou_scores = []
total_losses = []
D_losses = []
G_losses = []
L_data1_losses = []
L_data2_losses = []
L_cgan1_losses = []
D1_losses = []
G1_adv_losses = []
L_cgan2_losses = []
D2_losses = []
G2_adv_losses = []
# epochs = []
# stores epochs as float for each data point
# sample gathered every 25 iterations = 119 per epoch (2975 training files)

# for FILEPATH_TRAIN in [FILEPATH_TRAIN_both, FILEPATH_TRAIN_1only]:
with open(FILEPATH_TRAIN_both, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    # i = 0
    for row in plots:
        try:
            epochs.append(int(row['Col0']))
        except ValueError as e:
            print(e)
            epochs.append(-100)
        try:
            iou_scores.append(float(row['Col1']))
        except ValueError as e:
            print(e)
            iou_scores.append(-100)
        try:
            total_losses.append(float(row['Col2']))
        except ValueError as e:
            print(e)
            total_losses.append(-100)
        try:
            D_losses.append(float(row['Col3']))
        except ValueError as e:
            print(e)
            D_losses.append(-100)
        try:
            G_losses.append(float(row['Col4']))
        except ValueError as e:
            print(e)
            G_losses.append(-100)
        try:
            L_data1_losses.append(float(row['Col5']))
        except ValueError as e:
            print(e)
            L_data1_losses.append(-100)
        try:
            L_data2_losses.append(float(row['Col6']))
        except ValueError as e:
            print(e)
            L_data2_losses.append(-100)
        try:
            L_cgan1_losses.append(float(row['Col7']))
        except ValueError as e:
            print(e)
            L_cgan1_losses.append(-100)
        try:
            D1_losses.append(float(row['Col8']))
        except ValueError as e:
            print(e)
            D1_losses.append(-100)
        try:
            G1_adv_losses.append(float(row['Col9']))
        except ValueError as e:
            print(e)
            G1_adv_losses.append(-100)
        try:
            L_cgan2_losses.append(float(row['Col10']))
        except ValueError as e:
            print(e)
            L_cgan2_losses.append(-100)
        try:
            D2_losses.append(float(row['Col11']))
        except ValueError as e:
            print(e)
            D2_losses.append(-100)
        try:
            G2_adv_losses.append(float(row['Col12']))
        except ValueError as e:
            print(e)
            G2_adv_losses.append(-100)


plt.figure()

# ax1 = plt.subplot(211)
# ax1.set_ylim([0, 1])
plt.title('Train Metrics')
# plt.plot(iters, epochs, marker=',')
# plt.plot(epochs, iou_scores, marker='o')
# plt.plot(epochs, total_losses, marker='o')
# plt.plot(epochs, D_losses, marker='o')
# plt.plot(epochs, G_losses, marker='o')
plt.plot(epochs, L_data1_losses, marker=',')
plt.plot(epochs, L_data2_losses, marker=',')
plt.plot(epochs, L_cgan1_losses, marker=',')
plt.plot(epochs, D1_losses, marker=',')
plt.plot(epochs, G1_adv_losses, marker=',')
plt.plot(epochs, L_cgan2_losses, marker=',')
plt.plot(epochs, D2_losses, marker=',')
plt.plot(epochs, G2_adv_losses, marker=',')

plt.ylabel('Train Metric')
plt.xlabel('Epoch')
# plt.legend(['iou_scores', 'total_losses'], loc='lower right')

# plt.legend(['iou_scores', 'D_loss', 'G_loss'], loc='lower right')

# plt.legend(['L_data1_losses', 'L_data2_losses'], loc='lower right')

# all G_loss components
# plt.legend(['total_losses', 'G_loss', 'L_data1_losses', 'L_data2_losses', 'G1_adv_losses', 'G2_adv_losses'], loc='lower right')

# all D_loss components
# plt.legend(['total_losses', 'D_loss', 'D1_losses', 'D2_losses'], loc='lower right')

# plt.legend(['iou_scores',
#             'D1_losses', 'G1_adv_losses'], loc='lower right')

plt.legend(['L_data1_losses', 'L_data2_losses', 'L_cgan1_losses',
            'D1_losses', 'G1_adv_losses', 'L_cgan2_losses', 'D2_losses', 'G2_adv_losses'], loc='lower right')

plt.show()
'''
# exit(0)
'''
# lists in validation files
val_epochs = []
val_iou_scores = []
val_total_losses = []
val_D_losses = []
val_G_losses = []
val_L_data1_losses = []
val_L_data2_losses = []
val_L_cgan1_losses = []
val_D1_losses = []
val_G1_adv_losses = []
val_L_cgan2_losses = []
val_D2_losses = []
val_G2_adv_losses = []
# only average per epoch is gathered for val set
# epochs = []  # used to represent epoch
# num_points_per_epoch = 119  # use to match graph with training data gathered

with open(FILEPATH_VAL, 'r') as csvfile:
    plots = csv.DictReader(csvfile)
    for row in plots:
        try:
            val_epochs.append(int(row['Col0']))
        except ValueError as e:
            print(e)
            val_epochs.append(-100)
        try:
            val_iou_scores.append(float(row['Col1']))
        except ValueError as e:
            print(e)
            val_iou_scores.append(-100)
        try:
            val_total_losses.append(float(row['Col2']))
        except ValueError as e:
            print(e)
            val_total_losses.append(-100)
        try:
            val_D_losses.append(float(row['Col3']))
        except ValueError as e:
            print(e)
            val_D_losses.append(-100)
        try:
            val_G_losses.append(float(row['Col4']))
        except ValueError as e:
            print(e)
            val_G_losses.append(-100)
        try:
            val_L_data1_losses.append(float(row['Col5']))
        except ValueError as e:
            print(e)
            val_L_data1_losses.append(-100)
        try:
            val_L_data2_losses.append(float(row['Col6']))
        except ValueError as e:
            print(e)
            val_L_data2_losses.append(-100)
        try:
            val_L_cgan1_losses.append(float(row['Col7']))
        except ValueError as e:
            print(e)
            val_L_cgan1_losses.append(-100)
        try:
            val_D1_losses.append(float(row['Col8']))
        except ValueError as e:
            print(e)
            val_D1_losses.append(-100)
        # try:
        #     val_G1_adv_losses.append(float(row['Col9']))
        # except ValueError as e:
        #     print(e)
        #     val_G1_adv_losses.append(-100)
        # try:
        #     val_L_cgan2_losses.append(float(row['Col10']))
        # except ValueError as e:
        #     print(e)
        #     val_L_cgan2_losses.append(-100)
        # try:
        #     val_D2_losses.append(float(row['Col11']))
        # except ValueError as e:
        #     print(e)
        #     val_D2_losses.append(-100)
        # try:
        #     val_G2_adv_losses.append(float(row['Col12']))
        # except ValueError as e:
        #     print(e)
        #     val_G2_adv_losses.append(-100)
        # val_epochs.append(int(row['Col0']))
        # val_iou_scores.append(float(row['Col1']))
        # val_total_losses.append(float(row['Col2']))
        # val_D_losses.append(float(row['Col3']))
        # val_G_losses.append(float(row['Col4']))
        # val_L_data1_losses.append(float(row['Col5']))
        # val_L_data2_losses.append(float(row['Col6']))
        # val_L_cgan1_losses.append(float(row['Col7']))
        # val_D1_losses.append(float(row['Col8']))
        # val_G1_adv_losses.append(float(row['Col9']))
        # val_L_cgan2_losses.append(float(row['Col10']))
        # val_D2_losses.append(float(row['Col11']))
        # val_G2_adv_losses.append(float(row['Col12']))
        # epochs.append(int(e))  # gets epoch
        # e +=1

# plt.figure()


ax2 = plt.subplot(212)
ax2.set_ylim([0, 1])
plt.title('Val Metrics')
# plt.plot(iters, val_epochs, marker='o')
plt.plot(val_epochs, val_iou_scores, marker='o')
# plt.plot(val_epochs, val_total_losses, marker='o')
plt.plot(val_epochs, val_D_losses, marker='o')
plt.plot(val_epochs, val_G_losses, marker='o')
# plt.plot(val_epochs, val_L_data1_losses, marker='o')
# plt.plot(val_epochs, val_L_data2_losses, marker='o')
# plt.plot(val_epochs, val_L_cgan1_losses, marker='o')
# plt.plot(val_epochs, val_D1_losses, marker='o')
# plt.plot(val_epochs, val_G1_adv_losses, marker='o')
# plt.plot(val_epochs, val_L_cgan2_losses, marker='o')
# plt.plot(val_epochs, val_D2_losses, marker='o')
# plt.plot(val_epochs, val_G2_adv_losses, marker='o')

plt.ylabel('Val Metric')
plt.xlabel('Epoch')
# plt.legend(['val_iou_scores', 'val_total_losses'], loc='lower right')

plt.legend(['val_iou_scores', 'val_D_loss', 'val_G_loss'], loc='lower right')

# plt.legend(['val_L_data1_losses', 'val_L_data2_losses'], loc='lower right')

# all G_loss components
# plt.legend(['val_total_losses', 'val_G_loss', 'val_L_data1_losses', 'val_L_data2_losses', 'val_G1_adv_losses', 'val_G2_adv_losses'], loc='lower right')

# all D_loss components
# plt.legend(['val_total_losses', 'val_D_loss', 'val_D1_losses', 'val_D2_losses'], loc='lower right')

# plt.legend(['val_iou_scores', 'val_total_losses', 'val_D_loss', 'val_G_loss', 'val_L_data1_losses', 'val_L_data2_losses',
#             'val_L_cgan1_losses', 'val_D1_losses', 'val_G1_adv_losses', 'val_L_cgan2_losses', 'val_D2_losses',
#             'val_G2_adv_losses'], loc='lower right')

plt.show()

'''
