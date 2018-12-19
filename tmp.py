# -*-coding:utf-8-*-
import os
import shutil
# file = os.listdir('E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output/06/test')
# file2 = os.listdir('E:/open Source/kinetics-i3d/kinetics-i3d/preprocess/data/flow/06/test')
#
# # for i in file:
# #      if i[:-4] not in file1:
# #              list2.append[i]
#
# list1 = [i[:-4] for i in file]
# list2 = [i[:-4] for i in file2]
# list3 = [i for i in list1 if i not in list2]
# print(list3)
# print(len(list3))

# list4 = ['868900865_1549_part_3', '868900865_1549_part_4', '868900865_1549_part_5', '868900865_1549_part_6', '868900865_1549_part_7', '868900865_1549_part_8', '868900865_1549_part_9', '869153222_31_part_0', '869153222_31_part_1', '869153222_31_part_10', '869153222_31_part_11', '869153222_31_part_12', '869153222_31_part_13', '869153222_31_part_14', '869153222_31_part_15', '869153222_31_part_16', '869153222_31_part_17', '869153222_31_part_18', '869153222_31_part_2', '869153222_31_part_3', '869153222_31_part_4', '869153222_31_part_5', '869153222_31_part_6', '869153222_31_part_7', '869153222_31_part_8', '869153222_31_part_9', '883903434_154_part_0', '883903434_154_part_1', '883903434_154_part_10', '883903434_154_part_2', '883903434_154_part_3', '883903434_154_part_4', '883903434_154_part_5', '883903434_154_part_6', '883903434_154_part_7', '883903434_154_part_8', '883903434_154_part_9', '89356062_3984_part_1', '89356062_3984_part_10', '89356062_3984_part_11', '89356062_3984_part_12', '89356062_3984_part_13', '89356062_3984_part_14', '89356062_3984_part_15']
# print(len(list4))

# f = open('UCF101-label.txt','a')
# for i in os.listdir('./UCF101'):
#     f.write(i)
# f.close()

# f1 = open('C:/Users/aiyanye/Desktop/tmp_nan5.txt')
# f2 = open('E:/open Source/kinetics-i3d/kinetics-i3d/preprocess/data/train_test_label/train_videoImage_list_v5.txt')
# l1 = [i for i in f1.readlines()]
# l2 = [i for i in f2.readlines()]
# l3 = [i for i in l1 if i not in l2]
# print(l3)
# print(len(l3))

#
# batch =  ['/02/train/24881317_114_part_3', '/03/train/24881317_19_part_1', '/04/train/987893423_1795_part_10', '/09/train/577220551_28483_part_1', '/01/train/443661357_15771_part_13', '/01/train/740066596_238_part_7', '/00/train/564129820_20060_part_4', '/04/train/496678133_19459_part_9']


# f1 = open('C:/Users/aiyanye/Desktop/tmplog7.txt')
# f2 = open('C:/Users/aiyanye/Desktop/tmp_nan.txt','wb')
# count = 0
# for i in f1.readlines():
#     if 'preprocess' in i:
#         tmp = i.split(' ')
#
#         f2.write(tmp[0][20:-4] + ',' + tmp[1])
#         print(tmp[0][20:-4])
#         count += 1
# print(count)
# f1.close()
# f2.close()
# DATA_DIR =  'E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output'
# DATA_DIR = 'E:/open Source/kinetics-i3d/kinetics-i3d/preprocess/data/nan'
# # file = os.listdir('E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output/')
# file2 = os.listdir('E:/open Source/kinetics-i3d/kinetics-i3d/preprocess/data/nan')
# # f2 = open('preprocess/data/train_test_label/train_videoImage_list_v5.txt')
# f2 = open('C:/Users/aiyanye/Desktop/rgb-1.txt')
# list2 = [i for i in f2.readlines()]
# list1 = [i[:-4] for i in file]
# list2 = [i[:-4] for i in file2]

# filenames = ['/'+ class_fold + "/" + 'train'+ "/" + filename[:-4] + ',' + class_fold[1]+'\n'  # filename + "//"  + class_fold + "//" + train_or_test
#              for class_fold in
#                 os.listdir(DATA_DIR)
#              for filename in
#              # tf.gfile.Glob(os.path.join(class_fold, '*'))
#                os.listdir(DATA_DIR  + "//" + class_fold + "//" + 'train')
#              #  os.listdir(FLAGS.data_dir + "//" + '00' + "//" + train_or_test)
#              ]
# print(len(filenames))
# list3 = [i for i in list2 if i not in filenames ]
# print(list3)
# print(len(list3))
# print(list2[:3])
# print(len(list2))
# # f = open('C:/Users/aiyanye/Desktop/train_rgb.txt','wb')
# # for i in filenames:
# #     f.write(i)
# # f.close()

# srcdir = 'E:/dataset/instruments_video/self labeling video/frame_part2/07_d'
# targetdir = 'E:/dataset/instruments_video/self labeling video/frame_part2/07_d_ok'
#
# for i in os.listdir(srcdir):
#     if '.xml' not in i:
#         if i[:-4] + '.xml' in os.listdir(srcdir):
#             srcFile = os.path.join(srcdir, i)
#             targetFile = os.path.join(targetdir, i)
#             srcfile_xml = os.path.join(srcdir, i[:-4] + '.xml')
#             targetFile_xml = os.path.join(targetdir, i[:-4] + '.xml')
#             shutil.copy(srcFile, targetFile)
#             shutil.copy(srcfile_xml, targetFile_xml)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

y_true = [8, 1, 6, 8, 8, 1, 0, 8, 1, 6, 1, 5, 1, 0, 8, 0, 6, 8, 8, 5, 1, 3, 8, 0, 1, 8, 0, 1, 8, 5, 0, 2, 8, 7, 8, 4, 1, 6, 0, 8, 8, 0, 3, 0, 1, 8, 3, 8, 8, 1, 0, 6, 4, 5, 5, 1, 0, 6, 1, 3, 4, 8, 5, 1, 3, 0, 0, 7, 0, 0, 8, 3, 4, 0, 8, 4, 0, 1, 3, 1, 1, 4, 1, 6, 1, 8, 5, 8, 0, 4, 1, 1, 8, 8, 1, 6, 1, 8, 6, 4, 5, 8, 6, 4, 6, 1, 3, 4, 8, 2, 1, 6, 0, 7, 8, 2, 1, 4, 5, 1, 0, 7, 7, 0, 1, 8, 6, 7, 3, 3, 8, 8, 5, 8, 8, 0, 0, 1, 0, 8, 8, 1, 2, 8, 1, 3, 0, 1, 6, 1, 1, 0, 1, 0, 8, 3, 6, 4, 8, 1, 8, 1, 3, 3, 5, 5, 1, 8, 8, 3, 1, 3, 5, 8, 1, 8, 7, 1, 0, 3, 5, 1, 8, 0, 0, 3, 5, 0, 8, 1, 6, 7, 2, 1, 6, 0, 1, 4, 1, 2, 2, 7, 1, 1, 8, 5, 8, 1, 3, 3, 3, 2, 6, 0, 5, 8, 4, 6, 8, 8, 7, 8, 1, 3, 8, 6, 0, 0, 1, 0, 0, 6, 6, 1, 6, 8, 4, 8, 0, 8, 5, 8, 8, 8, 8, 0, 8, 0, 0, 8, 3, 5, 2, 0, 6, 8, 8, 4, 5, 3, 1, 4, 8, 5, 6, 3, 4, 1, 3, 8, 3, 8, 8, 3, 0, 8, 1, 4, 3, 8, 6, 7, 6, 5, 1, 1, 8, 2, 8, 5, 8, 3, 0, 8, 3, 2, 8, 8, 8, 0, 6, 8, 8, 1, 1, 0, 4, 1, 1, 8, 6, 2, 6, 8, 1, 1, 8, 8, 2, 1, 1, 3, 6, 2, 8, 2, 1, 1, 0, 1, 0, 8, 1, 6, 0, 1, 1, 1, 4, 0, 1, 5, 8, 3, 4, 1, 8, 6, 8, 5, 0, 1, 5, 8, 8, 1, 3, 0, 6, 8, 5, 4, 0, 6, 1, 1, 3, 5, 8, 1, 8, 8, 0, 6, 0, 7, 0, 3, 0, 1, 3, 0, 0, 1, 8, 0, 3, 1, 8, 0, 7, 1, 0, 0, 0, 1, 2, 1, 8, 8, 1, 1, 5, 0, 0, 4, 1, 4, 1, 3, 6, 1, 6, 8, 0, 8, 6, 5, 6, 1, 1, 3, 0, 8, 6, 3, 0, 1, 8, 1, 7, 6, 0, 1, 3, 2, 6, 0, 7, 6, 6, 1, 0, 3, 6, 8, 8, 3, 8, 3, 0, 5, 8, 5, 0, 1, 0, 1, 0, 0, 8, 3, 7, 8, 1, 0, 0, 1, 1, 8, 2, 8, 7, 8, 8, 7, 8, 1, 1, 0, 5, 3, 4, 1, 1, 2, 6, 0, 5, 6, 3, 1, 8, 7, 0, 5, 1, 1, 5, 3, 8, 8, 1, 1, 1, 1, 8, 6, 8, 8, 8, 5, 8, 8, 6, 7, 5, 1, 3, 4, 0, 5, 8, 2, 0, 6, 0, 1, 5, 8, 1, 0, 0, 0, 8, 2, 5, 7, 1, 8, 4, 0, 0, 0, 6, 8, 3, 8, 2, 1, 3, 5, 2, 3, 7, 3, 6, 3, 4, 1, 5, 8, 3, 5, 8, 8, 8, 0, 6, 8, 1, 1, 0, 2, 6, 0, 8, 8, 5, 2, 8, 6, 2, 8, 3, 8, 7, 7, 1, 2, 8, 1, 6, 6, 1, 0, 8, 0, 3, 2, 8, 1, 1, 6, 2, 1, 6, 1, 0, 1, 8, 6, 3, 4, 0, 6, 2, 1, 5, 0, 6, 8, 4, 6, 0, 1, 8, 6, 5, 8, 1, 6, 1, 0, 3, 1, 0, 1, 8, 3, 5, 0, 1, 8, 8, 8, 0, 1, 3, 0, 8, 2, 6, 0, 7, 8, 8, 1, 3, 8, 8, 5, 8, 0, 4, 1, 8, 6, 1, 0, 5, 7, 1, 3, 5, 8, 3, 1, 5, 8, 6, 0, 1, 6, 8, 0, 0, 4, 4, 8, 0, 8, 1, 5, 6, 8, 1, 3, 1, 8, 6, 0, 2, 8, 1, 3, 4, 5, 1, 0, 3, 8, 1, 8, 5, 5, 4, 1, 8, 5, 8, 6, 0, 3, 5, 3, 8, 1, 6, 1, 6, 0, 0, 8, 8, 8, 0, 8, 2, 1, 7, 0, 5, 1, 3, 3, 4, 8, 1, 5, 3, 0, 8, 0, 8, 7, 8, 8, 8, 0, 3, 2, 5, 6, 0, 0, 2, 0, 8, 8, 0, 8, 5, 8, 1, 1, 1, 1, 5, 8, 2, 1, 2, 4, 0, 0, 3, 5]
y_pred = [2, 1, 6, 8, 8, 1, 0, 8, 1, 6, 1, 5, 1, 0, 8, 0, 6, 8, 0, 5, 1, 1, 8, 0, 1, 8, 0, 1, 5, 5, 0, 2, 0, 7, 8, 4, 1, 6, 0, 8, 8, 0, 3, 8, 1, 8, 3, 8, 8, 1, 0, 6, 8, 5, 5, 1, 0, 6, 1, 3, 4, 8, 5, 1, 3, 1, 0, 7, 0, 0, 8, 8, 8, 0, 8, 8, 0, 1, 3, 1, 1, 4, 1, 6, 1, 8, 5, 8, 0, 8, 1, 1, 8, 8, 1, 6, 1, 8, 6, 3, 5, 8, 6, 8, 6, 1, 8, 4, 8, 2, 1, 6, 0, 0, 8, 2, 1, 4, 5, 1, 0, 7, 7, 0, 1, 8, 6, 7, 3, 3, 5, 0, 5, 8, 5, 0, 0, 1, 0, 5, 8, 1, 2, 5, 1, 3, 0, 1, 6, 1, 1, 0, 1, 0, 8, 3, 6, 4, 8, 1, 8, 1, 3, 3, 5, 5, 1, 8, 8, 3, 1, 3, 5, 5, 1, 8, 0, 1, 0, 3, 5, 1, 8, 0, 1, 3, 5, 0, 8, 1, 6, 7, 2, 1, 6, 0, 1, 3, 1, 2, 2, 7, 1, 1, 2, 5, 8, 1, 3, 3, 3, 2, 6, 0, 5, 8, 1, 6, 8, 8, 6, 8, 1, 3, 8, 6, 0, 0, 1, 0, 0, 6, 6, 1, 6, 8, 4, 5, 0, 8, 5, 8, 8, 8, 8, 0, 8, 0, 0, 8, 3, 5, 2, 0, 6, 8, 8, 4, 5, 3, 1, 4, 3, 5, 6, 3, 8, 1, 3, 8, 3, 8, 8, 3, 8, 8, 1, 4, 3, 8, 6, 8, 6, 5, 1, 1, 8, 2, 8, 5, 8, 3, 1, 8, 3, 2, 8, 8, 5, 0, 6, 8, 8, 1, 1, 0, 1, 1, 1, 8, 6, 2, 6, 6, 1, 1, 8, 3, 2, 1, 1, 3, 6, 2, 8, 2, 1, 1, 0, 1, 0, 8, 1, 6, 0, 1, 1, 1, 4, 0, 1, 5, 8, 3, 4, 1, 8, 6, 5, 5, 0, 1, 5, 8, 8, 1, 3, 0, 6, 8, 5, 4, 0, 6, 1, 1, 3, 5, 8, 1, 6, 8, 0, 6, 0, 7, 0, 3, 0, 1, 3, 8, 0, 1, 8, 8, 3, 1, 8, 0, 7, 1, 0, 0, 0, 1, 2, 1, 8, 8, 1, 1, 5, 1, 8, 8, 1, 4, 1, 3, 6, 1, 6, 8, 0, 0, 6, 5, 6, 1, 1, 3, 0, 8, 6, 3, 0, 1, 8, 1, 7, 6, 0, 1, 3, 2, 6, 0, 7, 6, 6, 1, 0, 3, 8, 8, 8, 3, 8, 8, 0, 5, 6, 5, 0, 1, 0, 1, 0, 8, 8, 3, 7, 8, 1, 0, 0, 1, 1, 8, 2, 8, 7, 6, 8, 7, 8, 1, 1, 0, 5, 3, 4, 1, 1, 2, 6, 0, 5, 6, 8, 1, 8, 7, 0, 5, 1, 1, 5, 3, 8, 8, 1, 1, 1, 1, 8, 6, 8, 8, 8, 5, 8, 8, 6, 7, 5, 1, 3, 8, 0, 5, 8, 2, 0, 6, 0, 1, 5, 8, 1, 0, 1, 0, 8, 2, 5, 7, 1, 8, 4, 0, 0, 0, 6, 8, 3, 8, 2, 1, 3, 5, 2, 3, 7, 3, 6, 8, 4, 1, 5, 8, 3, 5, 1, 8, 8, 0, 6, 8, 1, 1, 0, 2, 6, 0, 8, 5, 5, 2, 0, 6, 2, 8, 3, 8, 7, 7, 1, 2, 8, 1, 6, 6, 1, 0, 5, 0, 3, 2, 8, 1, 1, 6, 2, 1, 6, 1, 0, 1, 8, 6, 3, 4, 0, 6, 2, 1, 5, 0, 6, 8, 4, 6, 0, 1, 8, 6, 5, 5, 1, 6, 1, 0, 3, 1, 0, 1, 8, 3, 5, 8, 1, 8, 8, 8, 0, 1, 3, 0, 8, 2, 6, 0, 7, 8, 8, 1, 3, 8, 8, 5, 8, 0, 4, 1, 0, 6, 1, 0, 5, 7, 1, 3, 5, 8, 3, 1, 5, 8, 6, 0, 1, 6, 6, 0, 0, 4, 8, 0, 0, 8, 1, 5, 6, 8, 1, 3, 1, 8, 6, 0, 2, 8, 1, 3, 8, 5, 1, 0, 3, 8, 1, 8, 5, 5, 4, 1, 8, 5, 8, 6, 0, 3, 5, 3, 8, 1, 6, 1, 6, 0, 0, 8, 8, 8, 0, 8, 2, 1, 7, 0, 5, 1, 3, 3, 8, 8, 1, 5, 3, 0, 8, 0, 8, 7, 8, 8, 8, 0, 3, 2, 5, 6, 0, 0, 2, 0, 8, 8, 0, 8, 5, 8, 1, 1, 1, 1, 5, 8, 2, 1, 2, 4, 0, 0, 3, 5]
labels = ['钢琴','吉他','萨克斯','笛子','葫芦丝','架子鼓','古筝','二胡','非乐器']
print(confusion_matrix(
    y_true,   # array, Gound true (correct) target values
    y_pred,  # array, Estimated targets as returned by a classifier
    labels=None,  # array, List of labels to index the matrix.
    sample_weight=None  # array-like of shape = [n_samples], Optional sample weights
))


tick_marks = np.array(range(len(labels))) + 0.5
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues): #plt.cm.binary
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90, fontsize=14)
    plt.yticks(xlocations, labels, fontsize=14)
    plt.ylabel('真实类别', fontsize=16)
    plt.xlabel('预测类别', fontsize=16)

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%.1f%%" % (c*100,), color='red', fontsize=16, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='乐器识别混淆矩阵')
# show confusion matrix
# plt.savefig('../Data/confusion_matrix.png', format='png')
plt.show()