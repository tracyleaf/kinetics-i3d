# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# -*- coding: utf-8 -*-
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d
import sonnet as snt
import time
from math import isnan

from tensorflow.python import debug as tf_debug
import os
import logging

_IMAGE_SIZE = 224
frameHeight = 224#480
frameWidth = 224#640
dropout_keep_prob = 0.8
batch_size = 4 #32
epoch = 200
_LEARNING_RATE = 0.001
n_classes = 10
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10
_PREFETCH_BUFFER_SIZE = 30
_MOMENTUM = 0.9

_SAMPLE_VIDEO_FRAMES = 15
_SAMPLE_PATHS = {
    # 'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
   'rgb': 'preprocess/data/nan',  #'E:/dataset/instruments_video/UCF-101/',  # '24881317_23_part_6.npy', #'./24881317_23_part_6_rgb.npy',
   'flow': 'preprocess/data/flow', #'E:/dataset/instruments_video/UCF-101/', #'preprocess/data/flow/24881317_23_part_6.npy',#v_BabyCrawling_g06_c05.npy',
    # 'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

# with tf.name_scope('input'):
#     # x = tf.placeholder(tf.float32, [None, _SAMPLE_VIDEO_FRAMES, 224, 224, 2], name='x-input') #2
#     y = tf.placeholder(tf.float32, [None, n_classes], name='y-input') #[None, n_classes]

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'preprocess/label_kugou.txt'  #'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
train_path = 'C:/Users/aiyanye/Desktop/rgb-1.txt'# 'preprocess/data/train_test_label/train_videoImage_list_v5.txt' #train_videoImage_list_v5
# test_path = 'preprocess/data/train_test_label/test_videoImage_list_v5.txt'
log_dir = 'preprocess/log'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, flow, or joint')  #'joint'
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}
_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51,
    'kugou': 10
}

def main(unused_argv):
# def main(dataset='kugou'):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES =  400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        f1 = open(_LABEL_MAP_PATH_600)
        kinetics_classes = [x.strip() for x in f1]
        f1.close()
    else:
        f2 = open(_LABEL_MAP_PATH, 'r', encoding='utf8')
        kinetics_classes = [x.strip().split(' ')[0] for x in f2 if x.strip() != ''] #['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
        f2.close()


    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log.txt'),
                    filemode='w', format='%(message)s')

    trainpathlist = split_data(train_path) #, labels_one_hot_list
    testpathlist = split_data(train_path)###
    print(len(trainpathlist))
    train_info_tensor = tf.constant(trainpathlist)
    test_info_tensor = tf.constant(testpathlist)
    train_info_dataset = tf.data.Dataset.from_tensor_slices(train_info_tensor)#.shuffle(len(trainpathlist))
    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(x,eval_type), num_parallel_calls=_NUM_PARALLEL_CALLS)

    train_dataset = train_dataset.repeat().batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # Phase 2 Testing
    # one element in this dataset is (train_info list)
    test_info_dataset = tf.data.Dataset.from_tensor_slices(
        (test_info_tensor))
    # one element in this dataset is (single image_postprocess, single label)
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, eval_type), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset = test_dataset.batch(1).repeat()
    test_dataset = test_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    clip_holder, label_holder = iterator.get_next()
    clip_holder = tf.squeeze(clip_holder,  [1])
    # label_holder = tf.squeeze(label_holder, [1])
    clip_holder.set_shape(
        [None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, _CHANNEL[eval_type]])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 3)) #_1, IMAGE_SIZE

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
              NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Logits') #'Logits'
            rgb_logits, _ = rgb_model(
              clip_holder, is_training=is_train_holder, dropout_keep_prob=dropout_holder)
            rgb_logits = tf.nn.dropout(rgb_logits, dropout_holder)


        # rgb_variable_map = {}
        # for variable in tf.global_variables(): #<tf.Variable 'RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_mean:0' shape=(1, 1, 1, 1, 64) dtype=float32_ref>
        #
        #   if variable.name.split('/')[0] == 'RGB':
        #     if eval_type == 'rgb600':
        #       rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        #     else:
        #       rgb_variable_map[variable.name.replace(':0', '')] = variable #variable.name='RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0'
            fc_out = tf.layers.dense(
                rgb_logits, _CLASS_NUM['kugou'], use_bias=True)
            # compute the top-k results for the whole batch size
            is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)  # 最大值的索引是否相等

            rgb_variable_map = {}
            for variable in tf.trainable_variables():  # tf.global_variables():
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    if variable.name.split('/')[0] == 'RGB' and ('Adam' not in variable.name.split('/')[-1]) and (
                        'dense' not in variable.name):  ##ye #  and variable.name.split('/')[2] != 'Logits'
                        rgb_variable_map[variable.name.replace(':0', '')] = variable
            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 2)) # 1
        with tf.variable_scope('rgb'):#Flow
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Logits') #'Logits' Mixed_5c
            flow_logits, _ = flow_model(
                clip_holder, is_training=is_train_holder, dropout_keep_prob=dropout_holder)#flow_input false 1.0
            flow_logits = tf.nn.dropout(flow_logits, dropout_holder)
            # flow_mix, _ = flow_model(
            #     flow_input, is_training=False, dropout_keep_prob=1.0)
            # To change 400 classes to the ucf101 or hdmb classes
            fc_out = tf.layers.dense(
                flow_logits, _CLASS_NUM['kugou'], use_bias=True)
            # compute the top-k results for the whole batch size
            is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)#最大值的索引是否相等

            flow_variable_map = {}
            for variable in tf.trainable_variables(): #tf.global_variables():
              if variable.name.split('/')[0] == 'Flow' and ('Adam' not in variable.name.split('/')[-1]) and ('dense' not in variable.name):##ye #  and variable.name.split('/')[2] != 'Logits'
                    flow_variable_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)  # 需要恢复的变量列表

            # netflow = tf.nn.avg_pool3d(flow_mix, ksize=[1, 2, 7, 7, 1],  # [1, 2, 7, 7, 1],
            #                          strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            # netflow = tf.nn.dropout(netflow, dropout_keep_prob)
            # logits = i3d.Unit3D(output_channels=NUM_CLASSES,
            #                   kernel_shape=[1, 1, 1],
            #                   activation_fn= tf.nn.sigmoid, #None,
            #                   use_batch_norm=False,
            #                   use_bias=True,
            #                   name='Conv3d_0c_1x1_ye')(netflow, is_training=True)  # True
            # # if self._spatial_squeeze:
            # flow_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
            # flow_logits = tf.reduce_mean(flow_logits, axis=1)

    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits

    # model_predictions = tf.nn.softmax(model_logits)
    # weights = {'wd2': tf.Variable(tf.truncated_normal([400, 10], stddev = 0.01))}
    # biases = {'bd2': tf.Variable(tf.truncated_normal([10], stddev = 0.01))}
    # fc1 = tf.add(tf.matmul(model_logits, weights['wd2']), biases['bd2'])
    # model_logits = tf.nn.sigmoid(fc1)
    model_predictions = tf.nn.softmax(fc_out)

    varlist = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name) or ('dense' in i.name)] #or
                                              #   ('wd2' in i.name) or ('bd2' in i.name) and ('Adam' not in i.name) and('Reshape' not in i.name)]
    # [<tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/w:0' shape=(1, 1, 1, 1024, 10) dtype=float32_ref>, \
    # <tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/b:0' shape=(10,) dtype=float32_ref>]
    # varlist = tf.global_variables()
    varlist1 = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name)]
    # print(varlist)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= fc_out+ 1e-10, labels=label_holder)) + 0.0001*tf.nn.l2_loss(varlist1[0])
    global_step = tf.Variable(0, trainable=False)#softmax_cross_entropy_with_logits  model_logits
    # learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.95, staircase= True) #100000
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-08).minimize(cost, var_list = varlist, global_step = step) #  AdamOptimizer

    # Set learning rate schedule by hand, also you can use an auto way
    boundaries = [10000, 20000, 30000, 40000, 50000]
    values = [_LEARNING_RATE, 0.0008, 0.0005, 0.0003, 0.0001, 5e-5]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate,_MOMENTUM)
    gradients = tf.gradients(cost, varlist) #optimizer.compute_gradients(cost)#
    checked_grad = []
    max_gradient_norm = 5

    for check in gradients:
      # flag = isnan(float(check))
      # if(flag):
      #     break
      # else:
      #   clipped_gradients, gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)
          checked_grad.append(check)
          # checked_grad.append(tf.check_numerics(check, "error occur"))
    with tf.control_dependencies(checked_grad):
        optimizer2 = optimizer.apply_gradients(zip(checked_grad, varlist),global_step=global_step)

    init = tf.global_variables_initializer()  ###ye
    variable1  = 'Flow/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0'
    variable2 = [i for i in tf.global_variables() if 'Reshape' in i.name]
    with tf.Session() as sess:
        sess.run(init)
        sess.run(train_init_op)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        # print(sess.run(variable1))
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')
            # rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
            # tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            # feed_dict[rgb_input] = rgb_sample

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')

        batch_num = (int)(np.ceil(len(trainpathlist) / batch_size))
        step = 0
        count = 0
        tmp_count = 0
        print(" -----------------Start train-------------------")
        logging.info(train_path)
        while step < epoch * batch_num:
            step += 1
            time0 = time.time()

            time1 = time.time()
            _, out_logits, out_predictions, cost1, is_in_top_1, label = sess.run([optimizer2,fc_out, model_predictions,cost, is_in_top_1_op,label_holder],
                                                             feed_dict={dropout_holder: dropout_keep_prob, is_train_holder: True})
            # print("logits max:",np.max(out_logits))
            # print("input max:",np.max(input))
            tmp = np.sum(is_in_top_1)
            tmp_count = tmp
            # accuracy = np.mean(np.argmax(out_predictions, axis = 1) == np.argmax(batch_y, axis = 1))
            # accuracy = np.mean(np.argmax(out_predictions, axis=1) == is_in_top_1)
            accuracy = tmp_count /batch_size
            print(np.argmax(out_predictions, axis = 1))
            # print(np.argmax(batch_y, axis = 1))
            print(label)

            # print("out_logits", out_logits)
            print("cost:",cost1)
            print("learning_rate:",sess.run(learning_rate))
            # print(sess.run(varlist))
            duration = time.time() - time1
            # print(round(time2-time1, 2),'s, count:', count, ', step:', str(step) + '/'+ str(batch_num), ',Norm of logits: %f' % np.linalg.norm(out_logits), ", Prediction accuracy: {:.3f}".format(accuracy))
            print("(%.2f sec/batch) epoch:%d, step:%d/%d, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1, cost1, accuracy))
            logging.info("(%.2f sec/batch) epoch:%d, step:%d/%d, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1, cost1, accuracy))
            if step % batch_num == 0:
                count += 1
            # if step % 5== 0:# and accuracy > _RUN_TEST_THRESH:
            #     sess.run(test_init_op)
            #     true_count = 0
            #     # start test process
            #     print(len(testpathlist))
            #     for i in range(len(testpathlist)):
            #         # print(i,true_count)
            #         is_in_top_1 = sess.run(is_in_top_1_op,
            #                                feed_dict={dropout_holder: 1,
            #                                           is_train_holder: False})
            #         true_count += np.sum(is_in_top_1)
            #     accuracy = true_count / len(testpathlist)
            #     true_count = 0
            #     # to ensure every test procedure has the same test size
            #     # test_data.index_in_epoch = 0
            #     print('Step%d, test accuracy: %.3f' %
            #           (step, accuracy))
            #     # logging.info('Epoch%d, test accuracy: %.3f' %
            #     #              (train_data.epoch_completed, accuracy))
            #     # saving the best params in test set
            #     # if accuracy > _SAVE_MODEL_THRESH:
            #     #     if accuracy > accuracy_tmp:
            #     #         accuracy_tmp = accuracy
            #     #         saver2.save(sess, os.path.join(log_dir,
            #     #                                        test_data.name+'_'+train_data.mode +
            #     #                                        '_{:.3f}_model'.format(accuracy)), step)
            #     sess.run(train_init_op)

        #     if step%5 == 0:
        #         testdata(sess, flow_input, cost, model_logits, model_predictions, count, batch_num)
        #
        # testdata(sess, flow_input, cost, model_logits, model_predictions, count, batch_num)
        # saver = tf.train.saver()
        # saver.save(sess,"./model/flow-model.ckpt")
        #    train_writer.close()
        sess.close()
        print("Optimization Finished!")

# def testdata(sess,flow_input, cost, model_logits, model_predictions, count, batch_num):
#     testpathlist = path2list(test_path)
#     np.random.shuffle(testpathlist)
#     batch_num_test = 1#(int)(np.ceil(len(testpathlist) / batch_size))
#     out_logits = []
#     out_predictions = []
#     cost1 = 0
#     accuracy = 0
#     for batch_id in range(0, batch_num_test):
#         time1 = time.time()
#         batch_y = []
#         if (batch_id + 1) * batch_size < len(testpathlist):
#             batch = testpathlist[batch_id * batch_size:(batch_id + 1) * batch_size]
#         else:
#             batch = testpathlist[batch_id * batch_size:len(testpathlist)]
#         batch_x_flow = batch2array(batch, 'flow')
#         for i in batch:
#             label = videolabel_dict[i]
#             batch_y.append(label)
#         batch_y = np.asarray(batch_y)
#         out_logits1, out_predictions1, cost2 = sess.run(
#             [model_logits, model_predictions, cost],
#             feed_dict={flow_input: batch_x_flow, y:batch_y})
#
#         accuracy1 = np.mean(np.argmax(out_predictions1, axis=1) == np.argmax(batch_y, axis=1))
#         # out_logits += out_logits1
#         out_logits.append(out_logits1)
#         # out_predictions += out_predictions1
#         out_predictions.append(out_predictions1)
#         accuracy += accuracy1
#         cost1 += cost2
#
#     out_logits, out_predictions, cost1, accuracy = np.mean(np.array(out_logits),0), np.mean(np.array(out_predictions),0), cost1/batch_num_test, accuracy/batch_num_test
#     time2 = time.time()
#
#     print(round(time2 - time1, 2), 's, count:', count, ', step:', str(batch_id) + '/' + str(batch_num),
#           ',Norm of logits: %f' % np.linalg.norm(out_logits), ", TEST accuracy: {:.3f}".format(accuracy))

#path的list转换为batch的矩阵
def batch2array(pathlist, rgb_or_flow):
    pathdir = _SAMPLE_PATHS[rgb_or_flow.decode("utf-8")]
    path = str(pathlist[0])[2:-1]
    videolabel = int(str(pathlist[1])[2:-1])
    file = pathdir + path + '.npy'
    print(path,videolabel)
    logging.info(path + ','+ str(videolabel))
    array = np.load(file)
    # labels_one_hot = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # labels_one_hot[int(videolabel)] = 1
    return array, videolabel

def _get_data_label_from_info(train_info_tensor, rgb_or_flow):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    print("input:", train_info_tensor)
    clip_holder,label_holder = tf.py_func(
        batch2array, [train_info_tensor, rgb_or_flow], [tf.float32, tf.int32]) #train_info_tensor里面包含input和label
    return clip_holder,label_holder

def split_data(data_info):
    f = open(data_info)
    train_info = list()
    for line in f.readlines():
        info = line.strip().split(',')
        print(info[1])
        assert(info[1])
        train_info.append(info)
    f.close()
    return train_info #[文件名,label]

if __name__ == '__main__':
  tf.app.run(main)
    # main()