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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2

_IMAGE_SIZE = 224
frameHeight = 224#480
frameWidth = 224#640
dropout_keep_prob = 0.8 #0.8
batch_size = 16 #8
epoch = 200
_LEARNING_RATE = 0.01
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10
_PREFETCH_BUFFER_SIZE = 30
_MOMENTUM = 0.9
rgb_or_flow = 'flow'
_SAVER_MAX_TO_KEEP = 3
_SAMPLE_VIDEO_FRAMES = 15
_SAMPLE_PATHS = {
    # 'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
   'rgb': '/data2/ye/data/rgb',  #'E:/dataset/instruments_video/UCF-101/',  # '24881317_23_part_6.npy', #'./24881317_23_part_6_rgb.npy',
   'flow': '/data2/ye/data/flow', #'E:/dataset/instruments_video/UCF-101/', #'preprocess/data/flow/24881317_23_part_6.npy',#v_BabyCrawling_g06_c05.npy',
    # 'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'preprocess/label_kugou.txt'  #'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
train_path = 'preprocess/video_9k_train_list_v2.txt'
test_path = 'preprocess/video_9k_test_list_v2.txt'
log_dir = 'preprocess/log-joint/'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', rgb_or_flow, 'rgb, rgb600, flow, or joint')  #'joint'
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
    'kugou': 15
}

def main(unused_argv):
# def main(dataset='kugou'):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 15 #400
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
        kinetics_classes = [x.strip().split(' ')[0] for x in f2 if x.strip() != '']
        f2.close()


    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log.txt'),
                    filemode='a', format='%(message)s')

    trainpathlist = split_data(train_path)
    testpathlist = split_data(test_path)
    print(len(trainpathlist))
    print(len(testpathlist))
    train_info_tensor = tf.constant(trainpathlist)
    test_info_tensor = tf.constant(testpathlist)
    train_info_dataset = tf.data.Dataset.from_tensor_slices(train_info_tensor).shuffle(len(trainpathlist))
    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(x,eval_type), num_parallel_calls=_NUM_PARALLEL_CALLS)

    train_dataset = train_dataset.repeat().batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    # Phase 2 Testing
    # one element in this dataset is (train_info list)
    test_info_dataset = tf.data.Dataset.from_tensor_slices(test_info_tensor).shuffle(len(testpathlist))
    # one element in this dataset is (single image_postprocess, single label)
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, eval_type), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset = test_dataset.batch(batch_size).repeat()#1
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
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
              NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Mixed_5c') #'Logits'
            rgb_logits, _ = rgb_model(
              clip_holder, is_training= True, dropout_keep_prob=dropout_holder) #is_train_holder
            with tf.variable_scope('Logits_ye_rgb'):
                netrgb = tf.nn.avg_pool3d(rgb_logits, ksize=[1, 2, 7, 7, 1],
                                           strides=[1, 1, 1, 1, 1], padding=snt.VALID)
                netrgb = tf.nn.dropout(netrgb, dropout_keep_prob)
                logits = i3d.Unit3D(output_channels=NUM_CLASSES,
                                    kernel_shape=[1, 1, 1],
                                    activation_fn=None,  # tf.nn.relu,#
                                    use_batch_norm=False,  # ,True
                                    use_bias=True,
                                    name='Conv3d_0c_1x1_ye_rgb')(netrgb, is_training=is_train_holder)  # True
                rgb_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
                rgb_logits = tf.reduce_mean(rgb_logits, axis=1)
            rgb_variable_map = [v for v in tf.global_variables() if 'Logits_ye_rgb' not in v.name]
            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        with tf.variable_scope('Flow'):#Flow
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Mixed_5c')
            flow_logits, _ = flow_model(
                clip_holder, is_training=True, dropout_keep_prob=dropout_holder)#is_train_holder
        with tf.variable_scope('Logits_ye'):
            netflow = tf.nn.avg_pool3d(flow_logits, ksize=[1, 2, 7, 7, 1],
                                     strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            netflow = tf.nn.dropout(netflow, dropout_keep_prob)
            logits = i3d.Unit3D(output_channels=NUM_CLASSES,
                              kernel_shape=[1, 1, 1],
                              activation_fn= None,#tf.nn.relu,#
                              use_batch_norm= False,#,True
                              use_bias=True,
                              name='Conv3d_0c_1x1_ye')(netflow, is_training=is_train_holder)  # True
            flow_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
            flow_logits = tf.reduce_mean(flow_logits, axis=1)
        flow_variable_map = [v for v in tf.global_variables() if 'Logits_ye' not in v.name]
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    saver2 = tf.train.Saver(max_to_keep=_SAVER_MAX_TO_KEEP)
    if eval_type == 'rgb':
        fc_out = rgb_logits
    elif eval_type == 'flow':
        fc_out = flow_logits
    else:
        fc_out = rgb_logits + flow_logits

    model_predictions = tf.nn.softmax(fc_out)
    is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    varlist = [i for i in tf.trainable_variables()]
    # varlist1 = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name)]#Conv3d_0c_1x1  Logits_ye
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= fc_out, labels=label_holder)) #+ 7e-7*tf.nn.l2_loss(varlist1[0])
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.95, staircase= True) #100000

    # Set learning rate schedule by hand, also you can use an auto way
    boundaries = [20000, 30000, 50000] #[10000, 20000, 30000, 40000, 50000]
    values = [_LEARNING_RATE, 0.005, 0.001, 0.0001]#0.0008, 0.0005, 0.0003, 0.0001, 5e-5]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    # learning_rate = 0.1
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08).minimize(cost, var_list=varlist, global_step=global_step)#
    # optimizer = tf.train.MomentumOptimizer(learning_rate,
    #                                            _MOMENTUM).minimize(cost, var_list=varlist, global_step=global_step)
    # max_gradient_norm = 5
    init = tf.global_variables_initializer()
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
        true_count = 0
        accuracy_tmp = 0
        logging.info(train_path)
        time1 = time.time()
        while step < epoch * batch_num:
            step += 1
            _, out_logits, out_predictions, cost1, is_in_top_1, input, label, learning_rate_c  \
                = sess.run([optimizer,fc_out, model_predictions,cost, is_in_top_1_op, clip_holder, label_holder, learning_rate],
                                                             feed_dict={dropout_holder: dropout_keep_prob, is_train_holder: True})#False
            # learning_rate_c = 0.1
            tmp = np.sum(is_in_top_1)
            tmp_count = tmp
            true_count += tmp
            accuracy = tmp_count /batch_size
            duration = time.time() - time1
            # for i in range(input.shape[0]):
            #     for j in np.arange(0,input.shape[1],5):
            #         cv2.imshow('label_'+ str(label[i])+"frame_"+ str(j), input[i][j])
            #         cv2.waitKey(0)

            print("(%.2f sec/batch) epoch:%d, step:%d/%d, learning_rate:%f, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1,learning_rate_c, cost1, accuracy))
            logging.info("(%.2f sec/batch) epoch:%d, step:%d/%d, learning_rate:%f, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1,learning_rate_c, cost1, accuracy))
            if step % batch_num == 0:#batch_num
                accuracy = true_count / (batch_num * batch_size)
                print('Epoch%d, train accuracy: %.3f' %
                      (count, accuracy))
                logging.info('Epoch%d, train accuracy: %.3f' %
                             (count, accuracy))
                true_count = 0
                test_count = 0
                test_output = []
                test_label = []
                step_test = 0
                if accuracy > 0.1:
                    sess.run(test_init_op) # new iterator
                    # start test process
                    # for i in range(len(testpathlist)): # test batch is 1
                    #     is_in_top_1,test_predictions,label = sess.run([is_in_top_1_op,model_predictions,label_holder],
                    #                            feed_dict={dropout_holder: 1,
                    #                                       is_train_holder: False })#True
                    #     test_count += np.sum(is_in_top_1)
                    #     test_output.append(np.argmax(test_predictions, axis=1))
                    #     test_label.append(label)
                    #  accuracy = test_count / len(testpathlist)
                    test_batch_num = int(np.ceil(len(testpathlist)/batch_size))#最后一个batch的数目不满一个batch_size
                    while step_test < test_batch_num:
                        step_test += 1
                        test_predictions, is_in_top_1, input, label \
                            = sess.run([model_predictions,is_in_top_1_op, clip_holder, label_holder],
                            feed_dict={dropout_holder: 1, is_train_holder: False})  # False
                        test_label.extend(label)
                        test_output.extend(np.argmax(test_predictions, axis=1))
                        tmp = np.sum(is_in_top_1)
                        test_count += tmp
                    if step_test == test_batch_num:  # batch_num
                        # accuracy = test_count / (test_batch_num * batch_size)
                        accuracy = test_count /len(testpathlist)
                    # to ensure every test procedure has the same test size
                    # test_data.index_in_epoch = 0
                    # test_predictions, is_in_top_1, input, label \
                    #             = sess.run([model_predictions,is_in_top_1_op, clip_holder, label_holder],
                    #         feed_dict={dropout_holder: 1, is_train_holder: False})  # False
                    # test_label.extend(label)
                    # test_output.extend(np.argmax(test_predictions, axis=1))
                    # accuracy = np.sum(is_in_top_1)/batch_size
                    accuracy_sklearn = accuracy_score(test_label, test_output)
                    precision = precision_score(test_label, test_output, average= 'macro')
                    recall = recall_score(test_label, test_output, average= 'macro')
                    F1 = f1_score(test_label, test_output, average= 'macro')
                    print('Epoch%d, test accuracy: %.3f,accuracy_sklearn: %.3f, precision: %.3f, recall: %.3f, F1: %.3f' %
                          (count, accuracy, accuracy_sklearn, precision, recall, F1))
                    logging.info('Epoch%d, test accuracy: %.3f, accuracy_sklearn: %.3f, precision: %.3f, recall: %.3f, F1: %.3f' %
                                 (count, accuracy, accuracy_sklearn, precision, recall, F1))
                    # saving the best params in test set
                    if accuracy_sklearn > 0.81:
                        if accuracy_sklearn > accuracy_tmp:
                            accuracy_tmp = accuracy_sklearn
                            saver2.save(sess, os.path.join(log_dir,
                                                           'kugou' + '_' + rgb_or_flow +
                                                           '_{:.3f}_model'.format(accuracy_sklearn)), step)
                    sess.run(train_init_op)
                count += 1
                duration = time.time() - time1
                print("Time total:%.2f, Step: %d" %(duration, step))
           # train_writer.close()
        sess.close()
        print("Optimization Finished!")


def batch2array(pathlist, rgb_or_flow):
    pathdir = _SAMPLE_PATHS[rgb_or_flow.decode("utf-8")]
    path = str(pathlist[0])[2:-1]
    videolabel = int(str(pathlist[1])[2:-1])
    file = pathdir + path + '.npy'
    array = np.load(file)
    return array, videolabel

def _get_data_label_from_info(train_info_tensor, rgb_or_flow):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    clip_holder,label_holder = tf.py_func(
        batch2array, [train_info_tensor, rgb_or_flow], [tf.float32, tf.int64]) #train_info_tensor contains input and label
    return clip_holder, label_holder

def split_data(data_info):
    f = open(data_info)
    train_info = list()
    for line in f.readlines():
        info = line.strip().split(',')
        assert(info[1])
        train_info.append(info)
    f.close()
    return train_info #[filename,label]

if __name__ == '__main__':
  tf.app.run(main)
    # main()