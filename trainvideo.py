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
import cv2

_IMAGE_SIZE = 224
frameHeight = 224#480
frameWidth = 224#640
dropout_keep_prob = 0.8
batch_size = 32 #8
epoch = 200
_LEARNING_RATE = 0.01
n_classes = 10
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10
_PREFETCH_BUFFER_SIZE = 30
_MOMENTUM = 0.9
rgb_or_flow = 'flow'
_SAVER_MAX_TO_KEEP = 10
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
train_path = 'preprocess/video_8k_train_list_v3.txt'#'E:/dataset/instruments_video/Video_8k_dataset/label_8k/video_8k_test_list_v1.txt'# 'preprocess/data/train_test_label/train_videoImage_list_v5.txt'
test_path = 'preprocess/video_8k_test_list_v3.txt'#'preprocess/data/train_test_label/test_videoImage_list_v5.txt'
log_dir = 'preprocess/log'
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
    test_info_dataset = tf.data.Dataset.from_tensor_slices(test_info_tensor)
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
            is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)

            rgb_variable_map = {}
            for variable in tf.global_variables(): #tf.trainable_variables():
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    if variable.name.split('/')[0] == 'RGB' and ('Adam' not in variable.name.split('/')[-1]) and (
                        'dense' not in variable.name):   # and variable.name.split('/')[2] != 'Logits'
                        rgb_variable_map[variable.name.replace(':0', '')] = variable
            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 2)) # 1
        with tf.variable_scope('Flow'):#Flow
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
            is_in_top_1_op = tf.nn.in_top_k(fc_out, label_holder, 1)

            flow_variable_map = {}
            for variable in tf.global_variables(): #trainable_variables()
              if variable.name.split('/')[0] == 'Flow' and ('Adam' not in variable.name.split('/')[-1]) and ('dense' not in variable.name):##ye #  and variable.name.split('/')[2] != 'Logits'
                    flow_variable_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

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
    saver2 = tf.train.Saver(max_to_keep=_SAVER_MAX_TO_KEEP)
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

    varlist = [i for i in tf.trainable_variables() if ('Conv3d_0c_1x1' in i.name) or ('dense' in i.name)] #or
                                              #   ('wd2' in i.name) or ('bd2' in i.name) and ('Adam' not in i.name) and('Reshape' not in i.name)]
    # [<tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/w:0' shape=(1, 1, 1, 1024, 10) dtype=float32_ref>, \
    # <tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/b:0' shape=(10,) dtype=float32_ref>]
    # varlist = tf.trainable_variables() #tf.global_variables()
    varlist1 = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name)]
    # print(varlist)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits= fc_out, labels=label_holder)) #+ 0.0001*tf.nn.l2_loss(varlist1[0])
    global_step = tf.Variable(0, trainable=False)#softmax_cross_entropy_with_logits  model_logits
    # learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.95, staircase= True) #100000
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-08).minimize(cost, var_list = varlist, global_step = step) #  AdamOptimizer

    # Set learning rate schedule by hand, also you can use an auto way
    boundaries = [10000, 20000, 30000, 40000, 50000] #[10000, 20000, 30000, 40000, 50000]
    values = [_LEARNING_RATE, 0.0008, 0.0005, 0.0003, 0.0001, 5e-5]
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries, values)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # optimizer = tf.train.MomentumOptimizer(learning_rate,_MOMENTUM)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-08)
    gradients = tf.gradients(cost, varlist) #optimizer.compute_gradients(cost)#
    checked_grad = []
    max_gradient_norm = 5

    for check in gradients:
      # flag = isnan(float(check))
      # if(flag):
      #     break
      # else:
      #   clipped_gradients, gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)
      #     checked_grad.append(check)
          checked_grad.append(tf.check_numerics(check, "error occur"))
    with tf.control_dependencies(checked_grad):
        optimizer2 = optimizer.apply_gradients(zip(checked_grad, varlist),global_step=global_step)

    init = tf.global_variables_initializer()
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
        true_count = 0
        logging.info(" ----------------------------Start train----------------------------------")
        logging.info(train_path)
        while step < epoch * batch_num:
            step += 1
            time1 = time.time()
            _, out_logits, out_predictions, cost1, is_in_top_1, input, label, learning_rate_c  \
                = sess.run([optimizer2,fc_out, model_predictions,cost, is_in_top_1_op, clip_holder, label_holder, learning_rate],
                                                             feed_dict={dropout_holder: dropout_keep_prob, is_train_holder: False})#False
            # print("logits max:",np.max(out_logits))
            # print("input max:",np.max(input))
            tmp = np.sum(is_in_top_1)
            tmp_count = tmp
            true_count += tmp
            accuracy = tmp_count /batch_size
            # print(np.argmax(out_predictions, axis = 1))
            # print(label)
            # logging.info(label)
            duration = time.time() - time1
            # print("logits:", out_logits)
            # logging.info(out_logits)
            # logging.info(input)
            # for i in range(input.shape[0]):
            #     for j in np.arange(0,input.shape[1],5):
            #         cv2.imshow('label_'+ str(label[i])+"frame_"+ str(j), input[i][j])
            #         cv2.waitKey(0)

            print("(%.2f sec/batch) epoch:%d, step:%d/%d, learning_rate:%f, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1,learning_rate_c, cost1, accuracy))
            logging.info("(%.2f sec/batch) epoch:%d, step:%d/%d, learning_rate:%f, loss: %-.4f, accuracy: %.3f "
                  % (duration, count, step % batch_num, batch_num-1,learning_rate_c, cost1, accuracy))
            if step % batch_num == 0:
                accuracy = true_count / (batch_num * batch_size)
                print('Epoch%d, train accuracy: %.3f' %
                      (count, accuracy))
                logging.info('Epoch%d, train accuracy: %.3f' %
                             (count, accuracy))
                true_count = 0
                test_count = 0
                if accuracy > 0.1:
                    sess.run(test_init_op) # new iterator
                    # start test process
                    # test_output = []
                    # test_label = []
                    for i in range(len(testpathlist)): # test batch is 1
                        is_in_top_1,test_predictions,label = sess.run([is_in_top_1_op,model_predictions,label_holder],
                                               feed_dict={dropout_holder: 1,
                                                          is_train_holder: False})
                        test_count += np.sum(is_in_top_1)
                        # test_output.append(np.argmax(test_predictions, axis=1))
                        # test_label.append(label)
                    accuracy = test_count / len(testpathlist)
                    # print(test_count,test_output,test_label)
                    # to ensure every test procedure has the same test size
                    # test_data.index_in_epoch = 0
                    print('Epoch%d, test accuracy: %.3f' %
                          (count, accuracy))
                    logging.info('Epoch%d, test accuracy: %.3f' %
                                 (count, accuracy))
                    # saving the best params in test set
                    if accuracy > 0.85:
                        if accuracy > accuracy_tmp:
                            accuracy_tmp = accuracy
                            saver2.save(sess, os.path.join(log_dir,
                                                           'kugou' + '_' + rgb_or_flow +
                                                           '_{:.3f}_model'.format(accuracy)), step)
                    sess.run(train_init_op)
                count += 1
           # train_writer.close()
        sess.close()
        print("Optimization Finished!")

def batch2array(pathlist, rgb_or_flow):
    pathdir = _SAMPLE_PATHS[rgb_or_flow.decode("utf-8")]
    path = str(pathlist[0])[2:-1]
    videolabel = int(str(pathlist[1])[2:-1])
    file = pathdir + path + '.npy'
    # print(path,videolabel)
    # logging.info(path + ','+ str(videolabel))
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