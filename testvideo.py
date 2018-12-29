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
from preprocess.hmdb_extract_flow import compute_TVL1

_IMAGE_SIZE = 224
frameHeight = 224#480
frameWidth = 224#640
dropout_keep_prob = 1
batch_size = 64 #8
epoch = 200
_LEARNING_RATE = 0.01
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10
_PREFETCH_BUFFER_SIZE = 30
_MOMENTUM = 0.9
rgb_or_flow = 'flow'
testnum = 150
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
DATA_DIR = 'E:/dataset/instruments_video/Video_9k_dataset_v3/9k_test_video'#'/data1/csfu/test_video/9k_test_video'#'/data2/dataset/Video_9k_dataset_v3/video_9k'
rgb_model_path = 'kugou_rgb_0.902_model-56882'
flow_model_path = 'preprocess/log-joint/kugou_flow_0.894_model-51624'
_LABEL_MAP_PATH = 'preprocess/label_kugou.txt'  #'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
train_path = 'preprocess/video_9k_train_list_v2.txt'
test_path = '/data1/csfu/test_video/9k_test_video'#'preprocess/video_9k_test_list_v2.txt'
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
    'kugou': 15
}

def main(unused_argv):
# def main(dataset='kugou'):
    time1 = time.time()
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained
    NUM_CLASSES = 15  # 400

    # testpathlist = split_data2(test_path)
    testpathlist = os.listdir(DATA_DIR)
    print(len(testpathlist))
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)

    clip_holder_flow = tf.placeholder(tf.float32,
        [None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2], name='clip_holder_flow')
    clip_holder_rgb = tf.placeholder(tf.float32,
        [None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], name='clip_holder_rgb')

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')  # 'Logits'
            rgb_logits, _ = rgb_model(
                clip_holder_rgb, is_training=False, dropout_keep_prob=dropout_holder)  # is_train_holder
            with tf.variable_scope('Logits_ye_rgb'):
                netrgb = tf.nn.avg_pool3d(rgb_logits, ksize=[1, 2, 7, 7, 1],
                                          strides=[1, 1, 1, 1, 1], padding=snt.VALID)
                netrgb = tf.nn.dropout(netrgb, dropout_holder)
                logits = i3d.Unit3D(output_channels=NUM_CLASSES,
                                    kernel_shape=[1, 1, 1],
                                    activation_fn=None,  # tf.nn.relu,#
                                    use_batch_norm=False,  # ,True
                                    use_bias=True,
                                    name='Conv3d_0c_1x1_ye_rgb')(netrgb, is_training=is_train_holder)  # True
                rgb_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
                rgb_logits = tf.reduce_mean(rgb_logits, axis=1)
            # rgb_variable_map = [v for v in tf.global_variables() if 'Logits_ye_rgb' not in v.name]
        rgb_saver = tf.train.Saver(var_list=[i for i in tf.global_variables() if 'RGB' in i.name])
            # is_in_top_1_op_flow = tf.nn.in_top_k(fc_out, label_holder_flow, 1)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 2)) # 1
        with tf.variable_scope('Flow'):#Flow
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Mixed_5c') #'Logits' Mixed_5c
            flow_logits, _ = flow_model(
                clip_holder_flow, is_training=False, dropout_keep_prob=dropout_holder)#flow_input false 1.0   False  is_train_holder
        with tf.variable_scope('Logits_ye'):
            netflow = tf.nn.avg_pool3d(flow_logits, ksize=[1, 2, 7, 7, 1],  # [1, 2, 7, 7, 1],
                                     strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            netflow = tf.nn.dropout(netflow, dropout_holder)
            logits = i3d.Unit3D(output_channels=NUM_CLASSES,
                              kernel_shape=[1, 1, 1],
                              activation_fn= None,#tf.nn.relu,#
                              use_batch_norm= False,#,True
                              use_bias=True,
                              name='Conv3d_0c_1x1_ye')(netflow, is_training=is_train_holder)  # True
            flow_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
            flow_logits = tf.reduce_mean(flow_logits, axis=1)
            # is_in_top_1_op_flow = tf.nn.in_top_k(fc_out, label_holder_flow, 1)

        flow_saver = tf.train.Saver(var_list=[i for i in tf.global_variables() if 'RGB' not in i.name])
    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits

    # model_predictions = tf.nn.softmax(model_logits)

    init = tf.global_variables_initializer()

    cpu_num = int(os.environ.get('CPU_NUM', 1))
    config = tf.ConfigProto(device_count={"CPU": cpu_num},
                            inter_op_parallelism_threads=cpu_num,
                            intra_op_parallelism_threads=cpu_num,
                            log_device_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(init)
        if eval_type in ['rgb', 'joint']:
            rgb_saver.restore(sess, rgb_model_path)
            tf.logging.info('RGB checkpoint restored')
        if eval_type in ['flow', 'joint']:
            flow_saver.restore(sess, flow_model_path)
            tf.logging.info('Flow checkpoint restored')

        test_label_rgb = []
        test_rgb_logits = []
        test_label_flow = []
        test_flow_logits = []

        # if eval_type in ['rgb','joint']:
        #
        #     for i in range(len(testpathlist)): # test batch is 1
        #         rgbarray =
        #         label =
        #         rgb_logits_val = sess.run([rgb_logits],
        #                                feed_dict={dropout_holder: 1,
        #                                           is_train_holder: False, clip_holder_flow: rgbarray})#True
        #         test_rgb_logits.append(rgb_logits_val[0])
        #         test_label_rgb.append(label[0])
        #         tf.logging.info('rgb %d' % i)
        # import pdb
        # pdb.set_trace()
        flowcount = 0
        duration1 = time.time() - time1
        print('loading time:',duration1)

        index = 0
        batch_num = np.ceil(len(testpathlist)/batch_size)
        batch_index = 0
        endbatch = batch_num*batch_size - len(testpathlist)
        if eval_type in ['flow', 'joint']:
            while batch_index < batch_num:
                if batch_index < batch_num-1:
                    end = (batch_index+1) * batch_size
                elif batch_index == batch_num-1:
                    end = len(testpathlist)
                start = batch_index * batch_size
                flow = compute_TVL1(DATA_DIR + '/' + testpathlist[start])
                flowcount += 1
                batch_index += 1
                time2 = time.time()
                for index in range(start+1,end):
                    video_path = testpathlist[index]
                    time0 = time.time()
                    flowarray = compute_TVL1(DATA_DIR + '/' + video_path)
                    flow = np.concatenate((flow,flowarray))
                    # if flowarray is None:
                    #     continue
                    # label = int(i[1])
                    flowcount += 1
                    print('flow %d,per compute_flow time:%.2f' %(flowcount, time.time()-time0))
                time3 = time.time()
                flow_logits_val = sess.run([flow_logits],
                                       feed_dict={dropout_holder: 1,
                                                  is_train_holder: False, clip_holder_flow: flow})
                test_flow_logits.append(flow_logits_val[0])
                # test_label_flow.append(label)

                # tf.logging.info('flow %d'%flowcount)
                # print('batch_index %d,per compute_flow time:%.2f,per forward total :%.2f' %(batch_index, (time3-time2)/batch_size, (time.time()-time2)/batch_size))

        if eval_type == 'rgb':
            test_output = list(np.argmax(test_rgb_logits, axis=1))
            resultprint(test_label_rgb, test_output)
        if eval_type == 'flow':
            # test_output = list(np.argmax(test_flow_logits, axis=1))
            test_output = []
            for i in range(flowcount):
                test_output.extend(list(np.argmax(test_flow_logits[i], axis=1)))
            # resultprint(test_label_flow, test_output)
        if eval_type == 'joint':
            # flag = np.array(test_label_rgb) == np.array(test_label_flow)
            # print(flag)
            # sumx = np.sum(np.array(test_label_rgb) - np.array(test_label_flow))
            # print(sumx)
            if np.all(np.array(test_label_rgb) == np.array(test_label_flow)):
                logits = np.array(test_rgb_logits) + np.array(test_flow_logits)
                test_label = test_label_flow
                test_output = list(np.argmax(logits, axis=1))
                resultprint(test_label, test_output)
        sess.close()
        duration = time.time() - time1
        print('Optimization Finished! Time total:%.2f, Per Video: %2f' %(duration, duration/flowcount))

def resultprint(test_label,test_output):
        print(test_label)
        print(test_output)
        accuracy_sklearn = accuracy_score(test_label, test_output)
        precision = precision_score(test_label, test_output, average= 'macro')
        recall = recall_score(test_label, test_output, average= 'macro')
        F1 = f1_score(test_label, test_output, average= 'macro')
        print('accuracy_sklearn: %.3f, precision: %.3f, recall: %.3f, F1: %.3f' %
              (accuracy_sklearn, precision, recall, F1))

def split_data2(data_info):
  f = open(data_info)
  train_info = []
  i = 0
  for line in f.readlines():
        info = line.strip().split(',')
        train_info.append(info)
        i += 1
        if i > testnum-1:
            break
  f.close()
  return train_info

if __name__ == '__main__':
  tf.app.run(main)
    # main()

# reader = tf.train.NewCheckpointReader('/data2/ye/instrument-detect/preprocess/log/kugou_flow_0.988_model-12096')
# varibles = reader.get_variable_to_shape_map()
# for i in varibles:
#     print(i)