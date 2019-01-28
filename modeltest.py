# ============================================================================
# 加载数据测试，队列方式，设置rgb_or_flow 为 'flow'或'rgb'类型
# 输入数据为.npy格式
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
dropout_keep_prob = 1
batch_size = 1 #8
epoch = 200
_LEARNING_RATE = 0.01
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10
_PREFETCH_BUFFER_SIZE = 30
_MOMENTUM = 0.9
rgb_or_flow = 'flow'

_SAVER_MAX_TO_KEEP = 10
_SAMPLE_VIDEO_FRAMES = 15
_SAMPLE_PATHS = {
   'rgb': '/data2/ye/data/rgb',
   'flow': '/data2/ye/data/flow',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'preprocess/label_kugou.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
train_path = 'preprocess/video_9k_train_list_v2.txt'
test_path = 'preprocess/video_9k_test_list_v2.txt'
rgb_model_path = '/data2/ye/instrument-detect/preprocess/log-joint/kugou_rgb_0.902_model-56882'
flow_model_path = '/data2/ye/instrument-detect/preprocess/log-joint/kugou_flow_0.894_model-51624'
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
    tf.logging.set_verbosity(tf.logging.INFO)
    time1 = time.time()
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained
    NUM_CLASSES = 15  # 400
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


    testpathlist = split_data(test_path)
    print(len(testpathlist))
    test_info_tensor = tf.constant(testpathlist)

    # Phase 2 Testing
    # one element in this dataset is (train_info list)
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)
    test_info_dataset = tf.data.Dataset.from_tensor_slices(test_info_tensor)
    test_dataset_rgb = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, 'rgb'), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset_rgb = test_dataset_rgb.batch(batch_size).repeat()  # 1
    test_dataset_rgb = test_dataset_rgb.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    iterator = tf.data.Iterator.from_structure(
        test_dataset_rgb.output_types, test_dataset_rgb.output_shapes)
    test_init_op_rgb = iterator.make_initializer(test_dataset_rgb)

    clip_holder_rgb, label_holder_rgb = iterator.get_next()
    clip_holder_rgb = tf.squeeze(clip_holder_rgb, [1])
    # label_holder = tf.squeeze(label_holder, [1])
    clip_holder_rgb.set_shape(
        [None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])

    #flow
    test_dataset_flow = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, 'flow'), num_parallel_calls=_NUM_PARALLEL_CALLS)
    # one element in this dataset is (batch image_postprocess, batch label)
    test_dataset_flow = test_dataset_flow.batch(batch_size).repeat()  # 1
    test_dataset_flow = test_dataset_flow.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE)

    iterator = tf.data.Iterator.from_structure(
        test_dataset_flow.output_types, test_dataset_flow.output_shapes)
    test_init_op_flow = iterator.make_initializer(test_dataset_flow)

    clip_holder_flow, label_holder_flow = iterator.get_next()
    clip_holder_flow = tf.squeeze(clip_holder_flow, [1])
    # label_holder = tf.squeeze(label_holder, [1])
    clip_holder_flow.set_shape(
        [None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2])


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

    with tf.Session() as sess:
        sess.run(init)
        if eval_type in ['rgb', 'joint']:
            rgb_saver.restore(sess, rgb_model_path)#kugou_rgb_0.875_model-34944 kugou_flow_0.866_model-41664  kugou_rgb_0.869_model-30912
            tf.logging.info('RGB checkpoint restored')
        if eval_type in ['flow', 'joint']:
            flow_saver.restore(sess, flow_model_path) #_CHECKPOINT_PATHS['flow']kugou_flow_0.873_model-81088
            tf.logging.info('Flow checkpoint restored')

        test_label_rgb = []
        test_rgb_logits = []
        test_label_flow = []
        test_flow_logits = []

        if eval_type in ['rgb','joint']:
            sess.run(test_init_op_rgb)  # new iterator
            for i in range(len(testpathlist)): # test batch is 1
                rgb_logits_val,label = sess.run([rgb_logits,label_holder_rgb],
                                       feed_dict={dropout_holder: 1,
                                                  is_train_holder: False })#True
                test_rgb_logits.append(rgb_logits_val[0])
                test_label_rgb.append(label[0])
                tf.logging.info('rgb %d'%i)

        if eval_type in ['flow', 'joint']:
            sess.run(test_init_op_flow)
            for i in range(len(testpathlist)): # test batch is 1
                flow_logits_val,label = sess.run([flow_logits,label_holder_flow],
                                       feed_dict={dropout_holder: 1,
                                                  is_train_holder: False })#True
                test_flow_logits.append(flow_logits_val[0])
                test_label_flow.append(label[0])
                tf.logging.info('flow %d'%i)

        if eval_type == 'rgb':
            test_output = list(np.argmax(test_rgb_logits, axis=1))
            resultprint(test_label_rgb, test_output)
        if eval_type == 'flow':
            test_output = list(np.argmax(test_flow_logits, axis=1))
            resultprint(test_label_flow, test_output)
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
        print("Optimization Finished! Time total:%.2f, Per Video: %2f" %(duration, duration/len(testpathlist)))

def resultprint(test_label,test_output):
        print(test_label)
        print(test_output)
        accuracy_sklearn = accuracy_score(test_label, test_output)
        precision = precision_score(test_label, test_output, average= 'macro')
        recall = recall_score(test_label, test_output, average= 'macro')
        F1 = f1_score(test_label, test_output, average= 'macro')
        print('accuracy_sklearn: %.3f, precision: %.3f, recall: %.3f, F1: %.3f' %
              (accuracy_sklearn, precision, recall, F1))

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

# reader = tf.train.NewCheckpointReader('/data2/ye/instrument-detect/preprocess/log/kugou_flow_0.988_model-12096')
# varibles = reader.get_variable_to_shape_map()
# for i in varibles:
#     print(i)