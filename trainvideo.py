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

_IMAGE_SIZE = 224
frameHeight = 224#480
frameWidth = 224#640
dropout_keep_prob = 0.8
batch_size = 8 #32
epoch = 100
learning_rate = 0.001
n_classes = 10
videolabel_dict = {}
flag = False
_NUM_PARALLEL_CALLS = 10

_SAMPLE_VIDEO_FRAMES = 15
_SAMPLE_PATHS = {
    # 'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
   'rgb': 'preprocess/data/rgb',  #'E:/dataset/instruments_video/UCF-101/',  # '24881317_23_part_6.npy', #'./24881317_23_part_6_rgb.npy',
   'flow': 'preprocess/data/flow', #'E:/dataset/instruments_video/UCF-101/', #'preprocess/data/flow/24881317_23_part_6.npy',#v_BabyCrawling_g06_c05.npy',
    # 'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

with tf.name_scope('input'):
    # x = tf.placeholder(tf.float32, [None, _SAMPLE_VIDEO_FRAMES, 224, 224, 2], name='x-input') #2
    y = tf.placeholder(tf.float32, [None, n_classes], name='y-input') #[None, n_classes]

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'preprocess/label_kugou.txt'  #'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
train_path = 'preprocess/data/train_test_label/train_videoImage_list_v5.txt'
test_path = 'preprocess/data/train_test_label/test_videoImage_list_v5.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'flow', 'rgb, rgb600, flow, or joint')  #'joint'
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

# def main(unused_argv):
def main():
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

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 3)) #_1, IMAGE_SIZE


    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Logits') #'Logits'
      # rgb_logits, _ = rgb_model(
      #     rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_mix, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)

        netrgb = tf.nn.avg_pool3d(rgb_mix, ksize=[1, 2, 7, 7, 1],  # [1, 2, 7, 7, 1],
                               strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        netrgb = tf.nn.dropout(netrgb, dropout_keep_prob)
        logits = i3d.Unit3D(output_channels=NUM_CLASSES,
                        kernel_shape=[1, 1, 1],
                        activation_fn=None,
                        use_batch_norm=False,
                        use_bias=True,
                        name='Conv3d_0c_1x1')(netrgb, is_training=False)# True
        # if self._spatial_squeeze:
        rgb_logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        rgb_logits = tf.reduce_mean(rgb_logits, axis=1)

    rgb_variable_map = {}
    for variable in tf.global_variables(): #<tf.Variable 'RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_mean:0' shape=(1, 1, 1, 1, 64) dtype=float32_ref>

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable #variable.name='RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0'

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, frameHeight, frameWidth, 2)) # 1
    with tf.variable_scope('Flow'):
        flow_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint= 'Logits') #'Logits' Mixed_5c
        flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
        # flow_mix, _ = flow_model(
        #     flow_input, is_training=False, dropout_keep_prob=1.0)

        flow_variable_map = {}
        for variable in tf.trainable_variables(): #tf.global_variables():
          if variable.name.split('/')[0] == 'Flow' and 'Adam' not in variable.name.split('/')[-1]:##ye #  and variable.name.split('/')[2] != 'Logits'
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
  weights = {'wd2': tf.Variable(tf.truncated_normal([400, 10], stddev = 0.01))}
  biases = {'bd2': tf.Variable(tf.truncated_normal([10], stddev = 0.01))}
  fc1 = tf.add(tf.matmul(model_logits, weights['wd2']), biases['bd2'])
  model_logits = tf.nn.sigmoid(fc1)
  model_predictions = tf.nn.softmax(model_logits)

  varlist = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name) or
                                                 ('wd2' in i.name) or ('bd2' in i.name) and ('Adam' not in i.name) and('Reshape' not in i.name)]
  # [<tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/w:0' shape=(1, 1, 1, 1024, 10) dtype=float32_ref>, \
  # <tf.Variable 'Flow/Conv3d_0c_1x1_ye/conv_3d/b:0' shape=(10,) dtype=float32_ref>]
  varlist1 = [i for i in tf.global_variables() if ('Conv3d_0c_1x1' in i.name)]
  # print(varlist)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_logits + 1e-10, labels=y)) + 0.0001*tf.nn.l2_loss(varlist1[0])
  global_step = tf.Variable(0, trainable=False)
  # learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.95, staircase= True) #100000
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-08).minimize(cost, var_list = varlist, global_step = step) #  AdamOptimizer
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

          checked_grad.append(tf.check_numerics(check, "error occur"))
  with tf.control_dependencies(checked_grad):
      optimizer2 = optimizer.apply_gradients(zip(checked_grad, varlist),global_step=global_step)

  init = tf.global_variables_initializer()  ###ye
  variable1  = 'Flow/inception_i3d/Conv3d_1a_7x7/conv_3d/w:0'
  variable2 = [i for i in tf.global_variables() if 'Reshape' in i.name]
  with tf.Session() as sess:
    sess.run(init)
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
      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      # feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_'
                                                   'imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      # flow_sample = np.load(_SAMPLE_PATHS['flow'])
      # tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      # flow_sample = flow_sample[None,:] #ye
      # flow_sample = flow_sample.reshape(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2)
      # feed_dict[flow_input] = flow_sample

    trainpathlist, labels_one_hot_list  = path2list(train_path)
    print(len(trainpathlist))

    # test_x, test_y = testdata()
    # sess.run(init)
    #    train_writer = tf.summary.FileWriter("./log/",sess.graph)

    count = 0
    batch_num = (int)(np.ceil(len(trainpathlist) / batch_size))
    step = 0
    while count < epoch:
        count = count + 1
        np.random.shuffle(trainpathlist)
        time0 = time.time()
        time2 = time.time()
        print(round(time2- time0, 2),"s, --------------count:", count, "----------------------")
        # for batch_id in range(0, batch_num):
        time1 = time.time()
        #     batch_y = []
        #     if (batch_id + 1) * batch_size < len(trainpathlist):
        #         batch = trainpathlist[batch_id * batch_size:(batch_id + 1) * batch_size]
        #     else:
        #         batch = trainpathlist[batch_id * batch_size:len(trainpathlist)]
        #     batch_x_flow = batch2array(batch, 'flow')
        #     for i in batch:
        #         label = videolabel_dict[i]
        #         batch_y.append(label)
        #     batch_y = np.asarray(batch_y)
        #     tf.check_numerics(batch_x_flow, "input NAN")
        #     tf.check_numerics(tf.cast(batch_y, tf.float32), "input NAN")
            # print("batch_x_flow NAN:", np.any(np.isnan(batch_x_flow)))
            # print("batch_y NAN:", np.any(np.isnan(batch_y)))

        train_info_dataset = tf.data.Dataset.from_tensor_slices((trainpathlist, labels_one_hot_list)).shuffle(len(trainpathlist))
        train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(
                                 x, dataset, mode), num_parallel_calls=_NUM_PARALLEL_CALLS)
        repeat().batch(batch_size)
        iter = dataset.make_one_shot_iterator()
        batch, batch_y = iter.get_next()
        # batch_x_flow = batch2array(batch, 'flow')
        batch_x_flow = tf.py_func(batch2array,[batch,'flow'],[tf.float32] )
        print("batch:", batch)
        print("batch_y:", batch_y)
        sess.run(optimizer2, feed_dict={flow_input: batch_x_flow, y: batch_y})

        out_logits, out_predictions,cost1 = sess.run(
            [model_logits, model_predictions,cost],
            feed_dict={flow_input: batch_x_flow, y: batch_y})

        accuracy = np.mean(np.argmax(out_predictions, axis = 1) == np.argmax(batch_y, axis = 1))
        print(np.argmax(out_predictions, axis = 1))
        print(np.argmax(batch_y, axis = 1))
        # print("out_logits", out_logits)
        print("cost:",cost1)
        # print("learning_rate:",sess.run(learning_rate))
        # print(sess.run(varlist))
        time2 = time.time()
        print(round(time2-time1, 2),'s, count:', count, ', step:', str(step) + '/'+ str(batch_num), ',Norm of logits: %f' % np.linalg.norm(out_logits), ", Prediction accuracy: {:.3f}".format(accuracy))
        step += 1
        if step%5 == 0:
            testdata(sess, flow_input, cost, model_logits, model_predictions, count, batch_num)

    testdata(sess, flow_input, cost, model_logits, model_predictions, count, batch_num)
    saver = tf.train.saver()
    saver.save(sess,"./model/flow-model.ckpt")
#    train_writer.close()
    print("Optimization Finished!")

def testdata(sess,flow_input, cost, model_logits, model_predictions, count, batch_num):
    testpathlist = path2list(test_path)
    np.random.shuffle(testpathlist)
    batch_num_test = 1#(int)(np.ceil(len(testpathlist) / batch_size))
    out_logits = []
    out_predictions = []
    cost1 = 0
    accuracy = 0
    for batch_id in range(0, batch_num_test):
        time1 = time.time()
        batch_y = []
        if (batch_id + 1) * batch_size < len(testpathlist):
            batch = testpathlist[batch_id * batch_size:(batch_id + 1) * batch_size]
        else:
            batch = testpathlist[batch_id * batch_size:len(testpathlist)]
        batch_x_flow = batch2array(batch, 'flow')
        for i in batch:
            label = videolabel_dict[i]
            batch_y.append(label)
        batch_y = np.asarray(batch_y)
        out_logits1, out_predictions1, cost2 = sess.run(
            [model_logits, model_predictions, cost],
            feed_dict={flow_input: batch_x_flow, y:batch_y})

        accuracy1 = np.mean(np.argmax(out_predictions1, axis=1) == np.argmax(batch_y, axis=1))
        # out_logits += out_logits1
        out_logits.append(out_logits1)
        # out_predictions += out_predictions1
        out_predictions.append(out_predictions1)
        accuracy += accuracy1
        cost1 += cost2

    out_logits, out_predictions, cost1, accuracy = np.mean(np.array(out_logits),0), np.mean(np.array(out_predictions),0), cost1/batch_num_test, accuracy/batch_num_test
    time2 = time.time()

    print(round(time2 - time1, 2), 's, count:', count, ', step:', str(batch_id) + '/' + str(batch_num),
          ',Norm of logits: %f' % np.linalg.norm(out_logits), ", TEST accuracy: {:.3f}".format(accuracy))

#读取train或test的文件和label到列表
def path2list(path):
    f = open(path, 'r')
    videopathlist = []
    labels_one_hot_list = []
    labellist = []
    for line in f.readlines():
        videopath = line.strip().split(',')[0]
        videolabel = line.strip().split(',')[1]
        videopathlist.append(videopath)
        # label的one-hot编码
        labels_one_hot = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels_one_hot[int(videolabel)] = 1
        videolabel_dict[videopath] = labels_one_hot
        labels_one_hot_list.append(labels_one_hot)
    f.close()
    labels_one_hot_list = np.asarray(labels_one_hot_list)
    return videopathlist,labels_one_hot_list

#path的list转换为batch的矩阵
def batch2array(pathlist, rgb_or_flow):
  pathdir = _SAMPLE_PATHS[rgb_or_flow]
  batcharray = []#np.zeros(batch_size,_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2)
  # if rgb_or_flow == 'flow':
  #     channel = 2
  num = 0
  for path in pathlist: #'/00/train/125869795_4676_part_0
      file = pathdir + path + '.npy'
      array = np.load(file)
      # array = array.reshape(_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2)
      # batcharray[num] = array
      batcharray.append(array)
      num += 1
  batcharray = np.asarray(batcharray)
  return batcharray

def _get_data_label_from_info(train_info_tensor, name, mode):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    clip_holder, label_holder = tf.py_func(
        batch2array, [train_info_tensor, name, mode], [tf.float32])
    return clip_holder, label_holder

if __name__ == '__main__':
  # tf.app.run(main)
    main()