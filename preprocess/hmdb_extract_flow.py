from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from tensorflow.python.platform import app, flags
import os
import sys
import cv2
import threading
import tensorflow as tf
import numpy as np
import time

train_or_test = 'test'
DATA_DIR = '/data2/dataset/Video_9k_dataset_v3/video_9k'
SAVE_DIR = '/data2/ye/data/flow'
train_path = '/data2/ye/instrument-detect/preprocess/video_9k_train_list_v2.txt'
test_path = '/data2/ye/instrument-detect/preprocess/video_9k_test_list_v2.txt'
_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224
frameWidth = 224 #480
frameHeight = 224 #640
framenum = 15
frame_interval = 10
_CLASS_NAMES = 'label_kugou.txt' #'UCF101-label.txt'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('save_to', SAVE_DIR, 'where to save flow data.')
flags.DEFINE_string('name', 'HMDB', 'dataset name.')
flags.DEFINE_integer('num_threads', 16, 'number of threads.') #32


def _video_length(video_path):
  """Return the length of a video.


  Args:
    video_path: String to the video file location.

  Returns:
    Length of the video.

  Raises:
    ValueError: Either wrong path or the file extension is not supported.
  """
  _, ext = os.path.splitext(video_path)
  if not ext in _EXT:
    raise ValueError('Extension "%s" not supported' % ext)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError("Could not open the file.\n{}".format(video_path))
  if cv2.__version__ >= '3.0.0':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
  else:
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
  length = int(cap.get(CAP_PROP_FRAME_COUNT))
  return length

def compute_TVL1(video_path):
  """Compute the TV-L1 optical flow."""
  TVL1 = DualTVL1()
  cap = cv2.VideoCapture(video_path)
  for i in range(frame_interval):#former 10 frames deserted
    ret = False
    j = 0
    while not ret:
      ret, frame1 = cap.read()
      j += 1
      if j > 10: #empty video
        return None
  prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  # prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
  prev = cv2.resize(prev, (224, 168)) #
  prev = cv2.copyMakeBorder(prev, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  flow = []
  vid_len = _video_length(video_path)
  fc = 0
  i = 0
  max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
  while (fc < (framenum + 20) * frame_interval and ret and i < framenum):
    ret,   = cap.read()
    if fc % frame_interval == 0 and ret:
      # cv2.imshow('frame2', frame2)
      # cv2.waitKeyEx(-1)
      curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
      # curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
      curr = cv2.resize(curr, (224, 168))  #
      curr = cv2.copyMakeBorder(curr, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
      curr_flow = TVL1.calc(prev, curr, None)
      assert(curr_flow.dtype == np.float32)
      # truncate [-20, 20]
      curr_flow[curr_flow >= 20] = 20
      curr_flow[curr_flow <= -20] = -20
      # scale to [-1, 1]
      if max_val(curr_flow) != 0:
        curr_flow = curr_flow / max_val(curr_flow)
      else:
        curr_flow = curr_flow / 20. #devide NAN
      # cv2.imshow(str(fc),curr)
      # cv2.imshow('curr_flow', curr_flow)
      # cv2.waitKeyEx(-1)
      flow.append(curr_flow)
      prev = curr
      i += 1
    fc += 1
  cap.release()
  flow = np.array(flow)
  if flow.shape[0] < framenum:#if <15 frames break
    return None
  # flow = flow[None,:] #ye
  flow = flow.reshape(1,framenum,_IMAGE_SIZE, _IMAGE_SIZE, 2)
  # print(flow.shape)
  return flow

def _process_video_files(thread_index, filenames, save_to):
  for filename in filenames:
    flow = compute_TVL1(filename)
    if flow is None:
      continue
    fullname, _ = os.path.splitext(filename)
    split_name = fullname.split('/')
    # save_name = os.path.join(save_to, split_name[-2], split_name[-1] + '.npy')
    save_name = os.path.join(save_to, split_name[-3], split_name[-2], split_name[-1] + '.npy') # , './data/flow/00\\train\\125869795_4676_part_0.npy'
    np.save(save_name, flow)
    print("%s [thread %d]: %s done." % (datetime.now(), thread_index, filename))
    sys.stdout.flush()

def _process_dataset():
  import pdb
  pdb.set_trace()
  # filenames = [FLAGS.data_dir + "//" + class_fold + "//" + train_or_test + "//"+ filename #filename
  #              for class_fold in
  #                #tf.gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
  #                 os.listdir(FLAGS.data_dir) if 'zip' not in class_fold
  #                for filename in
  #                  # tf.gfile.Glob(os.path.join(class_fold, '*'))
  #                   os.listdir(FLAGS.data_dir + "//" + class_fold + "//" + train_or_test)
  #             ]
  #list1 = [FLAGS.data_dir + "//" + class_fold + "//" + train_or_test + "//"+ filename #filename
  #             for class_fold in
  #                os.listdir(FLAGS.data_dir) if 'zip' not in class_fold
  #               for filename in
  #                  os.listdir(FLAGS.data_dir + "//" + class_fold + "//" + train_or_test)
  #            ]
  if train_or_test == 'train':
    f = open(train_path)
  if train_or_test == 'test':
    f = open(test_path)
  train_info = []
  for line in f.readlines():
        info = line.strip().split(',')
        train_info.append(info[0])
  f.close()
  list2 = ["/" + class_fold + "/" + train_or_test + "/"+ filename
               for class_fold in
                  os.listdir(FLAGS.save_to) if 'zip' not in class_fold
                 for filename in
                    os.listdir(FLAGS.save_to + "//" + class_fold + "//" + train_or_test)
              ]
  #filenames = [i for i in train_info if i + '.npy' not in list2]
  filenames = [FLAGS.data_dir + '/'+i.split('/')[-3] + '/'+ i.split('/')[-2] +'/'+ i.split('/')[-1] + '.mp4' for i in train_info if i + '.npy' not in list2]
  # print(len(filenames))
  #print(filenames)
  time1 = time.time()
  filename_chunk = np.array_split(filenames, FLAGS.num_threads)
  threads = []

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Launch a thread for each batch.
  print("Launching %s threads." %  FLAGS.num_threads)
  for thread_index in range(FLAGS.num_threads):
    args = (thread_index, filename_chunk[thread_index], FLAGS.save_to)
    t = threading.Thread(target=_process_video_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d videos in data set '%s'." %
        (datetime.now(), len(filenames), FLAGS.name))
  duration = time.time() - time1
  print("Time total:%.2f, Per Video: %2f" %(duration, duration /len(filenames)))

def main(unused_argv):
  f = open(_CLASS_NAMES, 'r')#, encoding= 'utf-8')
  classes = [cls[:2] for cls in f.readlines() if cls[0] != '\n' ] #cls[:2]
  for cls in classes:
    path = FLAGS.save_to + '//' + cls + '//' + train_or_test
    if not tf.gfile.IsDirectory(path): #os.path.join(FLAGS.save_to, cls )
        tf.gfile.MakeDirs(path)

  _process_dataset()
  # buf = compute_TVL1('E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output/09/train/356692327_6617_part_0.mp4')
  # print(buf)
  # print(buf.shape)
  # i = 0
  # while i<buf.shape[1]:
  #     print('max', np.max(buf[0][i]))
  #     print('min', np.min(buf[0][i]))
  #     cv2.imshow('flow', buf[0][i])
  #     cv2.waitKeyEx(-1)
  #     i +=1
  #     print(i)
if __name__ == '__main__':
  app.run()
