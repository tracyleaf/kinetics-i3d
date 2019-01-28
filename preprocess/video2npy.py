# ============================================================================
# 计算RGB数据并输出保存为.npy文件
# ============================================================================
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from datetime import datetime
import threading
from tensorflow.python.platform import app, flags
import tensorflow as tf
import os
import sys
import time

_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224
frameWidth = 224
frameHeight = 224
frameCount = 15
frame_interval = 10
_CLASS_NAMES = 'label_kugou.txt'
# abspath = os.path.abspath(sys.argv[0])
DATA_DIR = '/data2/dataset/Video_9k_dataset_v3/video_9k'
SAVE_DIR = '/data2/ye/data/rgb'
train_or_test = 'test'
train_path = '/data2/ye/instrument-detect/preprocess/video_9k_train_list_v2.txt'
test_path = '/data2/ye/instrument-detect/preprocess/video_9k_test_list_v2.txt'
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('save_to', SAVE_DIR, 'where to save flow data.')
flags.DEFINE_integer('num_threads',16, 'number of threads.') #32
flags.DEFINE_string('train_or_test', train_or_test, 'train or test dirs')

def _process_video_files(thread_index, filenames, save_to):
    for filename in filenames:
        flow = computeRGB(filename)
        if np.sum(flow) == 0:
            break
        fullname, _ = os.path.splitext(filename)
        split_name = fullname.split('/')
        save_name = os.path.join(save_to, split_name[-3],split_name[-2], split_name[-1] + '.npy')
        np.save(save_name, flow)
        print("%s [thread %d]: %s done." % (datetime.now(), thread_index, filename))
        sys.stdout.flush()

def computeRGB(video_path):
    """
       计算RGB数据，并归一化数据到[-1,1]，size为224*224
       不足224*224则在上下加黑边
    """
    cap = cv2.VideoCapture(video_path)
    buf = np.zeros((1, frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))
    fc = 0
    i = 0
    ret = True
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    while (fc < (frameCount + 20)*frame_interval and ret and i < frameCount):
        ret, frame = cap.read()
        if fc% frame_interval == 0 and ret: #前10帧数据丢弃
            if max_val(frame) != 0:
                frame = cv2.resize(frame, (224, 168), None, 0, 0, cv2.INTER_LINEAR)  # width,height
                frame = (frame / float(max_val(frame))) * 2 -1 #rescale(-1,1)
                # frame = (frame /255. -0.5) * 2
                frame = cv2.copyMakeBorder(frame, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(-1, -1, -1)) #加黑边
                buf[0][i] = frame
                i += 1
        fc += 1
    cap.release()
    if i < frameCount - 3: #允许3帧为空
        buf = np.zeros((1, frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))
    return buf

def _process_dataset():
    """
      多线程处理数据
      线程数为FLAGS.num_threads
    """
    # import pdb
    # pdb.set_trace()
    # list1 = [FLAGS.data_dir + "//" + class_fold + "//" + train_or_test + "//" + filename  # filename
    #          for class_fold in
    #          os.listdir(FLAGS.data_dir) if 'zip' not in class_fold
    #          for filename in
    #          os.listdir(FLAGS.data_dir + "//" + class_fold + "//" + train_or_test)
    #          ]
    # list2 = [FLAGS.data_dir + "//" + class_fold + "//" + train_or_test + "//" + filename  # filename
    #          for class_fold in
    #          os.listdir(FLAGS.save_to) if 'zip' not in class_fold
    #          for filename in
    #          os.listdir(FLAGS.save_to + "//" + class_fold + "//" + train_or_test)
    #          ]
    # filenames = [i for i in list1 if i[:-4] + '.npy' not in list2]
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
    filenames = [FLAGS.data_dir + '/' + i.split('/')[-3] + '/' + i.split('/')[-2] + '/' + i.split('/')[-1] + '.mp4' for i
               in train_info if i + '.npy' not in list2]
    print(len(filenames))
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
        (datetime.now(), len(filenames), FLAGS.train_or_test))
    duration = time.time() - time1
    print("Time total:%.2f, Per Video: %2f" %(duration, duration /len(filenames)))

def main(unused_argv):
    f = open(_CLASS_NAMES, 'r')#, encoding= 'utf-8')
    classes = [cls[:2] for cls in f.readlines() if cls[0] != '\n' ] #cls[:2]
    for cls in classes:
        path = FLAGS.save_to + '//' + cls + '//' + train_or_test
        if not tf.gfile.IsDirectory(path):
            tf.gfile.MakeDirs(path)
    _process_dataset()

if __name__ == '__main__':
    app.run()
    # main()