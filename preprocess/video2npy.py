import cv2
import numpy as np
from datetime import datetime
import threading
from tensorflow.python.platform import app, flags
import tensorflow as tf
import os
import sys

_EXT = ['.avi', '.mp4']
_IMAGE_SIZE = 224
# cap = cv2.VideoCapture('24881317_23_part_6.mp4') #'24881317_23_part_6.mp4'
# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = 224 # 224 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
frameHeight = 224 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)
frameCount = 15
_CLASS_NAMES = 'label_kugou.txt'
DATA_DIR = 'E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output/' #'./video'#'./tmp/HMDB/videos'
SAVE_DIR = './data/rgb/'
train_or_test = 'test' #main函数中也需要修改

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('save_to', SAVE_DIR, 'where to save flow data.')
flags.DEFINE_integer('num_threads',2, 'number of threads.') #32
flags.DEFINE_string('train_or_test', train_or_test, 'train or test dirs')

def _process_video_files(thread_index, filenames, save_to):
  for filename in filenames:
    flow = computeRGB(filename)
    fullname, _ = os.path.splitext(filename)
    split_name = fullname.split('/')
    # save_name = os.path.join(save_to, split_name[-2], split_name[-1] + '.npy')
    save_name = os.path.join(save_to, split_name[-5], split_name[-3], split_name[-1] + '.npy') #'./data/flow/00\\train\\125869795_4676_part_0.npy'
    np.save(save_name, flow)
    print("%s [thread %d]: %s done." % (datetime.now(), thread_index, filename))
    sys.stdout.flush()

def computeRGB(video_path):
    cap = cv2.VideoCapture(video_path)  # '24881317_23_part_6.mp4'
    buf = np.empty((1, frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))#uint8
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, frame = cap.read()
        frame = cv2.resize(frame,(224, 168), None , 0, 0, cv2.INTER_LINEAR) #width,height
        max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten()))) #rescale(-1,1)
        frame = (frame / float(max_val(frame))) * 2 -1
        frame = cv2.copyMakeBorder(frame,28,28,0,0, cv2.BORDER_CONSTANT, value =(255,255,255))
        buf[0][fc] = frame
        fc += 1
    cap.release()
    return buf
    # np.save('24881317_23_part_6',buf) #24881317_23_part_6_rgb
# print(buf.shape)
# cv2.namedWindow('frame 10')
# cv2.imshow('frame 10', buf[0][9])
# cv2.waitKey(0)

def _process_dataset():
  filenames = [FLAGS.data_dir + "//" + class_fold + "//" + train_or_test + "//"+ filename #filename
               for class_fold in
                 #tf.gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
                  os.listdir(FLAGS.data_dir)
                 for filename in
                   # tf.gfile.Glob(os.path.join(class_fold, '*'))
                   #  os.listdir(FLAGS.data_dir + "//" + class_fold）
                    os.listdir(FLAGS.data_dir + "//" + class_fold + "//" + train_or_test)
              ]
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

def main(unused_argv):
  if not tf.gfile.IsDirectory(FLAGS.save_to):
    tf.gfile.MakeDirs(FLAGS.save_to)
    f = open(_CLASS_NAMES, 'r', encoding= 'utf-8')
    # classes = [cls.strip() for cls in f.readlines()]
    classes = [cls[:2] for cls in f.readlines() if cls[0] != '\n' ]
    for cls in classes:
        tf.gfile.MakeDirs(os.path.join(FLAGS.save_to, cls + '//' + train_or_test))
  if train_or_test == 'test':
    f = open(_CLASS_NAMES, 'r', encoding='utf-8')
    # classes = [cls.strip() for cls in f.readlines()]
    classes = [cls[:2] for cls in f.readlines() if cls[0] != '\n']
    for cls in classes:
      if not tf.gfile.IsDirectory(os.path.join(FLAGS.save_to, cls + '//' + train_or_test)):
        tf.gfile.MakeDirs(os.path.join(FLAGS.save_to, cls + '//' + train_or_test))

  _process_dataset()

if __name__ == '__main__':
  app.run()