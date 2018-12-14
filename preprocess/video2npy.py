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
frame_interval = 10
_CLASS_NAMES = 'label_kugou.txt'
# abspath = os.path.abspath(sys.argv[0])
DATA_DIR = '/data2/dataset/Video_8k_dataset/video_8k' #'E:/dataset/instruments_video/Video_8k_dataset/video_8k' #
SAVE_DIR = '/data2/ye/data/rgb/' #'./data/rgb'#
train_or_test = 'test'
train_path = '/data2/ye/instrument-detect/preprocess/video_8k_train_list_v3.txt'
test_path = '/data2/ye/instrument-detect/preprocess/video_8k_test_list_v3.txt'
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('save_to', SAVE_DIR, 'where to save flow data.')
flags.DEFINE_integer('num_threads',16, 'number of threads.') #32
flags.DEFINE_string('train_or_test', train_or_test, 'train or test dirs')

def _process_video_files(thread_index, filenames, save_to):
  for filename in filenames:
    flow = computeRGB(filename)
    fullname, _ = os.path.splitext(filename)
    split_name = fullname.split('/')
    # save_name = os.path.join(save_to, split_name[-2], split_name[-1] + '.npy')
    save_name = os.path.join(save_to, split_name[-3],split_name[-2], split_name[-1] + '.npy') #'./data/flow/00\\train\\125869795_4676_part_0.npy'
    np.save(save_name, flow)
    print("%s [thread %d]: %s done." % (datetime.now(), thread_index, filename))
    sys.stdout.flush()

def computeRGB(video_path):
    cap = cv2.VideoCapture(video_path)  # '24881317_23_part_6.mp4'
    buf = np.zeros((1, frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))#uint8  empty
    fc = 0
    i = 0
    ret = True
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    while (fc < (frameCount+20)*frame_interval and ret and i < frameCount):
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
        # else:
            # frame = (frame / float(max_val(frame)) + 1) * 2 -1 #加1防止分母为0，或frame = (frame /255. -0.5) * 2
    cap.release()
    return buf
    # np.save('24881317_23_part_6',buf)

def _process_dataset():
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
  print(len(filenames))
  # print(filenames)
  # f2 = open('C:/Users/aiyanye/Desktop/rgb-1.txt')
  # list2 = [i.split(',')[0] for i in f2.readlines()]
  # # list3 = ['/07/train/718334762_302_part_6,7\n', '/09/train/481834174_18838_part_19,9\n', '/00/train/681277329_1018_part_17,0\n', '/07/train/391186958_11605_part_5,7\n', '/05/train/51034854_3215_part_7,5\n', '/07/train/955483775_1168_part_8,7\n', '/09/train/420411263_13260_part_10,9\n', '/09/train/354709921_6504_part_12,9\n', '/00/train/602221806_32176_part_13,0\n', '/00/train/697213070_51_part_14,0\n', '/07/train/718334762_302_part_18,7\n', '/00/train/125869795_4676_part_1,0\n', '/09/train/611323248_32448_part_7,9\n', '/09/train/89356062_3993_part_16,9\n', '/08/train/951669046_1122_part_9,8\n', '/06/train/791838261_735_part_6,6\n', '/00/train/396479256_11947_part_8,0']
  # list3 = ['/07/train/718334762_302_part_6,7\n'] #'/07/train/718334762_302_part_6,7\n',
  # list2 = [i.split(',')[0] for i in list3]
  # filenames = [FLAGS.data_dir  + i +'.mp4' for i in list2]
  # print(filenames)
  # f2.close()
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
  # import pdb
  # pdb.set_trace()
  _process_dataset()
  # buf = computeRGB('E:/dataset/instruments_video/kugou_mv_dataset_part_v1/CutVideo_output/09/train/356692327_6617_part_0.mp4')
  # print(buf)
  # print(buf.shape)
  # i = 0
  # while i<buf.shape[1]:
  #     print('max', np.max(buf[0][i]))
  #     print('min', np.min(buf[0][i]))
  #     cv2.imshow('123',buf[0][i])
  #     cv2.waitKeyEx(-1)
  #     i +=1
  #     print(i)

if __name__ == '__main__':
  app.run()
    # main()