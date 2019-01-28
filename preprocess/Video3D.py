# ============================================================================
# 截取视频帧数并转换为rgb或flow的数据矩阵
# ============================================================================
import os
import numpy as np
from PIL import Image
from preprocess.data_aug import transform_data
import cv2
from cv2 import DualTVL1OpticalFlow_create as DualTVL1

_IMAGE_SIZE = 224
def get_frame_num(video_path):
    """
        计算视频帧数
    """
    video_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while(True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    return frame_count

def get_frames2(video_path, mode, frame_num, start=1, sample=1, data_augment=True, random_start=True, side_length=224):
    """
    获取视频帧输出array矩阵
    :param mode: rgb或flow
    :param frame_num: 截取视频的帧数
    :param data_augment: 数据增强
    :param random_start: 选取随机开始视频帧
    :return: frame_num * height * width * channel (rgb:3 , flow:2)
    """
    #assert frame_num <= self.total_frame_num
    total_frame_num = get_frame_num(video_path)
    start = start - 1
    if random_start:
        start = np.random.randint(max(total_frame_num-(frame_num-1)*sample, 1))
    frames = []
    video_cap = cv2.VideoCapture(video_path)
    for i in range(start, start + frame_num * sample):
        ret, frame = video_cap.read()
        if ret and ((i - start) % sample == 0):
            # frames.extend(load_img2(video_path, mode, i % total_frame_num + 1))
            frames.extend(load_img2(frame,mode))
        i += 1
    frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
    frames_np = []
    if mode == 'rgb':
        for i, img in enumerate(frames):
            frames_np.append(np.asarray(img))
        frames_np = np.expand_dims(frames_np, 0)
    elif mode == 'flow':
        for i in range(0, len(frames), 2):
            tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
            frames_np.append(tmp)
        frames_np = np.expand_dims(frames_np, 0)
    return np.stack(frames_np)

def load_img2(frame, mode):
    """
    根据rgb或flow,返回通道叠加的数据
    :param mode: rgb或flow
    :return:
    """
    if mode == 'rgb':
        return [Image.fromarray(frame)]
    elif mode == 'flow':
        # v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
        prev = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) #480*640
        return [prev,prev]
    else:
        raise ValueError('load_img2 error')
