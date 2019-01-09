import os
import numpy as np
from PIL import Image
from preprocess.data_aug import transform_data
import cv2
from cv2 import DualTVL1OpticalFlow_create as DualTVL1

_IMAGE_SIZE = 224
def get_frame_num(video_path):
    video_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while(True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    return frame_count

def get_frames2(video_path, mode, frame_num, start=1, sample=1, data_augment=True, random_start=True, side_length=224):
    '''
        return:
            frame_num * height * width * channel (rgb:3 , flow:2)
    '''
    #assert frame_num <= self.total_frame_num
    total_frame_num = get_frame_num(video_path)
    start = start - 1
    if random_start:
        start = np.random.randint(max(total_frame_num-(frame_num-1)*sample, 1))
    frames = []
    video_cap = cv2.VideoCapture(video_path)
    for i in range(start, start + frame_num * sample):
        ret, frame = video_cap.read()
        # print(ret,i)
        if ret and ((i - start) % sample == 0):
            # frames.extend(load_img2(video_path, mode, i % total_frame_num + 1))
            frames.extend(load_img2(frame,mode))
        i += 1
    # print(np.asarray(frames).shape)
    frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
    # cv2.imshow(frames)
    frames_np = []
    if mode == 'rgb':
            # cv2.imshow('tmp', np.asarray(img))
            # cv2.waitKey(-1)
        for i, img in enumerate(frames):
            frames_np.append(np.asarray(img))
        frames_np = np.expand_dims(frames_np, 0)
    elif mode == 'flow':
        for i in range(0, len(frames), 2):
            tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
            # cv2.imshow('tmp',tmp)
            frames_np.append(tmp)
        frames_np = np.expand_dims(frames_np, 0)
    return np.stack(frames_np)

def load_img2(frame, mode):
    if mode == 'rgb':
        return [Image.fromarray(frame)]
        # return [frame]
    elif mode == 'flow':
        # v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
        prev = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) #480*640
        # img = Image.fromarray(frame).convert('L')
        # cv2.imshow('u_img',prev)
        # cv2.waitKey(-1)
        return [prev,prev]
    else:
        raise ValueError('load_img2 error')
#
#
# def compute_TVL1(video_path):
#   """Compute the TV-L1 optical flow."""
#   TVL1 = DualTVL1()
#   cap = cv2.VideoCapture(video_path)
#   for i in range(frame_interval):#former 10 frames deserted
#     ret = False
#     j = 0
#     while not ret:
#       ret, frame1 = cap.read()
#       j += 1
#       if j > 10: #empty video
#         return None
#   prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#   # prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
#   prev = cv2.resize(prev, (224, 168)) #
#   prev = cv2.copyMakeBorder(prev, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#   flow = []
#   vid_len = _video_length(video_path)
#   fc = 0
#   i = 0
#   max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
#   while (fc < (framenum + 20) * frame_interval and ret and i < framenum):
#     ret, frame2 = cap.read()
#     if fc % frame_interval == 0 and ret:
#       # cv2.imshow('frame2', frame2)
#       # cv2.waitKeyEx(-1)
#       curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#       # curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
#       curr = cv2.resize(curr, (224, 168))  #
#       curr = cv2.copyMakeBorder(curr, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
#       curr_flow = TVL1.calc(prev, curr, None)
#       assert(curr_flow.dtype == np.float32)
#       # truncate [-20, 20]
#       curr_flow[curr_flow >= 20] = 20
#       curr_flow[curr_flow <= -20] = -20
#       # scale to [-1, 1]
#       if max_val(curr_flow) != 0:
#         curr_flow = curr_flow / max_val(curr_flow)
#       else:
#         curr_flow = curr_flow / 20. #devide NAN
#       # cv2.imshow(str(fc),curr)
#       # cv2.imshow('curr_flow', curr_flow)
#       # cv2.waitKeyEx(-1)
#       flow.append(curr_flow)
#       prev = curr
#       i += 1
#     fc += 1
#   cap.release()
#   flow = np.array(flow)
#   if flow.shape[0] < framenum:#if <15 frames break
#     return None
#   # flow = flow[None,:] #ye
#   flow = flow.reshape(1,framenum,_IMAGE_SIZE, _IMAGE_SIZE, 2)
#   # print(flow.shape)
#   return flow

# a = get_frames2('E:/dataset/instruments_video/Video_9k_dataset_v3/video_9k/00/train/24881317_15_part_0.mp4','flow',16)
# print(a.shape)