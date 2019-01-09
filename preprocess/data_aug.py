from __future__ import division
import random
from PIL import ImageOps,Image
import os
import cv2
import numpy as np

def transform_data(data, scale_size=256, crop_size=224, random_crop=False, random_flip=False):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    # data = Image.fromarray(np.asarray(data))
    if random_crop:
        x0 = random.randint(0, width - crop_size)
        y0 = random.randint(0, height - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
            # data[i] = img[y0:y1,x0:x1]
    # else:
        x0 = int((width-crop_size)/2)
        y0 = int((height-crop_size)/2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
            # data[i] = img[y0:y1, x0:x1]
    # if random_flip and random.randint(0,1) == 0:
    #     for i, img in enumerate(data):
    #         data[i] = ImageOps.mirror(img)

    return data
#
# def get_10_crop(data, scale_size=256, crop_size=224):
#     data = resize(data, scale_size)
#     width = data[0].shape[0]
#     height = data[0].shape[1]
#     top_left = [[0, 0],
#                 [width-crop_size, 0],
#                 [int((width-crop_size)/2), int((height-crop_size)/2)],
#                 [0, height-crop_size],
#                 [width-crop_size, height-crop_size]]
#     crop_data = []
#     for point in top_left:
#         non_flip = []
#         flip = []
#         x_0 = point[0]
#         y_0 = point[1]
#         x_1 = x_0 + crop_size
#         y_1 = y_0 + crop_size
#         for img in data:
#             tmp = img.crop((x_0, y_0, x_1, y_1))
#             non_flip.append(tmp)
#             flip.append(ImageOps.mirror(tmp))
#         crop_data.append(non_flip)
#         crop_data.append(flip)
#     return  crop_data

def resize(data, scale_size):
    height = data[0].size[0]
    width = data[0].size[0]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    # print('sd', np.asarray(data).shape)
    for i, image in enumerate(data):
        # data[i] = np.resize(image,(scale_size, scale_size))
        data[i] = image.resize((scale_size, scale_size))
        # print('sd',np.asarray(data).shape)
    return  data

