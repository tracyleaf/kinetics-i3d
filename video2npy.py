import cv2
import numpy as np

cap = cv2.VideoCapture('v_Biking_g06_c02.avi') #'24881317_23_part_6.mp4'
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = 224 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
frameHeight = 224 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)
frameCount = 30

buf = np.empty((1, frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))#uint8

fc = 0
ret = True

while (fc < frameCount and ret):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(frameWidth, frameHeight), None , 0, 0, cv2.INTER_LINEAR)
    # print(frame)
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten()))) #rescale(-1,1)
    # normalizedImg = np.zeros((224, 224))
    # normalizedImg = cv2.normalize(frame, normalizedImg, alpha=-1, beta=1.0, norm_type=cv2.NORM_MINMAX)
    # buf[0][fc] = normalizedImg
    buf[0][fc] = (frame / float(max_val(frame))) * 2 -1
    print(min((buf[0][fc]).flatten()))
    # buf[0][fc] = frame / 255.
    fc += 1


cap.release()
np.save('v_Biking_g06_c02',buf) #24881317_23_part_6_rgb
print(buf.shape)
print(buf.shape)
cv2.namedWindow('frame 10')
cv2.imshow('frame 10', buf[0][9])

cv2.waitKey(0)

