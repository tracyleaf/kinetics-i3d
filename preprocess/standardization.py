import tensorflow as tf
import cv2 as cv

image = cv.imread("E:/open Source/FaceRank_FaceCrop/FaceDetect_v2/TestImg_save/10445931.jpg")
cv.imshow("input", image)
std_image = tf.image.per_image_standardization(image)
image2 = image/255.
cv.imshow("image2", image2)
with tf.Session() as sess:
    result = sess.run(std_image)
    print(result)
    cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()