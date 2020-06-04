import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from nets import nets_factory
import numpy as np
import os
import time

s_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


CHAR_SET_LEN = 10   #训练结果个数
#图片高度宽度
IMAGE_HIGHT = 28
IMAGE_WEIGHT = 75

# 批次大小
BATCH_SIZE = 1  #可以改

# tf文件
TFRECORD_FILE = 'F:/python项目/work-gongdan/新验证码/tfrecord_data/test.tfrecords'

# 网络
x = tf.placeholder(tf.float32, [None, 224, 224])

# 读取数据的函数
def read_and_decode(filename):
    #tf 队列
    tf_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 读取图片
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [224, 224])

    # 木有预处理灰度图展示
    image_raw = tf.reshape(image, [224, 224])

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image, image_raw, label0, label1, label2, label3


image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#打乱顺序
image_batch, image_batch_raw, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3],
    batch_size=BATCH_SIZE,  #批次大小
    capacity=10000,         #队列大小
    min_after_dequeue=2000, #最小队列个数
    num_threads=1,          #线程数
)

# 'alexnet_v2_captcha_multi'
# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,  #要数据结果个数 10个数字 默认10000
    weight_decay=0.0005,
    is_training=False,          # 是否需要训练
)


with tf.Session() as sess:

    # end_points
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

    # 预测
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
    predict0 = tf.argmax(predict0, 1)

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
    predict1 = tf.argmax(predict1, 1)

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
    predict2 = tf.argmax(predict2, 1)

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
    predict3 = tf.argmax(predict3, 1)

    #保存模型+启动线程队列
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, './net_save/model2/crack_captcha.model-1000')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(6):
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, image_batch_raw,  label0_batch, label1_batch, label2_batch, label3_batch])

        #显示图片
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        #打印标签
        print("标签:", b_label0, b_label1, b_label2, b_label3)
        #预测
        label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x:b_image})
        print("预测:", label0, label1, label2, label3)

    coord.request_stop()
    coord.join(threads)

e_time = time.time()

print("总共运行%s秒" % (e_time-s_time))