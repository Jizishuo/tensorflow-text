import tensorflow as tf
from PIL import Image
from nets2 import nets_factory
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 字符集
CHAR_SET = [str(i) for i in range(10)]
CHAR_SET_LEN = len(CHAR_SET)
# 训练集大小
TRAIN_NUM = 5800
# 批次大小
BATCH_SIZE = 128
# 迭代次数
EPOCHES = 30
# 循环次数
LOOP_TIMES = EPOCHES*TRAIN_NUM//BATCH_SIZE
# tf文件
TFRECORD_FILE = 'F:/python项目/tensorflow--验证码识别/train.tfrecords'

# 初始学习率
LEARNING_RATE = 0.001

# 网络
x = tf.placeholder(tf.float32,[None,224,224])
y0 = tf.placeholder(tf.float32,[None])
y1 = tf.placeholder(tf.float32,[None])
y2 = tf.placeholder(tf.float32,[None])
y3 = tf.placeholder(tf.float32,[None])
lr = tf.Variable(LEARNING_RATE, dtype=tf.float32)

# 读取数据的函数
def read_and_decode(filename):
    tf_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tf_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':tf.FixedLenFeature([], tf.string),
                                            'label0': tf.FixedLenFeature([], tf.int64),
                                            'label1': tf.FixedLenFeature([], tf.int64),
                                            'label2': tf.FixedLenFeature([], tf.int64),
                                            'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image,[224,224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image, label0, label1, label2, label3

#获取图片数据和标签
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#随机打包
image_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3],
    batch_size=BATCH_SIZE,
    capacity=10000,
    min_after_dequeue=2000,
    num_threads=1
)

#'alexnet_v2_captcha_multi'
# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN*4,
    weight_decay=0.0005,
    is_training=True
)

with tf.Session() as sess:
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    logits, end_points = train_network_fn(X)
    one_hot_label0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    #吧便签转化为长度40 的向量
    label_40 = tf.concat([one_hot_label0, one_hot_label1, one_hot_label2, one_hot_label3], 1)
    loss_40 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_40))

    optimizer_40 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_40)

    # 计算准确率
    correct_pre_40 = tf.equal(tf.argmax(label_40, 1), tf.argmax(logits, 1))
    accuracy_40 = tf.reduce_mean(tf.cast(correct_pre_40, tf.float32))

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i_epoch = 0

    for i in range(LOOP_TIMES):
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, label0_batch, label1_batch, label2_batch, label3_batch])
        sess.run(optimizer_40, feed_dict={x: b_image,
                                       y0: b_label0,
                                       y1: b_label1,
                                       y2: b_label2,
                                       y3: b_label3})
        i_epoch_new = i // (5800 / BATCH_SIZE) + 1
        if i_epoch != i_epoch_new:
            i_epoch = i_epoch_new
            if i_epoch % 8 == 0:
                sess.run(tf.assign(lr, lr * 0.5))

        if i % 20 == 0:
            acc, loss_ = sess.run([accuracy_40, loss_40],
                                                     feed_dict={x: b_image,
                                                                y0: b_label0,
                                                                y1: b_label1,
                                                                y2: b_label2,
                                                                y3: b_label3})
            learning_rate = sess.run(lr)
            print("第:%d/%d 批次:%d,  Loss:%.3f  Accuracy:%.2f  Learning_rate:%.5f" % (
                i, LOOP_TIMES, i_epoch, loss_, acc, learning_rate))
        #if acc0 > 0.9 and acc1 > 0.9 and acc2 > 0.9 and acc3 > 0.9 and i == LOOP_TIMES - 1:
            saver.save(sess, 'F:/python项目/tensorflow--验证码识别/验证码识别net保存/model2/crack_captcha.model', global_step=i)

    coord.request_stop()
    coord.join(threads)