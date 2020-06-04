import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 字符集
# CHAR_SET = [str(i) for i in range(10)]
# CHAR_SET_LEN = len(CHAR_SET)   #训练结果个数
CHAR_SET_LEN = 10   #训练结果个数
#图片高度宽度
IMAGE_HIGHT = 28
IMAGE_WEIGHT = 75
# 训练集大小
# TRAIN_NUM = 5800
# 批次大小
# BATCH_SIZE = 128
BATCH_SIZE = 25  #可以改
# 迭代次数
EPOCHES = 30
# 循环次数
#LOOP_TIMES = EPOCHES * TRAIN_NUM // BATCH_SIZE 5800*30 //128
LOOP_TIMES = 2000
# tf文件
TFRECORD_FILE = 'F:/python项目/work-gongdan/新验证码/tfrecord_data/train.tfrecords'

# 初始学习率
LEARNING_RATE = 0.001

# 网络
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])
lr = tf.Variable(LEARNING_RATE, dtype=tf.float32)


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
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image, label0, label1, label2, label3


image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

#打乱顺序
image_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3],
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
    is_training=True,          # 是否需要训练
)




with tf.Session() as sess:
    # end_points
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

    one_hot_label0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_label3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)

    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label0, logits=logits0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label1, logits=logits1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label2, logits=logits2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label3, logits=logits3))
    # 计算总误差
    total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

    # 计算准确率
    correct_pre0 = tf.equal(tf.argmax(one_hot_label0, 1), tf.argmax(logits0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_pre0, tf.float32))

    correct_pre1 = tf.equal(tf.argmax(one_hot_label1, 1), tf.argmax(logits1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_pre1, tf.float32))

    correct_pre2 = tf.equal(tf.argmax(one_hot_label2, 1), tf.argmax(logits2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_pre2, tf.float32))

    correct_pre3 = tf.equal(tf.argmax(one_hot_label3, 1), tf.argmax(logits3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_pre3, tf.float32))


    # 保存模型
    saver = tf.train.Saver(max_to_keep=2)

    #保存模型+启动线程队列
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    i_epoch = 0

    for i in range(LOOP_TIMES):
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, label0_batch, label1_batch, label2_batch, label3_batch])
        sess.run(optimizer, feed_dict={x: b_image,
                                       y0: b_label0,
                                       y1: b_label1,
                                       y2: b_label2,
                                       y3: b_label3})
        # i_epoch_new = i // (5800 / BATCH_SIZE) + 1
        # if i_epoch != i_epoch_new:
        #     i_epoch = i_epoch_new
        #     if i_epoch % 8 == 0:
        #         sess.run(tf.assign(lr, lr * 0.5))


        if i % 20 == 0:
            # if i % 20 == 0:  # 什么时候更新lr
            #     sess.run(tf.assign(lr, lr*0.5))
            acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                                                     feed_dict={x: b_image,
                                                                y0: b_label0,
                                                                y1: b_label1,
                                                                y2: b_label2,
                                                                y3: b_label3})
            learning_rate = sess.run(lr)
            print("进度:%d/%d  误差:%.3f  准确度:%.2f,%.2f,%.2f,%.2f  学习率:%.5f" % (
                i, LOOP_TIMES, loss_, acc0, acc1, acc2, acc3, learning_rate))
            # if acc0 > 0.1 or i == LOOP_TIMES - 1:
            #     saver.save(sess, './net_save/model2/crack_captcha.model', global_step=i)
        if i % 500 == 0:
            saver.save(sess, './net_save/model2/crack_captcha.model', global_step=i)


    coord.request_stop()
    coord.join(threads)
