import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 每个批次大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
log_dir = "F:/python项目/tensorflow--验证码识别/CNN网络图/"
max_step = 21
keep_ = 0.8


# 参数
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)  # 平均值
        tf.summary.scalar("mean", mean)  # 平均值 起名字
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)  # 标准差
        tf.summary.scalar("max", tf.reduce_max(var))  # 最大值
        tf.summary.scalar("mix", tf.reduce_min(var))  # 最小值
        tf.summary.histogram("histogram", var)  # 直方图


# 初始化权值
def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)  # 生成一个截断的正态分布


# 初始化偏制
def bias_vairable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


# 卷积层
def conv2d(x, W):
    # conv2d--二维卷及操作
    # x 是一个tensor 4维[batch,height, wight, channels]batch批次大小 长宽 channels通道数
    # w 是一个tensor（录波器）[height,wight,channels, out-channels]
    # strides步长[1,*,*,1]固定 第二个1 代表x方向步长， 第三个代表 y方向步长
    # padding 2中 SAME(补0) VALID（不补0）
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # max_pool最大值池化，取卷积核里最大值
    # 同x， strides卷积核
    # ksize 窗口大小[1,*,*,1]固定 第二。三个池化核2*2 大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input")
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("x_image"):
        # 改变x的格式转为4d向量[batch,height, wight, channels] -1表示批次传入值=这个模型是100 1 一维图片黑白 彩色3
        x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope("Conv-1"):
    with tf.name_scope("Conv-1-W"):
        W_conv1 = weight_variable([5, 5, 1, 32],name='W_conv1')  # 5*5的采样窗口，黑白是1 彩色是3 32个卷积核从1个平面抽取特征
    with tf.name_scope("Conv-1-b"):
        b_conv1 = bias_vairable([32],name='b_conv1')  # 每个卷积核一个偏置值

    with tf.name_scope("conv2d-1"):
        # 28*28*1 的图片卷积之后变为28*28*32
        conv2d = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d)
    with tf.name_scope("h-pool-1"):
        # 池化之后变为 14*14*32
        h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope("Conv-2"):
    with tf.name_scope("Conv-2-W"):
        # 第二次卷积之后变为 14*14*64
        W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")  # 5*5的采样窗口，高度32 64个卷积核从1个平面抽取特征
    with tf.name_scope("Conv-2-b"):
        b_conv2 = bias_vairable([64], name="b_conv2")

    with tf.name_scope("conv2d-2"):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope("relu"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("h-pool-2"):
        # 第二次池化之后变为 7*7*64
        h_pool2 = max_pool_2x2(h_conv2)
'''
# 第二次卷积 14*14*32->14*14*64
conv_layer2 = conv_layer(h_pool1, [5, 5, 32, 64], 'conv_layer2')
# 第二次池化之后变为 7*7*64
with tf.name_scope('Max_pool2'):
    h_pool2 = max_pool_2x2(conv_layer2)
'''
with tf.name_scope("Fc-1"):
    # 第一个全连接层
    with tf.name_scope("Fc-1-W"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name="Fc-1-W")  # 上一层有7 * 7 * 64个神经元，全连接层1024个神经元
    with tf.name_scope("Fc-1-b"):
        b_fc1 = bias_vairable([1024],name="Fc-1-b")  # 1024个偏制

    with tf.name_scope("fc-h_pool2"):
        # 池化结束的7*7*64的图像变成1维向量 -1代表任意值 3维变1维
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name="h_pool2_flat")
    with tf.name_scope("Fc-1-wx-b1"):
        ws_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(ws_plus_b1)
    with tf.name_scope("keep_prob"):
        keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    with tf.name_scope("Fc-1-h-drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name="Fc-1-h-drop")

with tf.name_scope("Fc-2"):
    # 第二个全连接层
    with tf.name_scope("Fc-2-W"):
        W_fc2 = weight_variable([1024, 10], name="Fc-2-W")
    with tf.name_scope("Fc-2-b"):
        b_fc2 = bias_vairable([10], name="Fc-2-W")
    with tf.name_scope("Fc-1-wx-b2"):
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope("sigmoid"):
        prediction = tf.nn.sigmoid(logits)  # 变概率

with tf.name_scope("loss-all"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction),name="loss-1")
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.name_scope("accuracy"):
    # 准确率
    with tf.name_scope("correct_prediction"):
        correct_prediction = (tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

#合并所有summary
merged = tf.summary.merge_all()

'''
def get_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(batch_size)
        k = keep_
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y: ys, keep_prob: k}

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    sess.run(tf.global_variables_initializer())

    for i in range(max_step):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=get_dict(False))
            test_writer.add_summary(summary, i)
            print("第%s周期" % i, '准确率%s' % acc)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=get_dict(True))
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
'''
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    for i in range(max_step):
        #训练集
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        #记录
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        #测试集
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        # 记录
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        if i % 2 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            print("第%s周期" % i, '训练集准确率%s' % train_acc, '训练集准确率%s' % test_acc,)