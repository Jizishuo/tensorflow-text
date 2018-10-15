import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 每个批次大小
batch_size = 100
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # 生成一个截断的正态分布


# 初始化偏制
def bias_vairable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


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


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#改变x的格式转为4d向量[batch,height, wight, channels] -1表示批次传入值=这个模型是100 1 一维图片黑白 彩色3
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])  # 5*5的采样窗口，黑白是1 彩色是3 32个卷积核从1个平面抽取特征
b_conv1 = bias_vairable([32])  # 每个卷积核一个偏置值

# 28*28*1 的图片卷积之后变为28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化之后变为 14*14*32
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积之后变为 14*14*64
W_conv2 = weight_variable([5, 5, 32, 64])# 5*5的采样窗口，高度32 64个卷积核从1个平面抽取特征
b_conv2 = bias_vairable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二次池化之后变为 7*7*64
h_pool2 = max_pool_2x2(h_conv2)

# 第一个全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])#上一层有7 * 7 * 64个神经元，全连接层1024个神经元
b_fc1 = bias_vairable([1024])#1024个偏制

# 池化结束的7*7*64的图像变成1维向量 -1代表任意值 3维变1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_vairable([10])
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.sigmoid(logits)#变概率

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#准确率
prediction_2 = tf.nn.softmax(prediction)
correct_prediction = (tf.equal(tf.argmax(prediction_2, 1), tf.argmax(y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("第%s周期" % epoch, '准确率%s' % acc)
