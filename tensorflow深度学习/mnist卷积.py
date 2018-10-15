from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf


#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error



#定义一个初始化权重的函数
def weigth_va(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w

#定义一个初始化偏置的函数
def bias_va(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b

#定义模型得出输出
def models():
    """
    自定义卷积模型
    """

    #准备数据的占位符x[none, 784],y_true=[none, 10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    #卷积层1 卷积池化
    with tf.variable_scope('conv1'):
        #初始化权重 5*5*1,步长1
        w_con1 = weigth_va([5, 5, 1, 32])
        b_con1 = bias_va([32])

        #对x 改变形状[none, 784]--[none, 28, 28, 1] none 不知道填-1
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        #[none ,28, 28, 1]---[none, 28,28,32] #stride步长1  激活函数
        x_relu = tf.nn.relu(tf.nn.conv2d(x_reshape, w_con1, strides=[1, 1, 1, 1], padding='SAME') + b_con1)

        #池化
        #2*2, 步长2 [none, 28, 28, 32]--[none, 14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #卷积2  5*5*32, 64个fifter， 步长1
    with tf.variable_scope('conv2'):
        #初始化权重 5*5*32,
        w_con2 = weigth_va([5, 5, 32, 64])
        b_con2 = bias_va([64])

        #卷积 池化
        #[none, 14, 14, 32]==[none, 14,14,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_con2, strides=[1,1,1,1], padding='SAME')+ b_con2)

        #池化[none, 14,14,64]==[none,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #全连接层 [none,7,7,64]---[none, 7*7*64] * [7*7*64, 10] + [10] = [none, 10]
    with tf.variable_scope('FC'):
        #初始化权重偏置
        w_fc = weigth_va([7*7*64, 10])
        b_fc = bias_va([10])

        #修改形状[none,7,7,64]---[none, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7*7*64])

        #进行矩阵运算
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

        return x, y_true, y_predict


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x, y_true, y_predict = models()

#交叉熵损失运算
with tf.variable_scope('soft_loss'):
    # 求平均交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

with tf.variable_scope('optimizer'):
    # 梯度下降求损失
    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# 计算准确率
with tf.variable_scope('acc'):
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    # equal_list none个样本【0，0，0，0，1。。。】求平均值
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    #循环训练
    for i in range(1000):
        mnist_x, mnist_y = mnist.train.next_batch(50)

        sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
        ac = sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})
        print("%s次训练，准确率%s" % (i, ac))


#goole-net
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base