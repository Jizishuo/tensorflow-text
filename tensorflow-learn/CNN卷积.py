import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#对比函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    #输入xs 生成预测值y_pre> >y_pre是[0,1,....]类似 10个数据 每个都是0-1之间的概率 不是具体0或1
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    #计算100个中对了几个错了几个 概率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result


#定义weight
def weight_variable(shape):
    #产生随机变量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#定义bias
def bias_variable(shape):
    #开始是0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#卷积神经网络层 x 输入值(整一张图片) W = weight,
# strides不长 每一步跨多少,初始[1,,,1] [1,x轴跨步， y轴跨步，1]
#padding 有2种same跟原图一样大 填充边缘
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

#磁化conv2d 闯入x
def max_pool_2x2(x):
    #类似x y 多收集一次信息
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding="SAME")



#28*28=784   输出10个
xs = tf.placeholder(tf.float32, [None, 784])#28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  #防止过拟合

#处理图片[-1, 28, 28, 1]-1不管维度 28，28 个像素，1 黑白 rgb就是3
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#print(x_image.shapr) #[n_samples, 28, 28, 1]
#一层输入
W_conv1 = weight_variable([5,5,1,32]) #每一个一张图片取样 5*5 厚度从input=1变ouput=32
b_conv1 = bias_variable([32])#只有32个长度 32个卷积核？
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #y=ax+b --变成28*28*32
h_pool1 = max_pool_2x2(h_conv1)  #因为[1, 2, 2, 1]跨步 变成14*14*32

#二层输入
W_conv2 = weight_variable([5,5,32,64]) #每一个一张图片取样 5*5 厚度从input=32变ouput=64
b_conv2 = bias_variable([64])#只有64个长度
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #y=ax+b --变成14*14*64
h_pool2 = max_pool_2x2(h_conv2)  #因为[1, 2, 2, 1]跨步 变成7*7*64

#1处理层
# #取2层输出7*7*64--输出1024 变更高宽
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#[n_samples, 7,7,64]>>>[n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#矩阵相乘
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #防止过拟合

#2处理层
# #取2层输出1024--输出10 ___1到10个数字 10个判断
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#矩阵相乘 --预测值 f_fc1_drop 第一层被搞掉的值  softmax算概率
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #优化器 减少误差 1e-4=0.00001

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    #提取100个数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys,keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))