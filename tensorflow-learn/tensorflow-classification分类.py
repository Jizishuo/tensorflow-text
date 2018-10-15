import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

#导入数据 下载
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


#对比函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    #输入xs 生成预测值y_pre> >y_pre是[0,1,....]类似 10个数据 每个都是0-1之间的概率 不是具体0或1
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    #计算100个中对了几个错了几个 概率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

#神经层函数
def add_layer(input, input_size, out_size, activation_function = None):
    #随机生成变量矩阵
    Weights = tf.Variable(tf.random_normal([input_size, out_size]))
    #类似列表 1行 和out_size那么多列  让初始值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        #activation_function = None 说明是线性关系 不需要激活函数
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#28*28=784   输出10个
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

#一层输入
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
#误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #优化器 减少误差

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    #提取100个数据
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))