import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#下载 one_hot 转化标签 便于识别
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


#定义变量 每个批次大小
batch_size = 100
#计算一共有几个批次
n_batch = mnist.train.num_examples // batch_size


#定义命名空间
with tf.name_scope("input"):
    #定义占位符
    x = tf.placeholder(tf.float32, [None, 784], name="x-input") #每一批100 none=100 每个28*28
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")#每一批100 none=100 每个10个

#定义命名空间
with tf.name_scope("layer"):
    #简单的神经网络
    with tf.name_scope("wights"):
        W = tf.Variable(tf.zeros([784, 10]), name="W")
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]), name="b")
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(wx_plus_b)#softmax 计算概率


#二次代价函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
#梯度下降
with tf.name_scope("train_step"):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope("accuracy-all"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("F:/python项目/tensorflow--验证码识别/神经网络结构图/", sess.graph)
    for epoch in range(1):
        #几个批次
        for batch in range(n_batch):
            #batch_size=100 取一个批次
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            #print("第%s周期" % epoch, '第%s批次' % batch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("-----------------------------")
        print("第%s周期" % epoch, '准确率%s' % acc)