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

#定义占位符
x = tf.placeholder(tf.float32, [None, 784]) #每一批100 none=100 每个28*28
y = tf.placeholder(tf.float32, [None, 10])#每一批100 none=100 每个10个

#简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)#softmax 计算概率


#二次代价函数
#loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#返回判对错 equal对比2个数 相同true 不同 false argmax(y, 1)求y标签最大的值(数字和概率都行)在哪一个位置=数字 1是比较范围
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#求准确率 cast把布尔值 变float32 true=1 flase=0 reduce_mean求平均值=准确率（几个1几个0）
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    #21个周期，一个周期整个批次
    for epoch in range(21):
        #几个批次
        for batch in range(n_batch):
            #batch_size=100 取一个批次
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            #print("第%s周期" % epoch, '第%s批次' % batch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("-----------------------------")
        print("第%s周期" % epoch, '准确率%s' % acc)