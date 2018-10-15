import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

#参数
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)#平均值
        tf.summary.scalar("mean", mean)#平均值 起名字
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev) #标准差
        tf.summary.scalar("max", tf.reduce_max(var))#最大值
        tf.summary.scalar("mix", tf.reduce_min(var))#最小值
        tf.summary.histogram("histogram", var)#直方图


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input") #每一批100 none=100 每个28*28
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")#每一批100 none=100 每个10个


with tf.name_scope("layer"):
    with tf.name_scope("wights"):
        W = tf.Variable(tf.zeros([784, 10]), name="W")
        variable_summaries(W)
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]), name="b")
        variable_summaries(b)
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(wx_plus_b)#softmax 计算概率



with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train_step"):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope("accuracy-all"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("F:/python项目/tensorflow--验证码识别/神经网络结构图/", sess.graph)
    for epoch in range(21):
        #几个批次
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #挤上去同时运行[merged, train_step]
            #sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
        #记录
        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("-----------------------------")
        print("第%s周期" % epoch, '准确率%s' % acc)