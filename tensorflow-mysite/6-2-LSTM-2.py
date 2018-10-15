import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#输入图片28*28
n_inputs = 28 #输入一行有28个数据
max_time = 28 #一共28行
lstm_size = 100 #隐藏层神经元
n_class = 10 #10个分类
batch_size = 50 #每个批次50个样本
n_batch = mnist.train.num_examples // batch_size #计算一共有几个批次

#none 表示任意长度
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 初始化权值
weight = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev=0.1))  # 生成一个截断的正态分布

# 初始化偏制
biases = tf.Variable(tf.constant(0.1, shape=[n_class]))

def RNN(X, weight, biases):
    #inputs = [batch-size, max-time, n-inputs][50, 28 ,28 ]
    #tf.nn.dynamic_rnn闯入的input格式固定
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    #定义LSTM基本的CELL
    #lstm_cell = tf.contrib.rnn.core_rnn.cell.BasicLSTMCell(lstm_size)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    #final_stare=[stare, barch_size, cell.starte_size][(0或1)， 50， 100（隐藏单元的个数）]
    #fill_stare[0]是cell stare
    #final_stare[1]是hidden_stare
    #outsput=[batch_size, max_time, cekk_outsize]跟time_major有关=Fasle
    #time_major有关=True outsput=[max_time, batch_size,  cekk_outsize]
    outputs, final_stare = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_stare[1], weight) + biases)
    return results

#计算rnn的返回结果
prediction = RNN(x, weight, biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#求准确率
correct_prediction = (tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("第%s周期" % epoch, '准确率%s' % acc)