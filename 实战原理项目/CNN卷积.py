import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

mnist = input_data.read_data_sets('data', one_hot=True)
train = mnist.train.images
train_label = mnist.train.labels
testimg = mnist.test.images
test_label = mnist.test.labels

n_input = 784
n_output = 10
weights = {
    'wc1': tf.Variable(tf.random_normal([3,3,1,64], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3,3,64,128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([7*7*128,1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024,n_output], stddev=0.1)),
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1)),
}


def con_basic(_input, _w, _b, _keepratio):
    _input_r = tf.reshape(_input, shape=[-1,28,28,-1])
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1,1,1,1], padding='SAME')
    #_mean, _var = tf.nn.moments(_conv1, [0,1,2])
    #_conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0,1,0.00001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1,1,1,1], padding='SAME')
    #_mean, _var = tf.nn.moments(_conv1, [0,1,2])
    #_conv2 = tf.nn.batch_normalization(_conv1, _mean, _var, 0,1,0.00001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    _desel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_lsit()[0]])#7*7*128

    _fc1 = tf.nn.relu(tf.add(tf.matmul(_desel, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {'input_r': _input_r, 'conv1':_conv1, 'pool1':_pool1,'pool_dr1':_pool_dr1,
           'conv2':_conv2, 'pool2':_pool2, 'pool_dr2':_pool_dr2, 'desel':_desel,
           'fc1':_fc1, 'fc_dr':_fc_dr1, 'out':_out
           }

    return out

a = tf.Variable(tf.random_normal([3,3,1,64], stddev=0.1))
a = tf.Print(a, [a], 'a: ')


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

_pred = con_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
optm = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.case(_corr, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

training_epochs = 15
batch_size = 16
display_step =1

for epch in range(training_epochs):
    avg_cost = 0
    #total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = 10
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x:batch_xs, y:batch_ys, keepratio:0.7})
        avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys, keepratio:0.7})/total_batch

        if epch % display_step ==0:
            train_acc = sess.run(accr, feed_dict={x:batch_xs, y:batch_ys, keepratio:0.7})
            print('第%s步，误差:%s, 准确率%s' % (epch, avg_cost, train_acc))

sess.close()