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

ntrain, ntest, dim, nclasses = train.shape[0], testimg.shape[0],\
                            train.shape[1], train_label.shape[1]

diminput = 28
dimhidden = 128
dimoutput = nclasses#10
nsteps = 28 #28个1*28

weights = {
    'hidden':tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out':tf.Variable(tf.random_normal([dimhidden, dimoutput])),
}
biases = {
    'hidden':tf.Variable(tf.random_normal([dimhidden])),
    'out':tf.Variable(tf.random_normal([dimoutput])),
}


def RNN(_x, _w, _b, _nsteps, _name):
    _x = tf.transpose(_x, [1,0,2])#[batch, nstep diminput]-[n,b,d]-[n*b, d]
    _x = tf.reshape(_x, [-1, diminput])
    _h = tf.matmul(_x, _w['hidden'] + _b['hidden'])
    _hsplit = tf.split(0, _nsteps, _h)

    with tf.variable_scope(_name) as scope:
        scope.reuse_cariables()#命名不报错(共享多次变量名)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)#第一次不忘记
        _lstm_o, _lstm_s = tf.nn.rnn(lstm_cell, _hsplit, dtype=tf.float32)
    _o = tf.matmul(_lstm_o[-1], _w['out']) +_b['out']

    return {
        'x':_x, 'h':_h, 'hsplit':_hsplit, 'lstm_o':_lstm_o, 'lstm_s':_lstm_s, 'o':_o
    }

lr = 0.000001
x = tf.placeholder('float', [None, nsteps, diminput])
y = tf.placeholder('float', [None, dimoutput])
myrnn = RNN(x, weights, biases, nsteps, 'basic1') #basic1加1才不会报错
perd = myrnn['o']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(perd, y))
optm = tf.train.GradientDescentOptimizer(lr).minimize(cost)
_corr = tf.equal(tf.argmax(perd,1), tf.argmax(y, 1), tf.float32)
accr = tf.reduce_mean(tf.case(_corr, tf.float32))

init = tf.global_variables_initializer()

train_epochs = 5
batch_size = 16
display_step = 1
sess = tf.Session()

sess.run(init)

for epch in range(train_epochs):
    avg_cost = 0
    #total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = 10
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x:batch_xs, y:batch_ys})
        avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/total_batch

        if epch % display_step ==0:
            train_acc = sess.run(accr, feed_dict={x:batch_xs, y:batch_ys})
            print('第%s步，误差:%s, 准确率%s' % (epch, avg_cost, train_acc))

sess.close()
