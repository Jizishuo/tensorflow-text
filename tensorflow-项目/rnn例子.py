import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])



weights = {
    #28,128
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    #128, 10
    'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, biases):
    #X(128 batch, 28 steps, 28 inputs)==>(128*28, 28 inputs)
    X = tf.reshape(X,[-1, n_inputs])
    #(128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # (128 batch , 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    #cell
    #forget_bias=1.0初始忘记1 不忘记
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    #初始生成2条线记忆（c_state, m_state）
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)


    #ouput
    #results = tf.matmul(states[1], weights['out']) + biases['out']
    outputs = tf.stack(tf.transpose(outputs,[1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size <training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        bacth_xs = batch_xs.reshape([batch_size, n_steps,n_inputs])
        sess.run([train_op], feed_dict={
            x:bacth_xs,
            y:batch_ys,
        })
        if step %20 ==0:
            print(sess.run(accuracy,feed_dict={
                x: bacth_xs,
                y: batch_ys,
            }))
            step +=1
