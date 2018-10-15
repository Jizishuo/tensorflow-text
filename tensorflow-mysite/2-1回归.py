import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#使用np 生成200个随机点(-0.5, 0.5) 加维度200行一列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)#生成随机值shape（形状和x_data一样）
y_data = np.square(x_data) + noise #x_data 的平方加噪点

#定义一个占位符
x = tf.placeholder(tf.float32, [None, 1]) #[None, 1] 行不确定， 一列
y = tf.placeholder(tf.float32, [None, 1])

#神经网络10个神经元
Weights_L1 = tf.Variable(tf.random_normal([1, 10])) #随机1行10列的输出
biases_L1 = tf.Variable(tf.zeros([1, 10]))#[1,10]初始化为0 输出10个
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)#激活函数

#输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1])) #[1,1]初始化为0 输出1个
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)#预测的结果

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
    #查看预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    #绘图
    plt.figure()
    plt.scatter(x_data, y_data)#散点图
    plt.plot(x_data, prediction_value, "r-", lw =5)
    plt.show()
