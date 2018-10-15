import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#np生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#模型 优化k b
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b


#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y)) #reduce_mean求平均值 square平方
#定义一个梯度下降法 优化器--
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss) #minimize最小化

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for setp in range(201):
        sess.run(train)
        if setp %20 ==0:
            print(setp, sess.run([k, b]))
