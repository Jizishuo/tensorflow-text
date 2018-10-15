import tensorflow as tf
import numpy as np
#出来架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error



#创造数据 数据类型float32
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#创造tensorflow 结构
#weights变量一维数据，初始值-1到1之间随机取
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#biasses变量初始值从0开始
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
#误差reduce_mean平均值
loss = tf.reduce_mean(tf.square(y - y_data))
#优化器 0.5学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
#减少误差
train = optimizer.minimize(loss)

#初始
init = tf.global_variables_initializer()

#指针  初始化激活
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

sess.close()