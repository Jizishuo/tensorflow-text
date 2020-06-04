import tensorflow as tf
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
'''
#保存神经
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name="weights")
b = tf.Variable([[1,2,3]], dtype=tf.float32,name="biases")


init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "F:\\python项目\\tensorflow-项目\\神经网络训练结果保存\\test_net.ckpt")
    print("保存成功",save_path)
'''


#读取神经网络
#需要重新写神经网络框架
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name="biases")

#不需要init
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "F:\\python项目\\tensorflow-项目\\神经网络训练结果保存\\test_net.ckpt")
    print("weights:", sess.run(W))
    print("biases", sess.run(b))