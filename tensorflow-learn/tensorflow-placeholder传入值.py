import tensorflow as tf
#出来架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

#float32数据形式 2个传入
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

#乘法运算
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    #feed_dict类似字典 跑到哪里 指定传入到哪里
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
