import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning

#创建一张图，上下文环境不同
g = tf.Graph()

print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)

#实现一个加法
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)

with tf.Session() as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)
