import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add]) #fetch 同时运行几个op
    print(result)


#feed
input4 = tf.placeholder(tf.float32)#占位符
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4, input5)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input4:[7.0], input5:[2.0]})) #[14.]

