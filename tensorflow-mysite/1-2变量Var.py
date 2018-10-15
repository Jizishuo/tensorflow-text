import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable([1, 2])
a = tf.Variable([3,3])

#加法op
add = tf.add(x, a)
#减法op
sub = tf.subtract(x, a)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(add))
    print(sess.run(sub))


state = tf.Variable(0, name="counter") #创建一个变量初始值为0
#创建一个op 使state加一
new_state = tf.add(state, 1)
#赋值state==new_state
update = tf.assign(state, new_state)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))