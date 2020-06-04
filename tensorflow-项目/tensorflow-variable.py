import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


#tensorflow变量 可以有初始值， 名字
state = tf.Variable(0, name="counter")
#print(state)
#<tf.Variable 'counter:0' shape=() dtype=int32_ref>
#print(state.name)

#tensorflow 常量 1
one = tf.constant(1)
#变量 加 常量
new_value = tf.add(state, one)
#更新变量
update = tf.assign(state, new_value)

#初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

