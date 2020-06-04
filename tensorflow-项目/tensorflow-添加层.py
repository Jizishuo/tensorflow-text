import tensorflow as tf
import numpy as np

#出来架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


#神经层函数
def add_layer(input, input_size, out_size, activation_function = None):
    #随机生成变量矩阵
    Weights = tf.Variable(tf.random_normal([input_size, out_size]))
    #类似列表 1行 和out_size那么多列  让初始值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        #activation_function = None 说明是线性关系 不需要激活函数
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#加了维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#噪点
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#传入值 none 表示传入多少都ok 就一个
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


#第一层 输入1  一个值 输出10
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#第二次 接受l1 输出一个结果
predition = add_layer(l1, 10, 1, activation_function=None)

# 误差 》square 平方》 reduce_sum 求和》reduction_indices=[1] 表示向列方向压缩 1 压缩方向》reduce_mean求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))
#优化器  .minimize()减少误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1, 1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0 :
        #print(i)
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))

sess.close()