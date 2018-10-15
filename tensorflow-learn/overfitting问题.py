import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

#对比函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    #输入xs 生成预测值y_pre> >y_pre是[0,1,....]类似 10个数据 每个都是0-1之间的概率 不是具体0或1
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    #计算100个中对了几个错了几个 概率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result


#数据
digits = load_digits()
#0-9的图片data 类似mnist
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
#分成train 和 test 数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)

#神经层函数
def add_layer(input, input_size, out_size,layer_name,  activation_function = None):
    #随机生成变量矩阵
    Weights = tf.Variable(tf.random_normal([input_size, out_size]))
    #类似列表 1行 和out_size那么多列  让初始值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    #drop功能
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        #activation_function = None 说明是线性关系 不需要激活函数
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    #tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

#保持多少不被drop掉 去掉keep_prob:0.5  ---50% Wx_plus_b
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) #8*8
ys = tf.placeholder(tf.float32, [None, 10])

#神经层
l1 = add_layer(xs, 64, 50, "l1", activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, "l2", activation_function=tf.nn.softmax)


cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys *tf.log(prediction)), reduction_indices=[1]))
tf.summary.scalar("loss", cross_entropy)
#优化器  .minimize()减少误差

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("F:\\python项目\\tensorflow-项目\\111111", sess.graph)
test_writer = tf.summary.FileWriter("F:\\python项目\\tensorflow-项目\\111111\\test", sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    #去掉keep_prob:0.5  ---50% Wx_plus_b
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    if i %50 ==0:
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged,feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        #加到可视化
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        #print(compute_accuracy(X_test, y_test))