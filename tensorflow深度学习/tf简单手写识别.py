from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow as tf


#cpu架构报错
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#print(mnist.train.images.shape)     # (55000, 28 * 28)
#print(mnist.train.labels.shape)   # (55000, 10)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('is_train', 1, '指定程序是训练还是预测')

def fullconnetcted():


    #建立数据占位符x [none, 784], y_true = [none, 10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    #建立一层全连接层的神经网络 w[784, 10]   b[10]
    with tf.variable_scope('net'):
        #随机初始化权重# 偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='W')
        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        #预测none个样本的输出结果[none, 784]*[784,10]+[10]=[none, 10]
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('soft_loss'):
        #求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    with tf.variable_scope('optimizer'):
        #梯度下降求损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #计算准确率
    with tf.variable_scope('acc'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        #equal_list none个样本【0，0，0，0，1。。。】求平均值
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #收集变量
    tf.summary.scalar('losses', loss)
    tf.summary.scalar('acc', accuracy)

    #高纬度数据收集
    tf.summary.histogram('weight', weight)
    tf.summary.histogram('bias', bias)


    #初始化op
    init = tf.global_variables_initializer()

    #定义一个合并变量的op
    merged = tf.summary.merge_all()

    #保存模型 创建saver
    saver = tf.train.Saver()

    #开启回话
    with tf.Session() as sess:
        sess.run(init)

        #取出特征值和目标值
        mnist_x, mnist_y = mnist.train.next_batch(50)

        #建立event文件，写入
        filewriter = tf.summary.FileWriter('net_bord/shouxie-text/', graph=sess.graph)

        if FLAGS.is_train == True:
            #训练
            pass
        else:
            #加载模型
            saver.restore(sess, 'net_bord/shouxie-text/fc_models')
            #预测
            for i in range(100):
                #每次测试一张图片
                x_text, y_text = mnist.test.next_batch(1)
                print('第%s张图片，真实是%s, 预测结果是%s' % (
                    i,
                    tf.argmax(y_text, 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_text, y_true:y_text})).eval(),
                ))

        #迭代训练，更新参数预测
        for i in range(2000):
            #运行训练
            sess.run(train_op, feed_dict={x: mnist_x,y_true:mnist_y})
            ac = sess.run(accuracy, feed_dict={x: mnist_x, y_true:mnist_y})
            print("%s次训练，准确率%s" % (i, ac))

            #写入每步训练的值
            summary = sess.run(merged, feed_dict={x: mnist_x, y_true:mnist_y})
            filewriter.add_summary(summary)

        #保存模型
        saver.save(sess, 'net_bord/shouxie-text/fc_models')

    return None

if __name__ == '__main__':
    fullconnetcted()