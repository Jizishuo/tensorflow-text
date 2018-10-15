import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning

#名字
#tf.app.flags.DEDINT_integer('max_step', 100,'模型训练的步数')
#tf.app.flags.DEDINT_string('model_dir', '', '模型文件加载的路径')

#FLAGS = tf.app.flags.FLAGS
#使用获取
#FLAGS.max_step
#FLAGS.model_dir

with tf.variable_scope('data'):
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
    #矩阵相乘必须2维
    y_true = tf.matmul(x, [[0.7]]) + 0.8

with tf.variable_scope('model'):
    #建立线性回归模型 1个权重一个偏置
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.1, stddev=1.0, name='w'))
    bias = tf.Variable(0.1, name='b')
    y_predict = tf.matmul(x, weight) + bias

with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

with tf.variable_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#收集--合并-显示
tf.summary.scalar('losser', loss)
tf.summary.histogram('wights', weight)

#合并
merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

#保存模型的实例 默认保存5个
#saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #显示
    filewriter = tf.summary.FileWriter('F:\\python项目\\深度学习\\tensorflow深度学习\\', graph=sess.graph)

    #加载模型覆盖随机的权重编织（上边网络要一样）
    #if os.path.exists('F:\\python项目\\深度学习\\tensorflow深度学习\\test'):
        #saver.restore(sess, 'F:\\python项目\\深度学习\\tensorflow深度学习\\test')

    for i in range(300):
        sess.run(train)
        #运行合并图
        summary = sess.run(merged)
        filewriter.add_summary(summary, i)
        print("第%s,权重%s, 偏制%s" % (i, weight.eval(), bias.eval()))
        #if i % 50:
            #saver.save(sess, 'F:\\python项目\\深度学习\\tensorflow深度学习\\test')

    #saver.save(sess, 'F:\\python项目\\深度学习\\tensorflow深度学习\\test')