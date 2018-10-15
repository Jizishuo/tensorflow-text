import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
#运行次数
max_steps = 1001
#参考图片数量
images_num = 3000
DIR = "F:/python项目/tensorflow--验证码识别/神经网络结构图/"

sess = tf.Session()

#载入图片#stack矩阵转换
embedding = tf.Variable(tf.stack(mnist.test.images[:images_num]), trainable=False, name="embedding")

#参数
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)#平均值
        tf.summary.scalar("mean", mean)#平均值 起名字
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev) #标准差
        tf.summary.scalar("max", tf.reduce_max(var))#最大值
        tf.summary.scalar("mix", tf.reduce_min(var))#最小值
        tf.summary.histogram("histogram", var)#直方图


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x-input") #每一批100 none=100 每个28*28
    y = tf.placeholder(tf.float32, [None, 10], name="y-input")#每一批100 none=100 每个10个

#显示图片
with tf.name_scope("input_reshape"):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])#[-1, 28, 28 ,1] -1表示不确定-任意传入几个 28*28 1维度1黑白
    tf.summary.image("input", image_shaped_input, 10)#放10张


with tf.name_scope("layer"):
    with tf.name_scope("wights"):
        W = tf.Variable(tf.zeros([784, 10]), name="W")
        variable_summaries(W)
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([10]), name="b")
        variable_summaries(b)
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope("prediction"):
        prediction = tf.nn.softmax(wx_plus_b)#softmax 计算概率


with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train_step"):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess.run(tf.global_variables_initializer())

with tf.name_scope("accuracy-all"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

#生成一个metadata文件
if tf.gfile.Exists(DIR + "metadata.tsv"):
    tf.gfile.DeleteRecursively(DIR + "metadata.tsv")
with open(DIR + "metadata.tsv", "w") as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))#把测试集标签都拿到 argmax一行中的最大
    for i in range(images_num):
        f.write(str(labels[i]) + "\n")


#合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR, sess.graph)
#用来保存网络模型
saver = tf.train.Saver()
# 定义配置文件
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name#载入图片name
embed.metadata_path = DIR + "metadata.tsv"
embed.sprite.image_path = DIR + "mnist_10k_sprite.png"
# 切分图片
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #配置
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    #run 传进去
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)
    #记录
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("第%s周期" % i, '准确率%s' % acc)

saver.save(sess, DIR + 'a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()