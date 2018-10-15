import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning

# 自定义命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('captchares_dir', 'data/image_shibie', '验证码数据路径路径')
tf.app.flags.DEFINE_integer('batch_size', 100, '每批次训练个数')
tf.app.flags.DEFINE_integer('label_num', 4, '识别每个图片目标值元素数量')
tf.app.flags.DEFINE_integer('latter_num', 26, '识别字母元素个数')


def read_and_decode():
    """
    #读取数据api
    :return: image_batch, label_batch
    """
    # 构建文件队列
    file_queue = tf.train.string_input_producer([FLAGS.captchares_dir])  # 默认有打乱顺序

    # 构建阅读器，读取内容，默认一个样本
    reader = tf.TFRecordReader()

    # 读取内容
    key, value = reader.read(file_queue)

    # tfrecords格式ex，需要解析 features--str类型
    features = tf.parse_single_example(value, features={
        'image': tf.FixedLenFeature([], tf.string),  # 存进去str 解析也是str
        'label': tf.FixedLenFeature([], tf.string),
    })

    # 解码内容，字符串内容
    # 先解析图片的特征值 和标签
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    print(image, label)  # 形状木有固定

    # 改变形状
    image_reshape = tf.reshape(image, [20, 80, 3])
    label_reshape = tf.reshape(label, [4])

    # 进行批处理 每批次100 每次训练的样本数
    image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], \
                                              batch_size=FLAGS.batch_size, num_threads=1, capacity=FLAGS.batch_size)

    return image_batch, label_batch


# 定义一个初始化权重的函数
def weigth_va(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_va(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def fc_model(image):
    """
    进行预测结果
    :param image: 图片特征值[100, 20, 80,3]
    :return: y-predict--[100, 4*26]
    """
    with tf.variable_scope('model'):
        # 将图片数据转为二维[100, 20, 80,3]-[100,20*80*3] -1代表100
        image_reshape = tf.reshape(image, [-1, 20 * 80 * 3])

        # 随机初始化权重，偏置[100, 20, 80,3]*[20*80*3, 4*26]+[104]=[100, 4*26]
        weight = weigth_va([20 * 80 * 3, 4 * 26])
        bais = bias_va([4 * 26])

        # 进行全连接层预测 y_predict [100, 4*26]
        # image_reshape-ui8---filoat32
        image_reshape = tf.cast(image_reshape, tf.float32)
        y_predict = tf.matmul(image_reshape, weight) + bais

        return y_predict


def predict_to_onehot(label):
    """
    将读取的文件中的目标值转为onehot编码
    :param label_batch:[100,4]
    [[11,22,23,23], [22,21,3,12],[22,21,3,12],[22,21,3,12]].....
    :return:[0,1,0,0,0......] one_hot
    """
    # 进行onehot编码，提供给交叉熵，准确率运算
    # on_value对应数字的位置填上1 axis=2三维最里边的[[11,22,23,23], [..... 对应到11，22，23....
    # [100, 4 ,26]
    label_onehot = tf.one_hot(label, depth=FLAGS.latter_num, on_value=1.0, axis=2)

    return label_onehot


def captchares():
    """
    #验证码识别程序
    :return: none
    """
    # 读取验证码数据文件
    image_batch, label_batch = read_and_decode()

    # 通过输入图片特征数量， 建立模型，得出预测结果
    # 一层全连接神经网络进行预测[100, 20, 80,3]*[20*80*3, 4*26]+[104]=[100, 4*26]
    y_predict = fc_model(image_batch)
    print(y_predict)  # [100, 4*26]

    # label_batch[100,4]--[100, 4, 26]
    # 先把目标值转为one-hot编码 y_true [100, 4, 26]-[100, 4*26]
    y_true = predict_to_onehot(label_batch)
    # y_true = tf.reshape(y_true, [-1, 4*26]) #FLAGS.label_num*26

    # 交叉熵损失计算
    with tf.variable_scope('soft_loss'):
        # 求平均交叉熵损失 y_true [100, 4, 26]-[100, 4*26]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(y_true, [-1, 4 * 26]),  # FLAGS.label_num*26
            logits=y_predict))

    with tf.variable_scope('optimizer'):
        # 梯度下降求损失
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 计算准确率 三维比较 [100, 4, 26]
    with tf.variable_scope('acc'):
        # 比较每个预测值和目标值位置是否一样 4个位置
        # y_true [100, 4, 26]  2表示第二个位置(onehot)求最大值
        # y_predict [100, 4*26]--[100, 4, 26]
        equal_list = tf.equal(tf.argmax(y_true, 2),
                              tf.argmax(tf.reshape(y_predict, [-1, 4, 26]), 2))
        # equal_list none个样本【0，0，0，0，1。。。】求平均值==准确率
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # 定义线程协调器和开启
        coord = tf.train.Coordinator()

        # 开启线程运行读取文件
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 训练识别程序 训练5k步
        for i in range(5000):
            sess.run(train_op)
            acc = sess.run(accuracy)
            print('第%s批次,准确率%s' % (i, acc))

        # 回收线程
        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == '__main__':
    captchares()
