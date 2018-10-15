import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning

# 自定义命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_dir', 'data/image_shibie', '文件储存路径路径')
tf.app.flags.DEFINE_string('captcha_dir', 1, '图训练图片路径')
tf.app.flags.DEFINE_string('latter', 'ABCDEFGHIJKLMNOPGRSTUBWSYZ', '验证码数据的种类')


##获取验证码中的图片
def get_captcha_image():
    """
    获取验证码图片
    file_image:路径+文件名
    :return:image
    """
    filename = []
    for i in range(6000):
        string = str(i) + '.jpg'
        filename.append(string)

    # 构造路径+文件
    file_list = [os.path.join(FLAGS.captcha_dir, file) for file in filename]

    # 构造文件队列
    file_queue = tf.train.string_input_producer(file_list, shuffle=False)  ##shuffle=False不能乱序

    # 构造阅读器
    reader = tf.WholeFileReader()

    # 读取图片数据内容
    key, value = reader.read(file_queue)

    # 解码图片数据
    image = tf.image.decode_jpeg(value)  # [?,?,?]
    image.set_shape([20, 80, 3])  # 设置图片形状

    # 批处理数据 [6000, 20 ,80, 3]
    image_batch = tf.train.batch(image, batch_size=6000, num_threads=1, capacity=6000)

    return image_batch


# 获取验证码文件中的标签数据
def get_captcha_label():
    """
    读取验证码图片标签
    :return: label
    """
    file_queue = tf.train.string_input_producer(['...xx.csv'], shuffle=False)  # shuffle=False不能乱序

    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    records = [[1], ['None']]  # 【1】int， 【none】字符串

    number, label = tf.decode_csv(value, record_defaults=records)

    # [['xxxx], ['xxxx'],['xxxx']........]
    label_batch = tf.train.batch([label], batch_size=6000, num_threads=1, capacity=6000)

    return label_batch


# 处理字符串标签到数字张量
# [b'mnpl' b'mnbv' b'assdf'........]
def deal_with_label(label_str):
    # 构建字符索引{1:'A', 1:'B', 2:'C'.....}
    num_letter = dict(enumerate(list(FLAGS.latter)))

    # 键值对反装{'A':0, 'B':1,.....}
    letter_num = dict(zip(num_letter.values(), num_letter.keys()))
    # print(letter_num)

    # 构建标签列表
    array = []

    # 给标签数据进行处理
    # [b'mnpl' b'mnbv' b'assdf'........]
    for string in label_str:
        letter_list = []

        # 修改编码，b'FDGR'到字符串，并且循环到每张验证码的字符串对应的数字标记
        for letter in string.decode('utf-8'):
            letter_list.append(letter_num[letter])

        array.append(letter_list)
    # [[11,22,23,23], [22,21,3,12],[22,21,3,12],[22,21,3,12]]
    print(array)

    # 将array装换成tensor类型
    label = tf.constant(array)

    return label


# 将图片数据和内容写入到tfrecords文件当中
def write_to_tfrecords(image_batch, lable_batch):
    """
    将图片内容和标签写入到tfrecords文件当中
    :param image_batch: 特征值
    :param lable_batch: 标签值
    :return: none
    """

    # 装换类型
    lable_batch = tf.cast(lable_batch, tf.uint8)
    print(lable_batch)

    # 建立TFRecords存储器
    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecords_dir)  # 路径

    # 循环将每一个图片上的数据构造ex协议块，序列化后写入
    for i in range(6000):
        # 取出第i个图片数据，装换相应类型，图片的特征值要装换成字符串形式
        image_string = image_batch[i].eval().tostring()

        # 标签值，装换成整形
        label_string = lable_batch[i].eval().tostring()

        # 构造协议块
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
        }))

        writer.write(example.SerializeToString())

    # 关闭文件
    writer.close()

    return None


if __name__ == '__main__':
    # 获取验证码中的图片
    image_batch = get_captcha_image()

    # 获取验证码文件中的标签数据
    label = get_captcha_label()

    print((image_batch, label))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # [b'mnpl' b'mnbv' b'assdf'........]
        label_str = sess.run(label)
        print(label_str)

        # 处理字符串标签到数字张量
        lable_batch = deal_with_label(label_str)
        print(lable_batch)

        # 将图片数据和内容写入到tfrecords文件当中
        write_to_tfrecords(image_batch, lable_batch)

        coord.request_stop()
        coord.join(threads)

# tfrecords
# image--[1000, 20, 80, 3]
# label --[100, 4] ---[[11,22,23,23], [22,21,3,12],[22,21,3,12],[22,21,3,12]]

