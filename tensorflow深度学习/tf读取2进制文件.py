import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning

#定义cifarread数据的命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cifar_dir', '二进制文件路径', '文件的目录')
tf.app.flags.DEFINE_string('cifar_tfrecords', 'tfrecords文件路径', 'tfrecords文件的目录')


#读取2进制
class CifarRead(object):
    """
    读取二进制文件，写进tfecords， 读取tfrecord
    """
    def __init__(self, filelist):
        #文件列表
        self.file_list = filelist
        #定义读取图片的一些属性
        self.hight = 32
        self.width = 32
        self.channet = 3
        #二进制文件每张图片的字节
        self.label_bytes = 1
        self.image_bytes = self.hight*self.hight*self.channet
        self.bytes = self.bytes + self.image_bytes

    def read_anddecode(self):
        #构造文件队列
        fiel_queue = tf.train.string_input_producer(self.file_list)

        #构造二进制文件读取器
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read(self.bytes)

        #解码二进制内容
        label_image = tf.decode_raw(value, tf.uint8)

        #分隔图片和标签数据，特征值和目标值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)#从第一个切 标签 随便转化int类型
        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        #该表图片数形状【3072】--【32，32，3】
        image_reshape = tf.reshape(image, [self.hight, self.width, self.channet])

        #批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

    #将图片的特征值和目标值存进tfrecoreds
    def write_ro_tfrecords(self, image_batch, label_batch):
        #建立存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        #循环所有样本写入，每个图片都有ex协议
        for i in range(10):#这个批次只有10个
            #取出第i个图片的特征和目标 -- eval获取值再变字符串
            image = image_batch[i].eval().tostring()
            label = label_batch[i].eval()[0] #列表取第一个才是数字

            #构造一个样本的ex协议
            #tf.train.BytesList(value=[image])  只有3种类型bytes, string, int
            example = tf.train.Example(features=tf.train.Feature({
                'iamge': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            #写入单独的样本  序列化
            writer.write(example.SerializeToString())

        #关闭
        writer.close()

        return None

    def read_tfrecoeds(self):

        #构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        #构造文件阅读器，读取内容 value一个样本的序列化ex
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        #解析ex
        features = tf.parse_single_example(value, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
        print(features['image'], features['label'])

        #解码内容 如果读取是string需要解码，int,float不需要
        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.float32)
        print(image, label)

        #固定图片的形状 方便批处理
        image_reshape = tf.reshape(image, [self.hight, self.width, self.channet])

        #进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch

if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)
    file_list = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == 'bin']
    #print(file_list)
    cf = CifarRead(file_list)

    #读取二进制
    #image_batch, label_batch = cf.read_anddecode()
    #读取tfrecodes
    image_batch, label_batch = cf.read_tfrecoeds

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #定义一个线程协调器
        coord = tf.train.Coordinator()

        #开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        #存进tfrecords文件
        cf.write_ro_tfrecords(image_batch, label_batch)

        #打印
        print(sess.run([image_batch, label_batch]))

        #回收子线程
        coord.request_stop()
        coord.join(threads)