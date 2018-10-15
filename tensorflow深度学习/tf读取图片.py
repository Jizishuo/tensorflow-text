import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning


#读取图片
def readpic(filelist):
    #构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    #构造阅读器读取图片内容（默认一张图片）
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    #print(value)

    #对图片数据解码
    image = tf.image.decode_jpeg(value)
    #print(image)

    #处理图片大小（统一）
    image_resize = tf.image.resize_images(image, [200, 200])
    #print(image_resize)
    #统一通道
    image_resize.set_shape([200, 200, 3])
    #print(image_resize)

    #进行批处理(所有数据形状必须统一)
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)
    print(image_batch)
    return image_batch

if __name__ == '__main__':
    file_name = os.listdir('F:\\picture_data\\baer')
    file_list = [os.path.join('F:\\picture_data\\baer\\', file) for file in file_name]
    #print(file_list)
    image_batch = readpic(file_list)

    with tf.Session() as sess:
        #定义一个线程协调器
        coord = tf.train.Coordinator()

        #开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        #打印
        #image_resize = readpic(file_list)
        print(sess.run([image_batch]))

        #回收子线程
        coord.request_stop()
        coord.join(threads)