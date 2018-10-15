import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning

'''
#模拟同步处理数据---再取数据训练
#读数据 取数据 加一 再加入队列
Q = tf.FIFOQueue(3, tf.float32)
#放数据  [[0.1, 0.2, 0.3],]变列表
enq_mang = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#取数据 加一
out_q = Q.dequeue()
#data = tf.add(out_q, 1)  #out_q已经是tensor可以直接+
data = out_q + 1
en_q = Q.enqueue_many(data)


with tf.Session() as sess:
    sess.run(enq_mang)

    #处理
    for i in range(2):
        sess.run(en_q)

    #训练数据--取数据
    #for i in range(Q.size().eval()):
        #print(sess.run(Q.dequeue()))

'''

'''
#模拟异步 子线程 存入数据 主线程 读取样本

#定义一个队列,1000
Q = tf.FIFOQueue(1000, tf.float32)

#定义子线程 循环加一
var = tf.Variable(0.0, tf.float32)

#实现一个自增
data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue_many([[data],])

#定义队列管理器Op,指定子线程几个 做什么 [en_q,]可以多个
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #开启线程管理器
    coord = tf.train.Coordinator()

    #开启子线程
    threads = qr.create_threads(sess, start=True)

    #主线程,不断读取数据训练
    for i in range(300):
        print(sess.run(Q.dequeue()))

    #回收子线程
    coord.request_stop()
    coord.join(threads)
    
'''

#读取csv文件

def mycsv():
    #F:\python项目\work\zhibiao-5\lte-chuli.csv
    file_name = os.listdir('F:\\python项目\\work\\zhibiao-5\\')
    #得到路径列表
    flielist = [os.path.join('F:\\python项目\\work\\zhibiao-5\\', file) for file in file_name]

    for flie in flielist:
        if not flie[-3:] == 'csv':
            flielist.remove(flie)

    #print(flielist)

    #构造文件队列
    file_queue = tf.train.string_input_producer(flielist[3:4])
    print(flielist[3:4])
    #构造csv阅读器读取队列(按一行)
    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    #对每行解码
    #指定每一列的类型 指定默认值str
    records = [['None'], ['None'],['None'],['None'],['None'],['None'],['None'],['None']]
    enid, enid1, enname, NODEBID, endid3, CGI, pool, VLAN = tf.decode_csv(value, record_defaults=records)

    #print(NODEBID)
    #批处理大小跟队列，数量没有影响
    pool_batch, vlan_batch = tf.train.batch([pool, VLAN], batch_size=100, num_threads=10, capacity=100)
    print(pool_batch)
    return pool_batch, vlan_batch




if __name__=='__main__':
    with tf.Session() as sess:
        #定义一个线程协调器
        coord = tf.train.Coordinator()

        #开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        #打印
        pool_batch, vlan_batch = mycsv()
        print(sess.run([pool_batch, vlan_batch]))

        #回收子线程
        coord.request_stop()
        coord.join(threads)