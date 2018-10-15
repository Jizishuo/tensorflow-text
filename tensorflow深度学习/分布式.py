import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEDINT_string('job_name', 'worker', '启动服务的类型 ps or worker')
tf.app.flags.DEDINT_integer('task_index', 0 , '指定ps-or-worker当中的那一台服务器task:0')

def main(argv):
    #定义全局计算ap， 给钩子列表当中的训练步数使用
    global_step = tf.contrib.framework.get_or_create_global_step()

    #指定集群描述对象, ps, worker
    cluster = tf.train.ClusterSpec({'ps': ['111.111.11:2222'],
                                    'worker': ['192.168.65.44:2222']})

    #创建不同的服务
    sever = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    #根据不同的服务 做不同的事情
    #ps更新保存参数， worker指定设备运行模型计算
    if FLAGS.job_name == "ps":
        #参数服务器什么都不用干 只需要等待worker传递参数
        sever.join()
    #worker 服务器
    else:
        #可以指定设备去运行
        worker_device = '/job:worker/task:0/cpu:0/' #本机属于第几台设备

        with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster
        )):
            #做一个简单算法
            a = tf.Variable([[1, 2, 3, 4]])
            b = tf.Variable([[1],[2],[3],[4]])

            mat = tf.matmul(a, b)


        #初始化会话（指定一个老大）本机
        with tf.train.MonitoredTrainingSession(
            master="grpc://192.168.65.44:2222", #指定主worker
            is_chief=(FLAGS.task_index==0), #0就是主worker 判断是否主worker
            config=tf.ConfigProto(log_device_placement=True),#打印设备信息
            hooks=[tf.train.StopAtStepHook(last_step=50)],#指定运行几次mat = tf.matmul(a, b)
        ) as mon_sess:
            #判断sess是否出错
            while not mon_sess.should_stop:
                mon_sess.run(mat)

if __name__ =="__main__":
    tf.app.run()