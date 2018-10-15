import tensorflow as tf
import os
import tarfile
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

path = "F:/python项目/tensorflow--验证码识别/inception-v3/"
#创建目录
inception_pretrain_model_dir = path +"inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

#下载文件
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

if not os.path.exists(filepath):
    print('download:',filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print('完成：', filename)
#解压
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

log_dir = path + 'logs/inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classfy_image_graph_def.pb 为google训练好的模型
inception_graph_def_fiel = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_fiel, 'rb') as f:
        graph_def = tf.GraphDef()
        #         graph_def.ParseFromSring(f.read())
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()'''

inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

if not os.path.exists(filepath):
    print('download:', filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print('finish: ', filename)

# 解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

log_dir = 'logs/inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classfy_image_graph_def.pb 为google训练好的模型
inception_graph_def_fiel = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_fiel, 'rb') as f:
        graph_def = tf.GraphDef()
        #         graph_def.ParseFromSring(f.read())
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()