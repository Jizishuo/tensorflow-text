import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        #加载分类字符串n******对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #一行一行读取数据
        for line in proto_as_ascii_lines:
            #去掉换行符
            line = line.strip('\n')
            #按照‘\t’分隔
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号字符串n**********与分类名称的映射关系
            uid_to_human[uid] = human_string
            #{n00004475:organism, being}

        #加载分类字符串n********对应分类编号1-100的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-100
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n*******
                target_class_string = line.split(': ')[1]
                #保存分类编号1-100与字符串n******映射关系
                node_id_to_uid[target_class] = target_class_string[1: -2]
                #{354:n03838899}

        #建立分类编号1-100对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-100到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name
        #{001:organism, being}

    #传入分类编号1-100返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ""
        return self.node_lookup[node_id]

# 创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    #graph_def.ParseFromSring(f.read())
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录images文件夹
    #得到root文件，dirs子目录（空的）,files文件(5张图片)
    for root, dirs, files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0' :image_data})# 图片格式jpg 传进模型得到2维数据
            #转一维(1000个结果的概率)
            predictions = np.squeeze(prediction)

            #打印图片格式及名称
            image_data = os.path.join(root, file)
            print(image_data)
            #显示图片
            img = Image.open(image_data)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            #排序--概率从小到大取最后5个最大的概率,然后[::-1]做一次倒叙5个从大到小
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                #获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s 的概率=%.5f' % (human_string, score))
                #print(human_string +'的概率=' + score)
            print("-----------------")


