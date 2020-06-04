import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np

#验证集数量
_NUM_TEST = 1

#随机种子
_RANDOM_SEED = 0

#数据集路径
DATASET_DIR = "F:/python项目/work-gongdan/verification_code/ttttttttttt/img_data/"

#tfrecord存放路径
TFRECORD_DIR = "F:/python项目/work-gongdan/verification_code/ttttttttttt/tfrecord_data/"

#判断tfrecord是否存在
def _dataset_exists(dataset_dir):
    for split_name in ["train", "test"]:
        output_filename = os.path.join(dataset_dir, split_name + ".tfrecords")
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有图片验证码
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def in64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    return tf.train.Example(features=tf.train.Features(feature={
        "image": bytes_feature(image_data),
        "label0": in64_feature(label0),
        "label1": in64_feature(label1),
        "label2": in64_feature(label2),
        "label3": in64_feature(label3),
    }))

#数据转化为TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ["train", "test"]

    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR, split_name + ".tfrecords")
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                try:
                    sys.stdout.write("r\>> Creating image %d/%d" % (i + 1, len(filenames)))
                    sys.stdout.flush()
                    #读取图片
                    image_data = Image.open(filename)
                    #根据模型的结构resize160 60 --224,224
                    image_data = image_data.resize((224, 224))
                    #灰度化
                    image_data = np.array(image_data.convert("L"))
                    #将图片转换为bytes
                    image_data = image_data.tobytes()

                    #获取labelF:\\python项目\\tensorflow--验证码识别\\data-image\\[-1][0:4] 就是文件名就是数字
                    labels = filename.split("/")[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))

                    #生成protocol数据类型
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())

                except IOError as e:
                    print("Could not read:", filename)
                    print("Error:", e)
                    print("Skip it\n")
    sys.stdout.write("\n")
    sys.stdout.flush()

#判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print("tfcecord文件已经存在")
else:
    #获取图片
    photo_filenames = _get_filenames_and_classes(DATASET_DIR)
    #把数据集切分为训练集和测试集， 并打乱
    # random.seed(_RANDOM_SEED)
    # random.shuffle(photo_filenames)
    #training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames#[:_NUM_TEST]

    #数据类型转换
    #_convert_dataset("train", training_filenames, DATASET_DIR)
    _convert_dataset("test", testing_filenames, DATASET_DIR)
    print("生成tfcecord文件")