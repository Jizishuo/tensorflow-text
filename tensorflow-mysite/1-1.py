import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#常量constent--op
m1 = tf.constant([[3, 3]])

m2 = tf.constant([[2],
                 [3]])
#矩阵乘法
prosuct = tf.matmul(m1, m2)
print(prosuct)  #Tensor("Mul:0", shape=(1, 1), dtype=int32)
#启用默认图 定义一个绘画
sess = tf.Session()
result = sess.run(prosuct)
print(result) #[[15]]
sess.close()

with tf.Session() as sess:
    result = sess.run(prosuct)
    print(result)