import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error


#矩阵1
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])

#matmul矩阵乘法--------- np.dot(m1, m2)
product = tf.matmul(matrix1, matrix2)

#初始化
'''sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()'''

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    