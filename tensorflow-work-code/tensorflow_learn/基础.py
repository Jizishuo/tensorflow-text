import numpy as np
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

#
# print(tf.add(1, 2))
# print(tf.add([1, 2], [3, 4]))
# print(tf.square(5))
# print(tf.reduce_sum([1, 2, 3]))
# print(tf.encode_base64("hello world"))
#
# # Operator overloading is also supported
# print(tf.square(2) + tf.square(3))
#
# x = tf.matmul([[1]], [[2, 3]])
# print(x.shape)
# print(x.dtype)


# ndarray = np.ones([3, 3])
#
# print("TensorFlow operations convert numpy arrays to Tensors automatically")
# tensor = tf.multiply(ndarray, 42)
# print(tensor)
#
# print("And NumPy operations convert Tensors to numpy arrays automatically")
# print(np.add(tensor, 1))
#
# print("The .numpy() method explicitly converts a Tensor to a numpy array")
# print(tensor.numpy())

# x = tf.random_uniform([3, 3])
#
# print("Is there a GPU available: "),
# print(tf.test.is_gpu_available())
#
# print("Is the Tensor on GPU #0:  "),
# print(x.device.endswith('GPU:0'))

# import time
#
# def time_matmul(x):
#   start = time.time()
#   for loop in range(10):
#     tf.matmul(x, x)
#
#   result = time.time()-start
#
#   print("10 loops: {:0.2f}ms".format(1000*result))


# # Force execution on CPU
# print("On CPU:")
# with tf.device("CPU:0"):
#   x = tf.random_uniform([1000, 1000])
#   assert x.device.endswith("CPU:0")
#   time_matmul(x)
#
# # Force execution on GPU #0 if available
# print("On GPU:")
# if tf.test.is_gpu_available():
#   with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
#     x = tf.random_uniform([1000, 1000])
#     assert x.device.endswith("GPU:0")
#     time_matmul(x)


# v = tf.Variable(1.0)
# assert v.numpy() == 1.0
#
# # Re-assign the value
# v.assign(3.0)
# assert v.numpy() == 3.0
#
# # Use `v` in a TensorFlow operation like tf.square() and reassign
# v.assign(tf.square(v))
# assert v.numpy() == 9.0

import matplotlib.pyplot as plt


class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

model = Model()

# assert model(3.0).numpy() == 15.0

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())

#
#
#
# plt.scatter(inputs, outputs, c='b')
# plt.scatter(inputs, model(inputs), c='r')
# plt.show()
#
# print('Current loss: '),
# print(loss(model(inputs), outputs).numpy())
#
#
#
# model = Model()
#
# # Collect the history of W-values and b-values to plot later
# Ws, bs = [], []
# epochs = range(10)
# for epoch in epochs:
#   Ws.append(model.W.numpy())
#   bs.append(model.b.numpy())
#   current_loss = loss(model(inputs), outputs)
#
#   train(model, inputs, outputs, learning_rate=0.1)
#   print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
#         (epoch, Ws[-1], bs[-1], current_loss))
#
# # Let's plot it all
# plt.plot(epochs, Ws, 'r',
#          epochs, bs, 'b')
# plt.plot([TRUE_W] * len(epochs), 'r--',
#          [TRUE_b] * len(epochs), 'b--')
# plt.legend(['W', 'b', 'true W', 'true_b'])
# plt.show()
