import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import io, color
from PIL import Image
from matplotlib import pyplot as plt
import sys

# ����Ű��
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# Data Loading
mnist= input_data.read_data_sets('./data/mnist', one_hot=True)

# placeholder 
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
#rate=tf.placeholder(dtype=tf.float32)
keep_prob = tf.placeholder(dtype=tf.float32)

X_img = tf.reshape(X, [-1, 28, 28, 1])
## 2.2.1 Convolution Layer1
# kernel_size �� ������ ũ��
L1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], padding='SAME', strides=1, activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2,2], padding='SAME', strides=2)

## 2.2.1 Convolution Layer2
L2 = tf.layers.conv2d(inputs=L1, filters=32, kernel_size=[3,3], padding='SAME', strides=1, activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(inputs=L2, pool_size=[2,2], padding='SAME', strides=2)

## 2.3 Neural Network
L2 = tf.reshape(L2,[-1,7*7*32])

# shape [, logist�� ���� (logist layer�� �󸶳� ���� ���� �ϴ��� )]
W1 = tf.get_variable('weight1', shape=[7*7*32,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
_layer1 = tf.nn.relu(tf.matmul(L2, W1)+b1)
layer1 = tf.nn.dropout(_layer1, keep_prob=keep_prob) 


W2 = tf.get_variable('weight2', shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]), name='bias2')

#Hypothesis
logits = tf.matmul(layer1, W2) + b2

H = tf.nn.relu(logits)

# cost Function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))


# train
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)


# Session & �ʱ�ȭ
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
global_step = tf.Variable(0, trainable=False, name='global_step')

ckpt = tf.train.get_checkpoint_state('./model4')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else : 
    sess.run(tf.global_variables_initializer())

#img = Image.open('./data/number/0.png')
img = Image.open(sys.argv[1])

img_test =  img.resize((28,28))
img = np.array(img_test)
img_test = color.rgb2gray(img)

img_test = img_test.astype(np.float32)
test_img = img_test.reshape(-1, 784)
test_img = 1-test_img

predict = tf.argmax(H,1)
result = sess.run(predict, feed_dict={X:test_img,keep_prob:0.3 })
result2 = sess.run(H, feed_dict={X:test_img,keep_prob:0.3})
print(result)
#print(result2)