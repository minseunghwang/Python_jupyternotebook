### MNIST(CNN) For Saver ( No Ensemble )
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

## 1. Data Loading (tensorflow examples MNIST DATA)
mnist = input_data.read_data_sets("c:/python_ML/data/mnist", one_hot=True)

## 2. Model 정의(Tensorflow graph 생성)
tf.reset_default_graph() # tensorflow graph 초기화

## 2.1 placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32, name="x")
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
drop_rate = tf.placeholder(dtype=tf.float32)

## 2.2 Convolution
X_img = tf.reshape(X, shape=[-1,28,28,1])
L1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3], 
                      padding="SAME", strides=1, activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2,2], 
                             padding="SAME", strides=2)
L1 = tf.layers.dropout(inputs=L1, rate=0.3)
            
L2 = tf.layers.conv2d(inputs=L1, filters=64, kernel_size=[3,3], 
                      padding="SAME", strides=1, activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(inputs=L2, pool_size=[2,2], 
                             padding="SAME", strides=2)
L2 = tf.layers.dropout(inputs=L2, rate=0.3)
            
L3 = tf.layers.conv2d(inputs=L2, filters=128, kernel_size=[3,3], 
                      padding="SAME", strides=1, activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(inputs=L3, pool_size=[2,2], 
                             padding="SAME", strides=2)
L3 = tf.layers.dropout(inputs=L3, rate=0.3)
                                    
L3 = tf.reshape(L3, shape=[-1,4*4*128])

## 2.3 Neural Network

dense1 = tf.layers.dense(inputs=L3, units=128, activation=tf.nn.relu)
dense1 = tf.layers.dropout(inputs=dense1, rate=drop_rate)

dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
dense2 = tf.layers.dropout(inputs=dense2, rate=drop_rate)

dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.relu)
dense3 = tf.layers.dropout(inputs=dense3, rate=drop_rate)

dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu)
dense4 = tf.layers.dropout(inputs=dense4, rate=drop_rate)

dense5 = tf.layers.dense(inputs=dense4, units=1024, activation=tf.nn.relu)
dense5 = tf.layers.dropout(inputs=dense5, rate=drop_rate)
            
H = tf.layers.dense(inputs=dense5, units=10)
H_identity = tf.identity(H, name="h")

## cost function
cost = tf.losses.softmax_cross_entropy(Y, H)

## train
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

# accuracy
predict = tf.argmax(H,1)
correct = tf.equal(predict, tf.argmax(Y,1))
correct_num = tf.reduce_sum(tf.cast(correct, dtype=tf.float32))


## 4. 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 5. 학습
num_of_epoch = 30
batch_size = 100

for step in range(num_of_epoch):
    num_of_iter = int(mnist.test.num_examples / batch_size)
    
    for i in range(num_of_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train,cost], feed_dict={X:batch_x, Y:batch_y,
                                                       drop_rate:0.5})
    if step % 3 == 0:
        print("cost : {}".format(cost_val))    


num_of_iter = int(mnist.test.num_examples / batch_size)    
total_sum = 0
for i in range(num_of_iter):
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    total_sum += sess.run(correct_num, feed_dict={X:batch_x, Y:batch_y, drop_rate:1})
    
print("정확도 : {}".format(total_sum/mnist.test.num_examples))

builder = tf.saved_model.builder.SavedModelBuilder("C:/python_ML/TFJavaAPI/SaveModel")
builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
builder.save()