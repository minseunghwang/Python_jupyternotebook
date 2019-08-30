import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import io, color
from PIL import Image
from matplotlib import pyplot as plt
import sys
import warnings
# 싸이키런
from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data


warnings.filterwarnings(action="ignore")
tf.reset_default_graph()

# Data Loading
mnist= input_data.read_data_sets('./data/mnist', one_hot=True)
global_step = tf.Variable(0, trainable=False, name='global_step')

sess = tf.Session()
#ckpt = tf.train.get_checkpoint_state('C:\python_ML\model4')
#if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#    saver.restore(sess, ckpt.model_checkpoint_path)
#else : 
#    sess.run(tf.global_variables_initializer())

new_saver = tf.train.import_meta_graph('C:\python_ML\model4\cnn_model.ckpt-1000.meta')
new_saver.restore(sess, 'C:\python_ML\model4\cnn_model.ckpt-1000')

#img = Image.open('./data/number/1.png')
img = Image.open(sys.argv[1])

img_test =  img.resize((28,28))
img = np.array(img_test)
img_test = color.rgb2gray(img)

img_test = img_test.astype(np.float32)
test_img = img_test.reshape(-1, 784)
test_img = 1-test_img

X = sess.graph.get_tensor_by_name("Placeholder:0")
H = sess.graph.get_tensor_by_name("Relu_1:0")
keep_prob = sess.graph.get_tensor_by_name("Placeholder_2:0")

result2 = sess.run(H, feed_dict={X:test_img,keep_prob:0.3})
print(sess.run(tf.argmax(result2,1)))