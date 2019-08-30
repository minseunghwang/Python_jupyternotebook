import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import warnings
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from IPython.display import Image
from matplotlib.image import imread
from skimage.color import rgb2gray
from skimage import data, io, filters, color
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data



warnings.filterwarnings(action="ignore")
tf.reset_default_graph()

#img = imread("./data/number/9.png")
img = Image.open(sys.argv[1])

img2 = np.array(img)
img_test = color.rgb2gray(img2)
img = 1-img_test

#plt.imshow(img, cmap = "gray")
sess = tf.InteractiveSession()
img = sess.run(tf.reshape(img,[1,784]))
print(3)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(4)

new_saver = tf.train.import_meta_graph('C:/python_ML/momodel/train_model.ckpt.meta')
new_saver.restore(sess, 'C:/python_ML/momodel/train_model.ckpt')
print(5)
tf.all_variables()
modelNum = 2
print(6)
result = np.zeros([1, 10])
for num in range(modelNum):
    print(7)
    modelName = "model"+str(num)
    X = sess.graph.get_tensor_by_name(modelName+"/Placeholder_1:0")
    logits = sess.graph.get_tensor_by_name(modelName+"/dense_1/BiasAdd:0")
    train = sess.graph.get_tensor_by_name(modelName+"/Placeholder:0")
    result += sess.run(logits, feed_dict={X:img, train:False})
    
print("MNIST predicted Number : ", np.argmax(result))