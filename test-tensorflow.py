import tensorflow as tf
import deepMnist
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#test data
images = mnist.test.images[0:4]

result = deepMnist.predict(images)
print result

#close
deepMnist.close()