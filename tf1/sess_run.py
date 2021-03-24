import tensorflow as tf
import numpy as np


''' constant mode '''
x = tf.constant([1.0])
w = tf.Variable(tf.random_uniform([1, 1]))
y = tf.add(x, w)
output = tf.nn.softmax(y)

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  w_val, y_val, output_val = sess.run([w, y, output])
  print(w_val, y_val, output_val)


''' placeholder mode '''
x = tf.placeholder(tf.float32, shape=[1])
w = tf.Variable(tf.random_uniform([1, 1]))
y = tf.add(x, w)
output = tf.nn.softmax(y)

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  w_val, y_val, output_val = sess.run([w, y, output], {x: [1.0]})
  print(w_val, y_val, output_val)

  # notice: y_val's type is 'numpy.ndarray'
  print(type(y_val))
