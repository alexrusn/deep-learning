import numpy as np
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)
sess = tf.Session()
result = sess.run((total, hello))
print(result)

writer = tf.summary.FileWriter('logs')
writer.add_graph(tf.get_default_graph())
