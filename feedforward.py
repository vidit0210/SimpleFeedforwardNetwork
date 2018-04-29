import numpy as np 
import math

class FeedForwardNetwork():
	def __init__(self):
		self.weights = np.array([1.0,2.0])
		self.bias=0.0

	def Forward(self,inputs):
		a_cell_sum = np.sum(self.weights*inputs)+self.bias
		sigmoid_activation= 1.0 / (1.0 + math.exp(-a_cell_sum)) # This is the sigmoid activation function
		result = sigmoid_activation
		return result


neuron= FeedForwardNetwork()
FFN=neuron.Forward(np.array([1,1]))
print(FFN)

import tensorflow as tf

x = tf.placeholder("float",[ ,4])
y = x * 2

with tf.Session() as session:
    input_data = np.array([1,2,3,4])
    input_data = np.expand_dims(input_data,axis=0)
    result = session.run(y, feed_dict={x: input_data})
    print(result)