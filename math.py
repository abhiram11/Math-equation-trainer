import tensorflow as tf
import numpy as np 

x = np.random.rand(1000).astype(np.float32)
y = np.random.rand(1000).astype(np.float32)

z = 3*x + 4*y + 5

#tensorflow code
wx = tf.Variable(tf.random_uniform([1],-1.0,1.0))
wy = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))

z_pred = wx*x + wy*y + bias

loss = tf.reduce_mean(tf.square(z_pred-z))
optimizer = tf.train.GradientDescentOptimizer(0.2) #learning rate

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(1001):
	sess.run(train) #just to print the answer every 20 steps

print(sess.run(wx),sess.run(wy),sess.run(bias))
