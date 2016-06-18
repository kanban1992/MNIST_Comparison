import mnist_loader

import tensorflow as tf
import math
import numpy as np

import matplotlib.pyplot as plt



x_training,y_training,x_validation,y_validation,x_test,y_test = mnist_loader.load_data_wrapper()

N_training=len(x_training)
N_validation=len(x_validation)
N_test=len(x_test)

 
N_epochs = 5

learning_rate = 3.0	
batch_size = 10


N1 = 784 #equals N_inputs
N2 = 30
N3 = 30
N4 = 30
N5 = 10

N_in=N1
N_out=N5

x = tf.placeholder(tf.float32,[None,N1])#don't take the shape=(batch_size,N1) argument, because we need this for different batch sizes

W2 = tf.Variable(tf.random_normal([N1, N2],mean=0.0,stddev=1.0/math.sqrt(N1*1.0)))# Initialize the weights for one neuron with 1/sqrt(Number of weights which enter the neuron/ Number of neurons in layer before)
b2 = tf.Variable(tf.random_normal([N2]))
a2 = tf.sigmoid(tf.matmul(x, W2) + b2) #x=a1

W3 = tf.Variable(tf.random_normal([N2, N3],mean=0.0,stddev=1.0/math.sqrt(N2*1.0)))
b3 = tf.Variable(tf.random_normal([N3]))
a3 = tf.sigmoid(tf.matmul(a2, W3) + b3)

W4 = tf.Variable(tf.random_normal([N3, N4],mean=0.0,stddev=1.0/math.sqrt(N3*1.0)))
b4 = tf.Variable(tf.random_normal([N4]))
a4 = tf.sigmoid(tf.matmul(a3, W4) + b4)

W5 = tf.Variable(tf.random_normal([N4, N5],mean=0.0,stddev=1.0/math.sqrt(N4*1.0)))
b5 = tf.Variable(tf.random_normal([N5]))
y = tf.sigmoid(tf.matmul(a4, W5) + b5)

y_ = tf.placeholder(tf.float32,[None,N_out]) #  ,shape=(batch_size,N_out)


quadratic_cost= tf.scalar_mul(1.0/(N_training*2.0),tf.reduce_sum(tf.squared_difference(y,y_))) 

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(quadratic_cost)
init = tf.initialize_all_variables()

#launch the graph
sess = tf.Session()
sess.run(init)


#batch size of training input
N_training_batch=N_training/batch_size #rounds to samllest integer

correct=[0]*N_epochs
cost_training_data=[0.0]*N_epochs

for i in range(0,N_epochs):
	for j in range(0,N_training_batch):
		start=j*batch_size
		end=(j+1)*batch_size
		batch_x=x_training[start:end]
		batch_y=y_training[start:end]

		sess.run(train_step, feed_dict={x: batch_x, 
			y_: batch_y})

	perm = np.arange(N_training)
	np.random.shuffle(perm)
	x_training = x_training[perm]
	y_training = y_training[perm]

	
	#cost after each epoch
	cost_training_data[i]=sess.run(quadratic_cost, feed_dict={x: x_training, 
			y_: y_training})
	#correct predictions after each epoch
	y_out_validation=sess.run(y,feed_dict={x: x_validation})
	for k in range(0,len(y_out_validation)):
		arg=np.argmax(y_out_validation[k])
		if 1.0==y_validation[k][arg]:
			correct[i]+=1

	print "correct after "+str(i)+ " epochs: "+str(correct[i])



#plotting

plt.figure(1)
plt.title("Costfunction of Training-data")
plt.xlabel("epochs")
plt.ylabel("cost function")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,cost_training_data)
plt.savefig("cost_on_training_data.png")

plt.figure(2)
plt.title("correct classidied numbers (out of 10000)")
plt.xlabel("epochs")
plt.ylabel("evaluation accuracy")
x_range=[x+1 for x in range(0,N_epochs)]
plt.plot(x_range,correct)
plt.savefig("evaluation_accuracy.png")



