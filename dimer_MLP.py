import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

n = 2000
Input_1 = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\Input_1.csv', delimiter=','))
Input_2 = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\Input_2.csv', delimiter=','))
Output_1 = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\Dipole_Output.csv', delimiter=','))
Output_2 = np.array(genfromtxt('C:\\Users\\lakshay\\Desktop\\ML in Catalalysis\\train_data-1\\Dipole_Output.csv', delimiter=','))
Input = np.concatenate((Input_1, Input_2))
Output = np.concatenate((Output_1, Output_2))
count = 0
for i in range(n):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9):
        count+=1
for i in range(n-count):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9):
        Input = np.delete(Input, i, 0)
        Output = np.delete(Output, i, 0)
Input = normalize(Input, 'l2', 0)
Output = normalize(Output, 'l2', 0)
n = 1477
Input, test_input= train_test_split(Input, test_size= 0.25, random_state= 4)
Output, test_output= train_test_split(Output, test_size= 0.25, random_state= 4)
Input = normalize(Input, 'l2', 0)
Output = normalize(Output, 'l2', 0)
test_input = normalize(test_input, 'l2', 0)
test_output = normalize(test_output, 'l2', 0)
n = np.shape(Input)[0]
m = np.shape(test_input)[0]
dim = 3
x=tf.placeholder(tf.float32, [None, 6])
y=tf.placeholder(tf.float32, [None, dim])
w1=tf.Variable(tf.random_normal([6, 200], stddev=0.05), name="w1")
w2=tf.Variable(tf.random_normal([200, 100], stddev=0.05), name="w2")
w3=tf.Variable(tf.random_normal([100, dim], stddev=0.05), name='w3')
#w4=tf.Variable(tf.random_normal([50, dim], stddev=0.05), name='w4')
b1 = tf.Variable(tf.random_normal([200]), name='b1')
b2 = tf.Variable(tf.random_normal([100]), name='b2')
b3 = tf.Variable(tf.random_normal([dim]), name='b3')
#b4 = tf.Variable(tf.random_normal([dim]), name='b4')
a1=tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
a2=tf.nn.relu(tf.add(tf.matmul(a1, w2), b2))
#a3=tf.nn.relu(tf.add(tf.matmul(a2, w3), b3))
y_pred=tf.nn.sigmoid(tf.add(tf.matmul(a2, w3), b3))
#y_pred=tf.clip_by_value(y_pred, 0.0000001,0.9999999)
regul=(tf.nn.l2_loss(w2)+tf.nn.l2_loss(w1))/200000
MSE = tf.reduce_sum(tf.pow(y_pred-y, 2), 0)
optimiser = tf.train.AdamOptimizer(0.005).minimize(MSE)
#y_pred=tf.nn.softmax(tf.add(tf.matmul(a2, w3), b3))
y_pred=tf.clip_by_value(y_pred, 0.0000001,0.9999999)
accuracy /= tf.reduce_sum(y, 0)
init=tf.global_variables_initializer()
ith tf.device('/gpu:0'):
    runs = 3000
    with tf.Session() as sess:
        sess.run(init)
        for run in range(runs):
            #batch_x, batch_y = Input, Output[:,1].reshape((n, 1))
            #print(sess.run([MSE], {x: batch_x, y: batch_y}))
            batch_x, batch_y = Input, Output
            test_x, test_y = test_input, test_output
            a, c = sess.run([optimiser, MSE],{x: batch_x, y: batch_y})
            p  = sess.run([MSE], {x: test_x, y: test_y})
            print("epoch:", (run + 1), "cost =", c)
            print("Accuracy = ", p)
            #print(sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
