import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n = 2000
dimens = 15
Input_1 = np.array(genfromtxt('C:\\Users\\Lakshay\\Desktop\\ML in catalysis\\dimer\\Input_1.csv', delimiter=','))
Input_2 = np.array(genfromtxt('C:\\Users\\Lakshay\\Desktop\\ML in catalysis\\dimer\\Input_2.csv', delimiter=','))
Output_1 = np.array(genfromtxt('C:\\Users\\Lakshay\\Desktop\\ML in catalysis\\dimer\\Hexadecapole_Output_1.csv', delimiter=','))
Output_2 = np.array(genfromtxt('C:\\Users\\Lakshay\\Desktop\\ML in catalysis\\dimer\\Hexadecapole_Output_2.csv', delimiter=','))
Input = np.concatenate((Input_1, Input_2))
Output = np.concatenate((Output_1, Output_2))
count = 0
for i in range(n):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9 and Output[i][3] == 9):
        count+=1
for i in range(n-count):
    if(Output[i][0] == 9 and Output[i][1] == 9 and Output[i][2] == 9 and Output[i][3] == 9):
        Input = np.delete(Input, i, 0)
        Output = np.delete(Output, i, 0)
print(count)
n = 1477
print(np.shape(Output))
Data = np.concatenate((Input, Output), axis = 1)
for i in range(6+dimens):
    max = -1000
    min = 1000
    for j in range(1970):
        if Data[j][i] > max:
            max = Data[j][i]
        if Data[j][i] < min:
            min = Data[j][i]
    for j in range(1970):
        Data[j][i] = (Data[j][i] - min)/(max - min)
data, test_data= train_test_split(Data, test_size= 0.25, random_state= 56)
#data, test_data= Data[:1477], Data[1477:]  
Input, Output = (data.T[:6]).T, (data.T[6:]).T
test_input, test_output = (test_data.T[:6]).T, (test_data.T[6:]).T


#Input = normalize(Input, 'l2', 0)
#Output = normalize(Output, 'l2', 0)
#test_input = normalize(test_input, 'l2', 0)
#test_output = normalize(test_output, 'l2', 0)
n = np.shape(Input)[0]
m = np.shape(test_input)[0]
dim = 1
min = 10
ind = 0
for i in range(1):
    layer1 = 50
    layer2 = 1
    dim = 1
    x=tf.placeholder(tf.float32, [None, 6])
    y=tf.placeholder(tf.float32, [None, dim])
    w1=tf.Variable(tf.random_normal([6, layer1], stddev=0.05), name="w1")
    w2=tf.Variable(tf.random_normal([layer1, layer2], stddev=0.05), name="w2")
    #w3=tf.Variable(tf.random_normal([layer2, dim], stddev=0.05), name='w3')
    #w4=tf.Variable(tf.random_normal([50, dim], stddev=0.05), name='w4')
    b1 = tf.Variable(tf.random_normal([layer1]), name='b1')
    b2 = tf.Variable(tf.random_normal([layer2]), name='b2')
    #b3 = tf.Variable(tf.random_normal([dim]), name='b3')
    #b4 = tf.Variable(tf.random_normal([dim]), name='b4')
    a1=tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
    #a2=tf.nn.tanh(tf.add(tf.matmul(a1, w2), b2))
    #a3=tf.nn.relu(tf.add(tf.matmul(a2, w3), b3))
    y_pred=tf.nn.sigmoid(tf.add(tf.matmul(a1, w2), b2))
    #y_pred=tf.clip_by_value(y_pred, 0.0000001,0.9999999)
    y_pred=tf.clip_by_value(y_pred, 0.0000001,0.9999999)
    regul=(tf.nn.l2_loss(w2)+tf.nn.l2_loss(w1))/5000000
    avg = tf.reduce_sum(tf.pow(y - tf.reduce_sum(y, 0)/n, 2), 0)
    MSE_train = tf.reduce_sum(tf.pow(y_pred-y, 2), 0) + regul
    MSE_true = tf.reduce_sum(tf.pow(y_pred-y, 2), 0)
    optimiser = tf.train.AdamOptimizer(0.005).minimize(MSE_train)
    #y_pred=tf.nn.softmax(tf.add(tf.matmul(a2, w3), b3))
    #accuracy /= tf.reduce_sum(y, 0)
    init=tf.global_variables_initializer()
    axi = 14
    me_train = np.sum(np.square(Output[:,axi].reshape((n, 1)) - (np.sum(Output[:,axi].reshape((n, 1)))/n)))
    me_test = np.sum(np.square(test_output[:,axi].reshape((m, 1)) - np.sum(test_output[:,axi].reshape((m, 1)))/m))
    print(me_train)
    print(me_test)
    mini = 5000
    minacc = 100000
    with tf.device('/gpu:0'):
        runs = 10000
        with tf.Session() as sess:
            sess.run(init)
            train_error = []
            test_error = []
            for run in range(runs):
                batch_x, batch_y = Input, Output[:,axi].reshape((n, 1))
                #print(sess.run([MSE], {x: batch_x, y: batch_y}))
                #batch_x, batch_y = Input, Output
                test_x, test_y = test_input, test_output[:,axi].reshape((m, 1))
                #test_x, test_y = test_input, test_output
                a, c = sess.run([optimiser, MSE_true],{x: batch_x, y: batch_y})
                p  = sess.run([MSE_true], {x: test_x, y: test_y})
                c/=me_train
                train_error.append(c)
                test_error.append(p[0]/me_test)
                print("epoch:", (run + 1), "cost =", c)
                print("Accuracy = ", p[0]/me_test)
                if p[0]/me_test < minacc:
                    minacc = p[0]/me_test
                sum = 0
                Y = sess.run([y_pred], {x:test_x})
                for i in range(m):
                #print((Y[0][i][0]-test_y[i][0])/Y[0][i][0])
                    sum+=abs((Y[0][i][0]-test_y[i][0])/Y[0][i][0])
                sum/=m
                if(sum < mini):
                    mini = sum
            plt.plot(train_error[200:])
            plt.plot(test_error[200:])
            plt.show()
            print(mini)
                #print(sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
print(min)
print(minacc)
