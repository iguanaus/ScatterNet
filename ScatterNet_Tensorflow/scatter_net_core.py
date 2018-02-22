'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radi of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights
    This file is the core file containing all necesary bits. 
'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import argparse

#As per Xaiver init, this should be 2/n(input), though many different initializations can be tried. 
def init_weights(shape,stddev=.1):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def init_bias(shape, stddev=.1):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    for i in xrange(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"/b_"+str(i)+".txt",bias_i,delimiter=',')
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    for i in xrange(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"/w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"/b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    for i in xrange(0, num_layers):
        if i ==0:
            htemp = tf.nn.relu(tf.add(tf.matmul(X,weights[i]),biases[i]))
        else:   
            htemp = tf.nn.relu(tf.add(tf.matmul(htemp,weights[i]),biases[i]))
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    return yval

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(data,percentTest=.2,random_state=42):
    x_file = data+"_val.csv"
    y_file = data+".csv"
    train_X = np.genfromtxt(x_file,delimiter=',')#[0:20000,:]
    train_Y = np.transpose(np.genfromtxt(y_file,delimiter=','))#[0:20000,:]
    train_X = (train_X-train_X.mean(axis=0))/train_X.std(axis=0)
    X_train, test_X, y_train, test_Y = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(test_X,test_Y,test_size=.5,random_state=random_state)
    return X_train, y_train, X_test, y_test, X_val, y_val




