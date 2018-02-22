import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def forwardprop(X, weights,num_layers,minLimit,maxLimit):
    X = tf.maximum(X,minLimit)
    X = tf.minimum(X,maxLimit)

    htemp = None
    for i in xrange(0, num_layers):
        print("Multiplying: " , i)
        if i ==0:
            htemp = tf.nn.sigmoid(tf.matmul(X,weights[i]))    
        else:   
            htemp = tf.nn.sigmoid(tf.matmul(htemp,weights[i]))
    yval = tf.matmul(htemp,weights[-1])
    return yval

def gen_data_first(data,test_file,data_folder):
    x_file = data+"_val.csv"
    y_file = data_folder+test_file+".csv"
    train_X = np.genfromtxt(x_file, delimiter=',')
    train_Y = np.array([np.genfromtxt(y_file, delimiter=',')]) #37

    train_train_X = np.genfromtxt(data+"_val.csv",delimiter=',')
    #I need the max and min of this. 
    print train_train_X.all()
    max_val = np.amax(train_train_X)
    min_val = np.amin(train_train_X)

    return train_X, train_Y , max_val, min_val

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    for i in xrange(0, num_layers+1):
        print("Opening: " , i)
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
    return weights

def main(data,data_folder,output_folder,weight_name_load,spect_file,init_list,num_layers,n_hidden,percent_val,lr_rate,lr_decay,num_iterations):
    train_X, train_Y, max_val, min_val = gen_data_first(data,spect_file,data_folder)

    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    X = tf.get_variable(name="b1", shape=[1,x_size], initializer=tf.constant_initializer(init_list))

    y = tf.placeholder("float", shape=[None, y_size])

    weights = load_weights(output_folder,weight_name_load,num_layers)

    # Forward propagation
    yhat    = forwardprop(X, weights,num_layers,min_val,max_val)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))

    topval = tf.abs(tf.matmul(y,tf.transpose(tf.abs(yhat)))) #This will select all the values that we want.
    botval = tf.abs(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat)))) #This will get the values that we do not want. 
    cost = topval/botval#botval/topval#topval#/botval

    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay).minimize(cost,var_list=[X])
    #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost,var_list=[X])

    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        curEpoch=0
        cum_loss = 0
        cost_file_name = output_folder+"design_train_loss.txt"
        cost_file = open(cost_file_name,'w')

        start_time=time.time()
        print("========                         Iterations started                  ========")

        while step < num_iterations:

            sess.run(optimizer, feed_dict={y: train_Y})
            loss = sess.run(cost,feed_dict={y:train_Y})
            cum_loss += loss        
            step += 1
            cost_file.write(str(float(cum_loss))+str("\n"))
            print("Step: " + str(step) + " : Loss: " + str(cum_loss) + " : " + str(X.eval()))
            cost_file.flush()
            cum_loss = 0

    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='/Users/johnpeurifoy/Documents/Skewl/PhotoNet/ScatteringNet/ScatteringNet_Matlab/data/CompleteDataFiles/5_layer_tio2_combined_06_20') #For input size.
    parser.add_argument("--data_folder",type=str,default='data/') #For where the test_answer is
    parser.add_argument("--output_folder",type=str,default='/Users/johnpeurifoy/Documents/Skewl/PhotoNet/ScatteringNet/ScatteringNet_Matlab/spectrums/5_layer_TiO2_200_layer/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--spect_file",type=str,default='test_answer')
    parser.add_argument("--init_list",type=str,default="50,50,50,50,50")
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=75)
    parser.add_argument("--percent_val",default=.2)
    parser.add_argument("--num_iterations",default=200000)
    parser.add_argument("--lr_rate",default=.05)
    parser.add_argument("--lr_decay",default=.9)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
    dict['init_list']=dict['init_list'].split(',')
    dict['init_list'] = [int(ele) for ele in dict['init_list']]
        
    kwargs = {  
        'data':dict['data'],
        'data_folder':dict['data_folder'],
        'output_folder':dict['output_folder'],
        'weight_name_load':dict['weight_name_load'],
        'spect_file':dict['spect_file'],
        'init_list':dict['init_list'],
        'num_layers':dict['num_layers'],
        'n_hidden':dict['n_hidden'],
        'percent_val':dict['percent_val'],
        'lr_rate':dict['lr_rate'],
        'lr_decay':dict['lr_decay'],
        'num_iterations':dict['num_iterations']
        }

    main(**kwargs)









