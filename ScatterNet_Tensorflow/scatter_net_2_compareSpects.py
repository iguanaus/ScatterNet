import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os
#from numpy import genfromtxt

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    for i in xrange(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

def forwardprop(X, weights, biases, num_layers):
    htemp = None
    for i in xrange(0, num_layers):
        if i ==0:
            htemp = tf.add(tf.nn.relu(tf.matmul(X,weights[i])),biases[i])    
        else:   
            htemp = tf.add(tf.nn.relu(tf.matmul(htemp,weights[i])),biases[i])
        print("Bias: " , i, " : ", biases[i])
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    print("Last bias: " , biases[-1])
    return yval

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(data,percentTest=.2,random_state=42):
    x_file = data+"_val.csv"
    y_file = data+".csv"
    train_X = np.genfromtxt(x_file,delimiter=',')
    train_Y = np.transpose(np.genfromtxt(y_file,delimiter=','))
    try:
        i = len(train_X[0])
        X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)

    except:
        train_X = np.array([train_X])
        train_Y = np.array([train_Y])
        X_train = train_X
        X_val = X_train
        y_train = train_Y
        y_val = y_train
    #if True:
    #    y_train = y_train[:,0:-1]
    #    y_val = y_val[:,0:-1]
    print("Train X: " , train_X)
    print("Train Y: " , train_Y)
    return X_train, y_train, X_val, y_val



def main(data,output_folder,weight_name_load,spect_to_sample,sample_val,num_layers,n_hidden,percent_val):

    if not os.path.exists(output_folder):
        print("ERROR THERE IS NO OUTPUT FOLDER. PLEASE SYNC THIS TO PART 1")

    train_X, train_Y , val_X, val_Y = get_data(data,percentTest=percent_val)

    x_size = train_X.shape[1]
    y_size = train_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    weights, biases = load_weights(output_folder,weight_name_load,num_layers)

    yhat    = forwardprop(X, weights,biases,num_layers)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))
    
    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        start_time=time.time()
        print("========                         Iterations started                  ========")
        x_set = None
        y_set = None
        if sample_val:
            x_set = val_X
            y_set = val_Y
        else:
            x_set = train_X
            y_set = train_Y
        print("x_set: " , x_set)

        batch_x = x_set[spect_to_sample : (spect_to_sample+1) ]
        batch_y = y_set[spect_to_sample : (spect_to_sample+1) ]
        print("Batch x: " , batch_x)

        mycost = sess.run(cost,feed_dict={X:batch_x,y:batch_y})
        myvals0 = sess.run(yhat,feed_dict={X:batch_x,y:batch_y})

        filename = output_folder + "test_out_file_"+str(spect_to_sample) + ".txt"
        f = open(filename,'w')
        f.write("XValue\nActual\nPredicted\n")
        print("Batch: " , batch_x)
        f.write(str(batch_x[0])+"\n")
        for item in list(batch_y[0]):
            f.write(str(item) + ",")
        f.write("\n")
        for item in list(myvals0[0]):
            f.write(str(item) + ",")
        f.write("\n")
        f.flush()
        f.close()
        print("Cost: " , mycost)
        print(myvals0)
        print("Wrote to: " + str(filename))

    print "========Writing completed in : " + str(time.time()-start_time) + " ========"
        
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='/Users/johnpeurifoy/Documents/skewl/PhotoNet/ScatteringNet/ScatteringNet_Matlab/data/jagg_layer_tio2_fixed_06_21_1')
    parser.add_argument("--output_folder",type=str,default='results/J-Aggregate/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--spect_to_sample",type=int,default=300) #Zero Indexing
    parser.add_argument("--sample_val",type=str,default="True")
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=100)
    parser.add_argument("--percent_val",default=0.2)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'data':dict['data'],
            'output_folder':dict['output_folder'],
            'weight_name_load':dict['weight_name_load'],
            'spect_to_sample':dict['spect_to_sample'],
            'sample_val':dict['sample_val'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val']
            }

    main(**kwargs)
