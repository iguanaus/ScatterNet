'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radi of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights. 
'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import argparse
from scatter_net_core import *

def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val,patienceLimit = 10):

    # This is to ensure similiar results throughout. 
    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Structuring output in 3 files. Train file, validation file, and test file. 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train_file_name = ""
    while True:
        train_file_name = output_folder+"/train_loss_" + str(numFile) + ".txt"
        if os.path.isfile(train_file_name):
            numFile += 1
        else:
            break
    train_loss_file = open(train_file_name,'w')
    val_loss_file = open(output_folder+"/val_loss_"+str(numFile) + ".txt",'w')
    train_loss_file = open(output_folder+"/test_loss_" + str(numFile) + ".txt",'w')

    # Getting the data. 

    train_X, train_Y , test_X, test_Y, val_X, val_Y = get_data(data,percentTest=percent_val)

    x_size = train_X.shape[1]
    y_size = train_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []
    biases = []

    # Weight initializations
    if reuse_weights:
        (weights, biases) = load_weights(output_folder,weight_name_load,num_layers)
    else:
        for i in xrange(0,num_layers):
            if i ==0:
                weights.append(init_weights((x_size,n_hidden)))
            else:
                weights.append(init_weights((n_hidden,n_hidden)))
            biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,y_size)))
        biases.append(init_bias(y_size))

    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)
    
    # Backward propagation
    dif = tf.abs(y-yhat)
    peroff = tf.reduce_mean(dif/tf.abs(y))
    cost = tf.reduce_mean(tf.square(y-yhat))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr_rate,global_step,int(train_X.shape[0]/n_batch),lr_decay,staircase=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

    #Now do the training. 
    step , curEpoch, cum_loss, numFile, perinc = 0
    lowVal = 1000000.0 #Just make this some high number. 
    
    start_time=time.time()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("========                         Iterations started                  ========")
        while curEpoch < numEpochs:
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            peroffinc, cuminc, _ = sess.run([peroff,cost,optimizer], feed_dict={X: batch_x, y: batch_y})
            cum_loss += cuminc
            perinc += peroffinc
            step += 1
            #End of each epoch. 
            if step ==  int(train_X.shape[0]/n_batch): 
                curEpoch +=1            
                cum_loss = cum_loss/float(step)
                perinc = perinc/float(step)
                step = 0
                train_loss_file.write(str(float(cum_loss))+"," + str(perinc) + str("\n"))
                # Every 10 epochs, do a validation. 
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    val_loss, peroff2 = sess.run([cost,peroff],feed_dict={X:test_X,y:test_Y})
                    val_loss_file.write(str(float(val_loss))+","+str(peroff2)+str("\n"))
                    val_loss_file.flush()
                    train_loss_file.flush()
                    if (val_loss > lowVal):
                          patience += 1
                    else:
                          patience = 0
                    lowVal = min(val_loss,lowVal)
                    print("Validation loss: " , str(val_loss) , " per off: " , peroff2)
                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss) + " : " + str(perinc))
                    if (patience > patienceLimit):
                        print("Reached patience limit. Terminating")
                        break
                cum_loss = 0
                perinc = 0
        save_weights(weights,biases,output_folder,weight_name_save,num_layers)

        # Output test loss. 
        finalLoss, finalPer = sess.run([cost,peroff],feed_dict={X:val_X,y:val_Y})
        train_loss_file.write(str(finalLoss) + "," + str(finalPer))
        train_loss_file.flush()

    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/7_layer_tio2') # Where the data file is. Note: This assumes a file of _val.csv and .csv 
    parser.add_argument("--reuse_weights",type=str,default='False') # Whether to load the weights or not. Note this just needs to be set to true, then the output folder directed to the same location. 
    parser.add_argument("--output_folder",type=str,default='results/7_Layer_TiO2') #Where to output the results to. Note: No / at the end. 
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights. In most cases, just leave this as is, it will naturally take care of it. 
    parser.add_argument("--weight_name_save",type=str,default="") #Similiar to above, but for saving now. 
    parser.add_argument("--n_batch",type=int,default=100) # Batch Size
    parser.add_argument("--numEpochs",type=int,default=5000) #Max number of epochs to consider at maximum, if patience condition is not met. 
    parser.add_argument("--lr_rate",default=.0006) # Learning Rate. 
    parser.add_argument("--lr_decay",default=.99) # Learning rate decay. It decays by this factor every epoch.
    parser.add_argument("--num_layers",default=4) # Number of layers in the network. 
    parser.add_argument("--n_hidden",default=225) # Number of neurons per layer. Fully connected layers. 
    parser.add_argument("--percent_val",default=.2) # Amount of the data to split for validation/test. The validation/test are both split equally. 
    parser.add_argument("--patience",default=10) # Patience for stopping. If validation loss has not decreased in this many steps, it will stop the training. 

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'data':dict['data'],
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'n_batch':dict['n_batch'],
            'numEpochs':dict['numEpochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val'],
            'patience':dict['patience']
            }

    main(**kwargs)




