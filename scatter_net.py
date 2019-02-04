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

# This method is used when comparing spectrums. It outputs the desired/output to a file so it can be graphed and compared.
# Output Folder - the folder to go into
# Spec To Sample - number of the spectrum (for record keeping)
# Batch X - The input values (thicknesses)
# Batch Y - The output values (spectrum) ideal
# myvals0 - The output values (spectrum) approximation
# cost - the cost of the analysis. 
def outputSpectsToFile(output_folder,spect_to_sample,batch_x,batch_y,myvals0,cost,x_mean,x_std):
    filename = output_folder + "/test_out_file_"+str(spect_to_sample) + ".txt"
    f = open(filename,'w')
    f.write("XValue\nActual\nPredicted\n")
    xVals = batch_x[0]*x_std + x_mean
    for i in list(xVals):
        f.write(str(i)+",")
    f.write('\n')
    for item in list(batch_y[0]):
        f.write(str(item) + ",")
    f.write("\n")
    for item in list(myvals0[0]):
        f.write(str(item) + ",")
    f.flush()
    f.write("\n")
    f.flush()
    f.close()
    print("Cost: " , cost)
    print(myvals0)
    print("Wrote to: " + str(filename))

#This gest a data file for spectrum matching. 
def gen_data_first(data,test_file,data_folder=None):
    x_file = data+"_val.csv"
    y_file = test_file
    train_X = np.genfromtxt(x_file, delimiter=',')
    train_Y = np.array([np.genfromtxt(y_file, delimiter=',')])
    train_train_X = np.genfromtxt(data+"_val.csv",delimiter=',')
    print train_train_X.all()
    max_val = np.amax(train_train_X)
    min_val = np.amin(train_train_X)
    return train_X, train_Y , max_val, min_val

# This designs a spectrum. 
def design_spectrum(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val,patienceLimit,compare,sample_val,spect_to_sample,matchSpectrum,match_test_file,designSpectrum,design_test_file):
    print(num_layers)
    train_X, train_Y, max_val, min_val = gen_data_first(data,design_test_file)

    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)
    init_list_rand = tf.constant(np.random.rand(1,x_size)*40.0+30.0,dtype=tf.float32)

    X = tf.get_variable(name="b1", initializer=init_list_rand)
    y = tf.placeholder("float", shape=[None, y_size])
    print(num_layers)
    weights, biases = load_weights(output_folder,weight_name_load,num_layers)
    print(num_layers)
    print(type(num_layers))
    #This is the lambda list
    lambdaList = range(400,801,2)
    myl = np.array(lambdaList)
    newL = 1.0/(myl*myl*3).astype(np.float32)
    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers,minLimit=30,maxLimit=70)    
    # This will scale by the wavelength
    yhat = tf.multiply(yhat,newL)
    # Backward propagation
    topval = tf.abs(tf.matmul(y,tf.transpose(tf.abs(yhat)))) #This will select all the values that we want.
    botval = tf.abs(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat)))) #This will get the values that we do not want. 
    cost = topval/botval#botval/topval#topval#/botval
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay).minimize(cost,var_list=[X])
    global_step = tf.Variable(0, trainable=False)


    learning_rate = tf.train.exponential_decay(lr_rate,global_step,1000,lr_decay,staircase=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate).minimize(cost,global_step=global_step,var_list=[X])

    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        curEpoch=0
        cum_loss = 0
        cost_file_name = output_folder+"/design_train_loss.txt"
        cost_file = open(cost_file_name,'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while step < numEpochs*10:

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

# This matches a spectrum.
def match_spectrum(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val,patienceLimit,compare,sample_val,spect_to_sample,matchSpectrum,match_test_file,designSpectrum,design_test_file):
    # Change made on 08/16. Switching to gen_data_first. 
    train_X, train_Y, max_val, min_val = gen_data_first(data,design_test_file)
    #train_X, train_Y , test_X, test_Y, val_X, val_Y , x_mean, x_std = get_data(match_test_file,percentTest=0)
    print("Train x shape is: " , train_X.shape)
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)
    i = 0
    succes = 0
    start_time=time.time()
    weights, biases = load_weights(output_folder,weight_name_load,num_layers)
    while i < 50:
        i += 1
        init_list_rand = tf.constant(np.random.rand(1,x_size)*40.0+30.0,dtype=tf.float32)
        X = tf.get_variable(name="b1"+ str(i), initializer=init_list_rand)
        y = tf.placeholder("float", shape=[None, y_size])
        # Forward propagation
        yhat    = forwardprop(X, weights,biases,num_layers)
        # Backward propagation
        #extra_cost = tf.reduce_sum(tf.square(tf.minimum(X,30)-30) + tf.square(tf.maximum(X,70)-70))
        #Advanced Version
        cost = tf.reduce_sum(tf.square(y-yhat))#+5.0*extra_cost*extra_cost
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate, beta2=lr_decay,epsilon=0.1).minimize(cost,var_list=[X])
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0       
            cum_loss = 0
            cost_file_name = output_folder+"match_train_loss.txt"
            cost_file = open(cost_file_name,'w')            
            print("========                         Iterations started                  ========")
            prev_losses = 0
            while step < numEpochs*10:
                sess.run(optimizer,feed_dict={y:train_Y})
                cum_loss += sess.run(cost,feed_dict={y:train_Y})
                step += 1
                cost_file.write(str(float(cum_loss))+str("\n"))
                myvals0 = sess.run(yhat,feed_dict={y:train_Y})
                print("Step: " + str(step) + " : Loss: " + str(cum_loss) + " : " + str(X.eval()))
                if abs(cum_loss-prev_losses) < .00001:
                    print("Converged once")
                    break
                prev_losses = cum_loss
                cum_loss = 0
        if cum_loss < 0.1:
            succes += 1
            print succes
    print "Final: " , succes
    pass


def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,match_test_file,design_test_file,matchSpectrum,designSpectrum,percent_val,patienceLimit = 10,compare=False,sample_val=True,spect_to_sample=300):

    # This is to ensure similiar results throughout. 
    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Structuring output in 3 files. Train file, validation file, and test file. 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train_file_name = ""
    numFile = 0 # How many times you have trained it. 
    while True:
        train_file_name = output_folder+"/train_loss_" + str(numFile) + ".txt"
        if os.path.isfile(train_file_name):
            numFile += 1
        else:
            break
    train_loss_file = open(train_file_name,'w')
    val_loss_file = open(output_folder+"/val_loss_"+str(numFile) + ".txt",'w')
    test_loss_file = open(output_folder+"/test_loss_" + str(numFile) + ".txt",'w')
    spec_file = open(output_folder+"/spec_file_" + str(numFile) + ".txt",'w')

    # Getting the data. 

    train_X, train_Y , test_X, test_Y, val_X, val_Y , x_mean, x_std = get_data(data,percentTest=percent_val)

    # Write the mean/std to the file.
    out_mean = [str(float(i)) for i in list(x_mean)]
    out_std = [str(float(i)) for i in list(x_std)]
    spec_file.write(','.join(out_mean)+'\n' + ','.join(out_std)+'\n')
    spec_file.flush()
    spec_file.close()

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
    print("LR Rate: " , lr_rate)
    print(int(train_X.shape[0]/n_batch))
    print(lr_decay)
    print("--done--")
    learning_rate = tf.train.exponential_decay(lr_rate,global_step,int(train_X.shape[0]/n_batch),lr_decay,staircase=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

    #Now do the training. 
    step =0; curEpoch =0; cum_loss =0; perinc = 0;
    lowVal = 1000000.0 #Just make this some high number. 

    start_time=time.time()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        if (compare): #Just run a comparison
            x_set = train_X
            y_set = train_Y
            if sample_val:
                x_set = val_X
                y_set = val_Y
            batch_x = x_set[spect_to_sample : (spect_to_sample+1) ]
            batch_y = y_set[spect_to_sample : (spect_to_sample+1) ]
            mycost = sess.run(cost,feed_dict={X:batch_x,y:batch_y})
            myvals0 = sess.run(yhat,feed_dict={X:batch_x,y:batch_y})
            outputSpectsToFile(output_folder,spect_to_sample,batch_x,batch_y,myvals0,mycost,x_mean,x_std)
            return
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
        test_loss_file.write(str(finalLoss) + "," + str(finalPer))
        test_loss_file.flush()

    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/5_layer_tio2') # Where the data file is. Note: This assumes a file of _val.csv and .csv 
    parser.add_argument("--reuse_weights",type=str,default='False') # Whether to load the weights or not. Note this just needs to be set to true, then the output folder directed to the same location. 
    parser.add_argument("--output_folder",type=str,default='results/5_layer_tio2') #Where to output the results to. Note: No / at the end. 
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights. In most cases, just leave this as is, it will naturally take care of it. 
    parser.add_argument("--weight_name_save",type=str,default="") #Similiar to above, but for saving now. 
    parser.add_argument("--n_batch",type=int,default=100) # Batch Size
    parser.add_argument("--numEpochs",type=int,default=5000) #Max number of epochs to consider at maximum, if patience condition is not met. 
    parser.add_argument("--lr_rate",type=float,default=.001) # Learning Rate. 
    parser.add_argument("--lr_decay",type=float,default=.7) # Learning rate decay. It decays by this factor every epoch.
    parser.add_argument("--num_layers",default=4) # Number of layers in the network. 
    parser.add_argument("--n_hidden",default=225) # Number of neurons per layer. Fully connected layers. 
    parser.add_argument("--percent_val",default=.2) # Amount of the data to split for validation/test. The validation/test are both split equally. 
    parser.add_argument("--patience",default=10) # Patience for stopping. If validation loss has not decreased in this many steps, it will stop the training. 
    parser.add_argument("--compare",default='False') # Whether it should output the comparison or not. 
    parser.add_argument("--sample_val",default='True') # Wether it should sample from validation or not, for the purposes of graphing. 
    parser.add_argument("--spect_to_sample",type=int,default=300) # Zero Indexing for this. Position in the data file to sample from (note it will take from validation)
    parser.add_argument("--matchSpectrum",default='False') # If it should match an already existing spectrum file. 
    parser.add_argument("--match_test_file",default='results/2_layer_tio2/test_47.5_45.3') # Location of the file with the spectrum in it. 
    parser.add_argument("--designSpectrum",default='False') # If it should 
    parser.add_argument("--design_test_file",default='data/test_gen_spect.csv') # This is a file that should contain 0's and 1's where it should maximize and not maximize. 

    args = parser.parse_args()
    dict = vars(args)
    print(dict)

    for key,value in dict.iteritems():
        if (dict[key]=="False"):
            dict[key] = False
        elif dict[key]=="True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    print (dict)

    #Note that reuse MUST be set to true.
    if (dict['compare'] or dict['matchSpectrum'] or dict['designSpectrum']):
        if dict['reuse_weights'] != True:
            print("Reuse weights must be set true for comparison, matching, or designing. Setting it to true....")
            time.sleep(1)
        dict['reuse_weights'] = True
        
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
            'num_layers':int(dict['num_layers']),
            'n_hidden':int(dict['n_hidden']),
            'percent_val':dict['percent_val'],
            'patienceLimit':dict['patience'],
            'compare':dict['compare'],
            'sample_val':dict['sample_val'],
            'spect_to_sample':dict['spect_to_sample'],
            'matchSpectrum':dict['matchSpectrum'],
            'match_test_file':dict['match_test_file'],
            'designSpectrum':dict['designSpectrum'],
            'design_test_file':dict['design_test_file']
            }

    if kwargs['designSpectrum'] == True:
        design_spectrum(**kwargs)
    elif kwargs['matchSpectrum'] == True:
        match_spectrum(**kwargs)
    else:
        main(**kwargs)




