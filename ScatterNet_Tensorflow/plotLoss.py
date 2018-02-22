#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
#'results/Dielectric_Massive/train_train_loss_0.txt',
direc='results/TestingExpDecay/'
loss_files=[direc+"train_val_loss_8_val.txt"]

#loss_files=['results/3_Layer_TiO2_100_layer/traintrainl_loss_5.txt','results/3_Layer_TiO2_100_layer/traintrainl_loss_6.txt','results/3_Layer_TiO2_100_layer/train_val_loss_7_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_8_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_9_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_10_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_11_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_12_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_13_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_14_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_15_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_16_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_17_val.txt','results/3_Layer_TiO2_100_layer/train_val_loss_18_val.txt']

#loss_files_2=['results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_0_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_1_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_2_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_3_val.txt']

#Dielectric_Order_BiasTest/train_val_loss_1_val_bias_test.txt','results/Dielectric_Order_BiasTest/train_val_loss_val_bias_test_sigmoid_nobias.txt']
listLoss = []
for i in xrange(0, len(loss_files)):
	myls = list(np.genfromtxt(loss_files[i],delimiter=','))
	for ele in myls:
		listLoss.append(ele/200.0*100.0)
	#listLoss.append(np.genfromtxt(loss_files[i],delimiter=','))
#lossValues = listLoss
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#print(listLoss)
#print("ListLoss")
lossValues=np.array(listLoss)
lossValues=moving_average(lossValues,10)
#lossValues = np.append(np.genfromtxt(loss_files[0],delimiter=','),listLoss)

#print(lossValues+lossValues2)

#lossValues_2_1 = np.genfromtxt(loss_files[0],delimiter=',')
#lossValues_2_2 = np.genfromtxt(loss_files[1],delimiter=',')
# lossValues_2_3 = np.genfromtxt(loss_files[3],delimiter=',')
# lossValues_2_4 = np.genfromtxt(loss_files[5],delimiter=',')
# lossValues_2_5 = np.genfromtxt(loss_files[7],delimiter=',')
# lossValues_2_6 = np.genfromtxt(loss_files[9],delimiter=',')
# lossValues_2_7 = np.genfromtxt(loss_files[10],delimiter=',')
# lossValues_2_8 = np.genfromtxt(loss_files[10],delimiter=',')
# lossValues_2_9 = np.genfromtxt(loss_files[10],delimiter=',')
# lossValues_2_10 = np.genfromtxt(loss_files[10],delimiter=',')
# lossValues_2_11 = np.genfromtxt(loss_files[10],delimiter=',')

#myList = [lossValues_2_2,lossValues_2_3,lossValues_2_4,lossValues_2_5,lossValues_2_6,lossValues_2_7,lossValues_2_8,lossValues_2_9,lossValues_2_10]

#lossValues_2 = np.append(lossValues_2_1,myList)


#lossValues_22 = np.genfromtxt(loss_files_2[1],delimiter=',')
# lossValues_23 = np.genfromtxt(loss_files_2[2],delimiter=',')
# lossValues_24 = np.genfromtxt(loss_files_2[3],delimiter=',')
# lossValues_2 = np.append(lossValues_2,[lossValues_22,lossValues_23,lossValues_24])
#plt.plot(lossValues)
#plt.plot(lossValues2)
plt.xlabel('Epochs Trained (in 10s)')
plt.ylabel('MSE Training Error - Percent off per point')
plt.title('Training Error')
print(lossValues)
plt.plot(lossValues)

#plt.plot(lossValues_2)

plt.show()