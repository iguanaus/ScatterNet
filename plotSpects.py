#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

#This returns 3 numbers and 6 lists. Namely in num_list_list, num_list_list, num_list_list

def read_file(filename):
	print('Reading: ' , filename)
	fo = open(filename,'r')
	line1 = fo.readline()
	print("line1: " , line1)
	line2 = fo.readline()
	print(line2)
	line3 = fo.readline()
	print(line3)
	name = '/'.join([str(round(float(x),1)) for x in fo.readline()[:-2].split(',')])
	print(name)
	a1 = [float(x) for x in fo.readline()[:-2].split(',')]
	p1 = [float(x) for x in fo.readline()[:-2].split(',')]
	return name,a1, p1	

def plotFile(filename):


	legend = []
	name,a1,p1 = read_file(filename)
	legend.append(str(name) + "_actual")
	legend.append(str(name)+"_predicted")
	plt.plot(range(400,800,2),a1)
	plt.plot(range(400,800,2),p1)
	plt.title('Comparing spectrums')
	plt.ylabel("Cross Scattering Amplitude")
	plt.xlabel("Wavelength (nm)")
	plt.legend(legend, loc='top left')
	plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--filename",type=str,default='results/8_layer_tio2/test_out_file_20.txt') # 

    args = parser.parse_args()
    dict = vars(args)
    kwargs = {  
            'filename':dict['filename']
            }
    plotFile(**kwargs)




