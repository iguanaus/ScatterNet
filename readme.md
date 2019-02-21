---------------------- 
    Scatter Net     
      '.\|/.'         
      (\   /)         
      - -O- -         
      (/   \)         
      ,'/|\'.         
---------------------- 

# Scatter Net

An example repository of using machine learning to solve a physics problem. Based on the work presented in, Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks (https://arxiv.org/abs/1712.03222). This repository is specifically designed for solving inverse design problems, particularly surrounding photonics and optics.

## Geting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. This example will also generate Table I in the paper, and Figure 2,3, and 4.

### Prerequisites

To run the Matlab code, Matlab will need to be installed. For this code, we used Matlab R2017a. Note that the project can be done without Matlab, but comparisons of speed and data generation cannot be done unless Matlab is installed.

This codebase is based on Python 2.7, and the pip packages used are shown in the requirements.txt file. To run this on AWS, use AMI ami-52bb0c32, and a p2.xlarge instance. 

### Installing

1. Copy the github repo to your computer, and install the pip requirements.
```
git clone https://github.com/iguanaus/ScatterNet.git
cd ScatterNet
pip install -r requirements.txt
```
2. Option 1: Fetch the data
```
cd data
sh fetchData.sh
```
2. Option 2: View and Generate the data
```
scatter_sim_1_plot_data.m
scatter_sim_2_gen_data.m
```
3. Option 1: Fetch the models 
```
cd results
sh fetchResults.sh
```
3. Option 2: Train the models (Table I)
```
sh demo.sh
```
4. Compare spetrums (Figure 2)
```
sh demo_compareSpect.sh
```
5. Perform Inverse Design (Figure 3)
```
sh demo_matchSpect.sh
```
6. Perform Optimization (Figure 4)
```
sh demo_designSpect.sh
```

## Structure
  ScatteringNet_Matlab:
    This is the matlab code repository, intended to be run on a cluster or a high performance computer. Depends on matlab.

  ScatteringNet_Tensorflow:
    This is the tensorflow/python repository, intended to be run on a computer with a GPU and tensorflow capabilities.

Flow:
  1. scatter_0_generate_spectrum
          Pick the settings for your data in the scatter_0_generate_spectrum.
  2. scatter_1_plot_sample
        Run the scatter_1_plot_sample to get an idea of what the data looks like.
          Make sure the data set is hollistics enough/has interesting features within it.
          Save these graphs, so you have an idea of what the data looks like. 
          plotLoss.py is your friend.
          Use the pullFiles.sh script to pull the data locally from the server.
  3. scatter_2_generate_train
        Once you have that, run the scatter_2_generate_train on a cluster
          I recommend first changing the settings, then pushing it to the server.
  4. scatter_net_1_train
        Once you have the data, run the scatter_net_1_train to train the neural network on a GPU.
          Graph the loss.
  5. scatter_net_2_compareSpects
        Once you have the trained neural network, run the scatter_net_2_compareSpects.py to sample some spects and see what they are.
          Run plotSpects.py to see what these spectrums look like.
  6. scatter_3_generate_single_test
        Pick a spectrum, generate the data, move it over to the other repository. 
  7. scatter_net_3_matchSpect
        See how it matches the spectrum.
  8. scatter_4_graph_geometry
        See how it did
  9. scatter_net_4_design.py
        Pick an optimal figure of merit, and then run this.
 10. scatter_5_graph_desired
        Graph the desired on top. 


## Contributing



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc








---------------------- 
    Scatter Net     
      '.\|/.'         
      (\   /)         
      - -O- -         
      (/   \)         
      ,'/|\'.         
---------------------- 
 
MIT Department of Physics. All rights reserved.
Version 1.0 - 06/10/2017
Produced and used by John Peurifoy. Assistance and guidance provided by: Li Jing, Yichen Shen, and Yi Yang. Updates and code fixes provided by Samuel Kim. 
A product of a collaboration between Max Tegmark's and Marin Soljacic's group. 
Originally created 04/24/2017











