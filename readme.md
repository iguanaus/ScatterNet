---------------------- 
    Scatter Net     
      '.\|/.'         
      (\   /)         
      - -O- -         
      (/   \)         
      ,'/|\'.         
---------------------- 

Structure:
  
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
Produced and used by John Peurifoy. Assistance and guidance provided by: Li Jing, Yichen Shen, and Yi Yang.
A product of a collaboration between Max Tegmark's and Marin Soljacic's group. 
Originally created 04/24/2017











