These are all the data files used in the paper.
Note all these were generated using the "ScatterNet_Matlab" directory here in the repository. Be cautious of the order of the harmonics - as the particle get more layers, more orders must be added to compensate for more modes. 

Directory:
  Data for n layer particle with alternating silica/TiO2 shells:
    n_layer_tio2.csv
    n_layer_tio2_val.csv
    The _val file indicates what the values of the thickneses are (in nanometers). The other file - 2_layer_tio2.csv - indicates the values of the spectrum for each corresponding particle. That is, the first line in 2_layer_tio2 corresponds to the first line in 2_layer_tio2_val.
    The 2 layer particle has 30k records. The 3,4,5,6,7 layer has 40k. The 8 layer has 50k. 
  Data for 3 layer jaggregate particle:
    jagg_layer_tio2.csv
    jagg_layer_tio2_val.csv
    Same format as above. The _val file indicates the thickness of the metallic silver core, dielectric layer of silica, and outside layer of the J-Aggregate dye respectively. The last number is the tuned resonnance for the J-Aggregate dye. 

