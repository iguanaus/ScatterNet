% This file does the work. It will genreate all the data files given the bounds of the particle, the number of examples to generate, and the number of layers in the particle. It will save the data into the data/ folder. 
%
addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

values = [];
myspects = [];

low_bound = 30;
up_bound = 70;
num_iteration = 1000;
n = 0;

num_layers = 3;

tic
while n < num_iteration
  n = n + 1;
  r = [];
  for i = 1:num_layers
    r1 = round(rand*(up_bound-low_bound)+low_bound,1);
    r = [r r1];
  end
  spect = scatter_sim_0_gen_single_spect(r);
  myspects = [myspects spect(1:2:401,1)]; % Use 200 points in the spectrum.
  values = [values ; r];
  if rem(n, 100) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc

csvwrite(strcat('data/',num2str(num_layers),'_test_layer_tio2.csv'),myspects);
csvwrite(strcat('data/',num2str(num_layers),'_test_layer_tio2_val.csv'),values);