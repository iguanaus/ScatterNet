% This file generates a single spectrum to compare/perform inverse design with.

addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

lambda = linspace(400, 800, 401)';

omega = 2*pi./lambda;
low_bound = 30;
up_bound = 70;

r1 = round(rand*(up_bound-low_bound)+low_bound,1);
r2 = round(rand*(up_bound-low_bound)+low_bound,1);
r3 = round(rand*(up_bound-low_bound)+low_bound,1);
r4 = round(rand*(up_bound-low_bound)+low_bound,1);
r5 = round(rand*(up_bound-low_bound)+low_bound,1);
r6 = round(rand*(up_bound-low_bound)+low_bound,1);
r7 = round(rand*(up_bound-low_bound)+low_bound,1);
r8 = round(rand*(up_bound-low_bound)+low_bound,1);
r9 = round(rand*(up_bound-low_bound)+low_bound,1);
r10 = round(rand*(up_bound-low_bound)+low_bound,1);
% Manually set them if you would like

%r1 = 44.1; 
%r2 = 63.2;
%r3 = 53.4;
%r1 = 62.6
%r2 = 66.2
%r3 = 35.1
%r4 = 66.5
%r5 = 55.3
%r6 = 33.9
%r7 = 41.1
%r8 = 51.9
my_r = [r1,r2]

spect = scatter_sim_0_gen_single_spect(my_r);

myspects = [spect(1:2:401,1)];
values = [[] ; [my_r]];
plot(lambda(1:2:401),myspects);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Spectrum Example');
%Save into the results folder. 
myname = strcat('results/2_layer_tio2/test_',strjoin(string(my_r),'_'))
csvwrite(strcat(myname,'.csv'),myspects);
csvwrite(strcat(myname,'_val.csv'),values);





