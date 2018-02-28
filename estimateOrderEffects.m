% Path to the Matlab functions
% This file estimates the order effects of the various particles presented in the paper. 
addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

% Wavelength of interest: 300 nm to 800 nm
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

% load data on epsilon
data = load('data.mat');
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);
eps_silica = 2.04*ones(length(omega), 1);
eps_water  = 1.77*ones(length(omega), 1);

eps_silica = 2.04*ones(length(omega), 1);
%eps_tio2 = 8.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
plot(eps_tio2)
val = 500.0;
f = 1.0;
wo = 2*pi./val;
gamma = .01;
ep_no = 1.85;
eps_jagg = ep_no + f*wo*wo./(wo*wo-omega.*omega-i.*omega*gamma*wo);

% test case one: 40-nm-radius silver sphere in water
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_water];

a = [70,70,70,70,70,70,70,70,70,70];
cs_loworder = total_cs(a,omega,eps,18);
cs_highorder = total_cs(a,omega,eps,40);
dif = (cs_loworder(1:1:401,1)-cs_highorder(1:1:401,1))./cs_loworder(1:1:401,1)*100.0;
plot(lambda, [dif]);
spect = total_cs(a,omega,eps,18);
spect2= total_cs(a,omega,eps,40);
%spect3= total_cs(a,omega,eps,5);
%spect4= total_cs(a,omega,eps,7);
%spect5= total_cs(a,omega,eps,10);%
%plot(lambda, [spect(1:1:401,1),spect2(1:1:401,1)]);
legend('3','25');
xlabel('Wavelength (nm)');
%ylabel('\sigma/\pi r^2');
ylabel('Percent Dif');
title('Scattering of 240nm radi Versus Increasing Angular Order');
