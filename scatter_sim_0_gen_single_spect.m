%THis runs the specrum with a silver core fixed.
%spect(1:5:501,1)./(3*lambda.lambda)*2*pi
function spectrum = scatter_0_generate_spectrum(r)
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
data = load('data.mat');
%eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
%eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

eps_silica = 2.04*ones(length(omega), 1);
%Correcting for the lossy nature. 06/13/17 - John
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);
% load data on epsilon

% test case one: 40-nm-radius silver sphere in water
%eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_water];
%eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_water];

eps = [];
for i = 1:length(r)
	if mod(i,2) == 1
        %eps = [eps eps_gold];
		eps = [eps eps_silica];
    else
        %eps = [eps eps_silver];
        eps = [eps eps_tio2];
	end
end
eps = [eps eps_water];

order = 25;
if length(r) ==2 || length(r) == 3
	order = 4;
end
if length(r)  == 4 || length(r) == 5
	order = 9;
end
if length(r) == 6 || length(r) == 7
	order = 12;
end
if length(r) == 8 || length(r) == 9
	order = 15;
end
if length(r) == 10 || length(r) == 11
    order = 18;
end

spectrum = total_cs(r,omega,eps,order)/(pi*sum(r)^2);
%spectrum = total_cs(r,omega,eps,9);