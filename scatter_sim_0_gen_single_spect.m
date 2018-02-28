% This generates the data for a single spectrum of a particle composed of alternating shells of silica and TiO2. The r list gives the list of thicknesses in order from the center outward. The orders have been estimated based on how much loss was percieved as you go to higher orders. 
function spectrum = scatter_0_generate_spectrum(r)
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
data = load('data.mat');
%eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
%eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);
% load data on epsilon

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