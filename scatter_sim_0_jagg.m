%THis runs the specrum with a silver core fixed.
%spect(1:5:501,1)./(3*lambda.lambda)*2*pi
function spectrum = scatter_0_generate_spectrum_jagg(r,val)
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
data = load('data.mat');
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);

f = 1.0;
wo = 2*pi./val;
gamma = .01;
ep_no = 1.85;
eps_jagg = ep_no + f*wo*wo./(wo*wo-omega.*omega-i.*omega*gamma*wo);
eps_water  = 1.77*ones(length(omega), 1);

% 3 layer, in water.
eps = [eps_silver eps_silica eps_jagg eps_water];

% Tested with accuracy. Found 8 order to be optimal
order = 8;

spectrum = total_cs(r,omega,eps,order)/(pi*sum(r)^2);