addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';
data = load('data.mat');
lambdaLimit = 400
lambda = linspace(lambdaLimit, 800, (800-lambdaLimit)+1)';
omega = 2*pi./lambda;
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
%eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);

%epsIn = [eps_silver eps_silica eps_silver eps_silica eps_silver eps_water]
%orderLimit = 10

epsIn = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_water]
orderLimit = 0


myspects = []
x = [30 30 70 70 70]
%With the radius normalizing it.
%spect = scatter_sim_0_gen_single_spect(x,lambdaLimit,epsIn,orderLimit)*(pi*sum(x)^2)./(3*lambda.*lambda)*2*pi;
%Without
spect = scatter_sim_0_gen_single_spect(x,lambdaLimit,epsIn,orderLimit)./(3*lambda.*lambda)*2*pi;
myspects = [myspects spect(1:1:length(lambda),1)];
myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3),'--',num2str(v4),'--',num2str(v5)));
values = [values , string(myname)];

hold on

%The CSV file specifying where to scatter.
mylist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
plot(lambda(1:1:length(lambda)),myspects);
length(lambda)
plot(lambda(1:2:401),mylist)

%set(gca,'Color',[0.0 0.0 0.0]);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Data Set Samples');
legend(values)
hold off

