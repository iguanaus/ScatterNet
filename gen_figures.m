addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

%plotFigure3()
%plotFigure4b()
plotFigure4a()

function [] = plotFigure3()
	data = load('data.mat');
	lambda = linspace(400, 800, 401)';
	omega = 2*pi./lambda;

	values = [];
	myspects = [];
	desired_r = [48 45 61 62 38 50 48 56]
	desired_n = strcat(strjoin(string(desired_r))," Desired")
	spect1 = scatter_sim_0_gen_single_spect(desired_r)

	matlab_r = [49 54 54 54 45 54 53 51]
	matlab_n = strcat(strjoin(string(matlab_r))," Numerical")
	spect2 = scatter_sim_0_gen_single_spect(matlab_r)

	NN_r = [49 45 59 62 38 50 48 56]
	NN_n = strcat(strjoin(string(NN_r))," NN")
	spect3 = scatter_sim_0_gen_single_spect(NN_r)

	hold on
	plot(lambda,[spect1(1:1:end,1),spect2(1:1:end,1),spect3(1:1:end,1)]);
	xlabel('Wavelength (nm)');
	ylabel('\sigma/\pi r^2');
	title('Inverse-Design');
	legend(desired_n,matlab_n,NN_n);
end

function [] = plotFigure4b()
	data = load('data.mat');
	lambdaLimit = 300
	lambda = linspace(lambdaLimit, 800, (800-lambdaLimit)+1)';
	omega = 2*pi./lambda;
	eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
	%eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

	eps_silica = 2.04*ones(length(omega), 1);
	my_lam = lambda./1000;
	eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
	eps_water  = 1.77*ones(length(omega), 1);
	epsIn = [eps_silver eps_silica eps_silver eps_silica eps_silver eps_water]
	orderLimit = 10
	myspects = []
	x = [10 47 27 36 10]
	spect = scatter_sim_0_gen_single_spect(x,lambdaLimit,epsIn,orderLimit)*(pi*sum(x)^2)./(3*lambda.*lambda)*2*pi;
	myspects = [myspects spect(1:1:length(lambda),1)];
	x_n = strcat(strjoin(string(x))," Optimal")
	values = ['Desired',x_n];

	hold on
	area([425,540],[max(spect(1:2:501,1)),max(spect(1:2:501,1))],'EdgeColor','none')
	alpha(.2)
	plot(lambda(1:1:length(lambda)),myspects);
	%plot(lambda(1:2:501),mylist)

	%set(gca,'Color',[0.0 0.0 0.0]);
	xlabel('Wavelength (nm)');
	ylabel('\sigma/\pi r^2');
	title('Data Set Samples');
	legend(values)
	hold off
end

function [] = plotFigure4a()
	data = load('data.mat');
	lambdaLimit = 350
	lambda = linspace(lambdaLimit, 800, (800-lambdaLimit)+1)';
	omega = 2*pi./lambda;
	eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
	%eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

	eps_silica = 2.04*ones(length(omega), 1);
	my_lam = lambda./1000;
	eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
	eps_water  = 1.77*ones(length(omega), 1);
	epsIn = []%[eps_silver eps_silica eps_silver eps_silica eps_silver eps_water]
	orderLimit = 5
	myspects = []
	x = [70 30 34.9 50.3]
	%x = [18.5 59.5 32.5 51.5 9.5]
	spect = scatter_sim_0_gen_single_spect(x,lambdaLimit,epsIn,orderLimit)*(pi*sum(x)^2)./(3*lambda.*lambda)*2*pi;
	myspects = [myspects spect(1:1:length(lambda),1)];
	x_n = strcat(strjoin(string(x))," Optimal")
	values = ['Desired',x_n];

	hold on
	area([425,450],[max(spect(1:2:lambdaLimit,1)),max(spect(1:2:lambdaLimit,1))],'EdgeColor','none')
	alpha(.2)
	plot(lambda(1:1:length(lambda)),myspects);
	%plot(lambda(1:2:501),mylist)

	%set(gca,'Color',[0.0 0.0 0.0]);
	xlabel('Wavelength (nm)');
	ylabel('\sigma/\pi r^2');
	title('Data Set Samples');
	legend(values)
	hold off
end