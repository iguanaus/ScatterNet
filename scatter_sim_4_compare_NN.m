% This program will take in a trained network (given in results), and then compare its inverse design to Matlab's inverse design - using the same underlying optimization algorithm, but the NN gets the analytical gradient and NN computations versus classical computations. This was used to generate Figure 5 and Figure 6.

addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);


%%%%% =========================
%%%%% HyperParameters to Choose

%  Manually pick your layers
% Add in however many layers are necesary.
%% 2 Layer example
eps = [eps_silica eps_tio2 eps_water]
% 8 Layer example
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_water]
% Where the file is to optimize from. 
network_file = 'results/2_layer_tio2/'
test_file_name = 'test_47.5_45.3.csv'
% Number of optimizations
numberOpts = 50
%True for NN, false for Matlab
useNN = false
%Use numerical gradient from NN. 
useGradient = useNN
%If this is true, it will generate new start params. 
useStoredStartParams = false


%Similarly, make sure this matches the dimension above. In this case, 2. 
%NOTE: TO HAVE A FAIR COMPARISON, YOU SHOULD COMMENT THIS OUT. This would then require that the matlab and NN have the same starting positions, otherwise the results are skweed. 
if useStoredStartParams == false 
	all_start_params = []
	eps_size = size(eps);
	eps_size = eps_size(2);
	for i = 1:numberOpts
		start_params = []
	    for j = 1:(eps_size-1)
	        j
	    	start_params = [start_params ; round(rand*40+30,1)];
	    end
	    all_start_params = [all_start_params , start_params];
	end
	order = 25;
	if length(start_params) ==2 || length(start_params) == 3
		order = 4;
	end
	if length(start_params)  == 4 || length(start_params) == 5
		order = 9;
	end
	if length(start_params) == 6 || length(start_params) == 7
		order = 12;
	end
	if length(start_params) == 8 || length(start_params) == 9
		order = 15;
	end
	if length(start_params) == 10 || length(start_params) == 11
	    order = 18;
	end
end

wgts = cell(0); 
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load(strcat(network_file,'w_',num2str(i),'.txt')));
    bias{i+1} = load(strcat(network_file,'b_',num2str(i),'.txt'));
end

filename = strcat(network_file,test_file_name);
myspect = csvread(filename);
myspect = myspect(1:1:201,1); %If bad dimensions, change this. 
dim = size(wgts);

%Get the spect file. 
filename2 = strcat(network_file,'spec_file_0.txt');
myspect2 = csvread(filename2);
means = transpose(myspect2(1,:));
stds = transpose(myspect2(2,:));

threshold = 0.1 % Based on our own choice. Generally something small.
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',threshold,'SpecifyObjectiveGradient',useGradient);

% Optimize with either the NN or Matlab
% Uncomment to use matlab
if useNN == false
	cost_func = @(x)cost_function_math(x,wgts,bias,dim(2),myspect,omega,eps,order);
end
if useNN == true
	cost_func = @(x)cost_function_nn(x,wgts,bias,dim(2),myspect,means,stds);
end


%This is the actual computation
totconv = 0;
tottime = 0;
tot_avg_conv = 0;
for i = 1:numberOpts
	start_params = all_start_params(:,i);
	[mytime, convergence] = run_opt(start_params,cost_func,options);
	tot_avg_conv = tot_avg_conv + convergence;
	mytime;
	if (convergence < threshold)
		convergence = 1.0;
	else
		convergence = 0.0;
	end
	tot_avg_conv
	convergence;
	tottime = tottime + mytime;
	totconv = totconv + convergence;
	i
	totconv
	tottime  
end
disp('=======================');
disp('Optimizations complete.');
disp(['Percentage of time converged: ', num2str(totconv/numberOpts*100.0),'%'])
disp(['Average time taken: ' , num2str(tottime/numberOpts)]);
disp(['Average Error at end of convergence: ', num2str(tot_avg_conv/numberOpts)]);
disp('=======================');

% Optimization using interior-point

function [time,convergence,x] = run_opt(start_params,cost_func,options)
A = [];
b = [];
Aeq = [];
beq = [];
lb = 30 * ones(1,5);
ub = 70 * ones(1,5);
nonlcon=[];
x0 = start_params;
tic;
[x,fval,exitflag,output] = fmincon(cost_func,x0,A,b,Aeq,beq,lb,ub,nonlcon, options);
x
time = toc;
convergence = fval;
end


function gradient = Jacobian2Gradient(dfdx,out,expectedOut)
gradient = transpose(dfdx)*(out-expectedOut);
end


function [cost,gradient] = cost_function_math(r,weights,biases,depth,spectToCompare,omega,eps,order)
[layer, Jacobian] = NN(weights,biases,r);
spectrum_run = scatter_sim_0_gen_spect_faster(r,omega,eps,order);
spectrum_new = spectrum_run(1:2:401,1);
cost = sum((spectrum_new-spectToCompare).^2);
gradient = Jacobian2Gradient(Jacobian,spectrum_new,spectToCompare)*2.0;
end

function [cost,gradient] = cost_function_nn(x,weights,biases,depth,spectCompare,xmeans,xstds)
%Normalize the input.
x = (x-xmeans)./xstds;
[layer, Jacobian] = NN(weights,biases,x);
cost = sum((spectCompare-layer).^2);
gradient = Jacobian2Gradient(Jacobian,layer,spectCompare)*2.0;
end

function spectrum = scatter_sim_0_gen_spect_faster(r,omega,eps,order)
spectrum = total_cs(r,omega,eps,order)/(pi*sum(r)^2);
end

