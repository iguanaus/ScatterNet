% This file is namely for display purposes. It will graph the outputs to see what all the training examples look like.
% Note that this file is fairly hand-crafted. To change it you will have to modify the list of v1,v2,v3, etc.
% This will generate all permutations of v1....v5. Be warned: This will
% grow quickly and is not meant to graph substantial amounts of data. 

addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';


data = load('data.mat');
lambda = linspace(400, 800, 401)';
%order = 3 %Override the order (if you want)

values = [];
myspects = [];
%Ranges for most inner thickness


for v1=[30,70]; 
    % Ranges for next inner thickness
    for v2=[30,70];
        % Ranges for next layer
        for v3=[30,70];
            for v4=[30,70];
                for v5=[30,70];
                    x=[v1 ,v2,v3,v4,v5]
                    spect = scatter_sim_0_gen_single_spect(x)*(pi*sum(x)^2)./(3*lambda.*lambda)*2*pi;
                    myspects = [myspects spect(1:1:401,1)];
                    myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3),'--',num2str(v4),'--',num2str(v5)));
                    values = [values , string(myname)];
                        
                end
            end
        end
    end
end
%mylist=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hold on
plot(lambda(1:1:401),myspects);
%plot(lambda(1:2:501),mylist)

%set(gca,'Color',[0.0 0.0 0.0]);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Data Set Samples');
legend(values)
hold off