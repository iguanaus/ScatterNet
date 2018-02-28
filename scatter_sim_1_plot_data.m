% This file is namely for display purposes. It will graph the outputs to see what all the training examples look like.
% Note that this file is fairly hand-crafted. To change it you will have to modify the list of v1,v2,v3, etc.

data = load('data.mat');
lambda = linspace(400, 800, 401)';

values = [];
myspects = [];

for v1=[30 50 70];
    for v2=[30 50 70];
        for v3=[30 50 70];
            for v4=[30];
                for v5=[30];
                        spect = scatter_0_generate_spectrum([v1,v2,v3,v4,v5]);
                        myspects = [myspects spect(1:1:401,1)];
                        myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3),'--',num2str(v4),'--',num2str(v5)));
                        values = [values , string(myname)];
                end
            end
        end
    end
end
plot(lambda(1:1:401),myspects);
%set(gca,'Color',[0.0 0.0 0.0]);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Train Data Set');
legend(values)
