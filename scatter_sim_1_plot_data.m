data = load('data.mat');
lambda = linspace(400, 800, 401)';

values = [];
myspects = [];

for v1=[30 50 70];
    for v2=[30 50 70];
        for v3=[30 50 70];
            for v4=[30];
                for v5=[30];
                        spect = scatter_0_generate_spectrum([v1,v2,v3]);
                        %spect(1:1:501,1)./(3*lambda.*lambda)*2*pi
                        myspects = [myspects spect(1:1:401,1)];%./(3*lambda.*lambda)*2*pi];
                        myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3)));
                        values = [values , string(myname)];
                end
            end
        end
    end
end
%figure('Color',[0.8 0.8 0.8]);
plot(lambda(1:1:401),myspects);%spect(1:5:501,1));
%set(gca,'Color',[0.0 0.0 0.0]);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Train Data Set');
legend(values)
%csvwrite('test_large_fixed_five_temp.csv',myspects);
%csvwrite('test_large_fixed_five_val_temp.csv',values);
