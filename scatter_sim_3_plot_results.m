data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

values = [];
myspects = [];
%33.8_32.3_36.3_35.2_38.9
r1_1 = 31.2;
r2_1 = 37.5;
r3_1 = 67.7;
%r4_1 = 35.2;
%r5_1 = 38.9;
myname_1 = num2str(strcat('Desired :',num2str(r1_1),'-',num2str(r2_1),'-',num2str(r3_1)));
spect1 = scatter_0_generate_spectrum([r1_1,r2_1,r3_1])
r1_2 = 31.9;
r2_2 = 37.0;
r3_2 = 67.6;
%r4_2 = 30.7;
%r5_2 = 48.4;
myname_2 = num2str(strcat('NN :',num2str(r1_2),'-',num2str(r2_2),'-',num2str(r3_2)));
spect2 = scatter_0_generate_spectrum([r1_2,r2_2,r3_2])
r1_3 = 49.9;
r2_3 = 30.26;
r3_3 = 30.26;
%r4_3 = 30.38;
%r5_3 = 38.32;
myname_3 = num2str(strcat('MatLab :',num2str(r1_3),'-',num2str(r2_3),'-',num2str(r3_3)));
spect3 = scatter_0_generate_spectrum([r1_3,r2_3,r3_3])
hold on
%plot(lambda,[spect1(1:1:end,1)-spect2(1:1:end,1),spect1(1:1:end,1)-spect3(1:1:end,1)]);%spect(1:5:501,1));
plot(lambda,[spect1(1:1:end,1),spect2(1:1:end,1)]);%spect(1:5:501,1));
%plot(lambda,spect1(1:1:end,1));
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
%title('Residuals');
title('Inverse-Design');
%legend('14/25/24/38 after 2 hours','14/25/24/38 after 6 hours');
legend(myname_1,myname_2,myname_3);
%legend('Desired for 49/23/22/12/21','NN begin training','NN for reveresed 48/21/18/10/27');

