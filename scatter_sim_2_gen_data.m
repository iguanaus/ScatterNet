values = [];
myspects = [];

low_bound = 30;
up_bound = 70;
num_iteration = 20000;
n = 0;

num_layers = 6;

tic
while n < num_iteration
  n = n + 1;
  r = [];
  for i = 1:num_layers
    r1 = round(rand*(up_bound-low_bound)+low_bound,1);
    r = [r r1];
  end
  spect = scatter_0_generate_spectrum(r);%,r6,r7,r8]);
  myspects = [myspects spect(1:2:401,1)];
  values = [values ; r];%,r6,r7,r8]];
  if rem(n, 100) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc

csvwrite(strcat('data/',num2str(num_layers),'_layer_tio2_fixed_06_21_2.csv'),myspects);
csvwrite(strcat('data/',num2str(num_layers),'_layer_tio2_fixed_06_21_2_val.csv'),values);