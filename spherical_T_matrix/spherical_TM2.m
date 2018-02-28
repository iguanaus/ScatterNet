function M = spherical_TM2(k,l,a,omega,eps)

a = cumsum(a);
[N,K] = size(eps);
K = K - 1;
M = [ones(N,1),zeros(N,2),ones(N,1)];

for i = 1:K
    tmp = spherical_TM1(k,l,a(i),omega,eps(:,i),eps(:,i+1));
    M = product(tmp,M);
end