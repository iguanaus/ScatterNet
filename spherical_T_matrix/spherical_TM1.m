function M = spherical_TM1(k,l,a,omega,eps1,eps2)

k1 = omega.*sqrt(eps1);
k2 = omega.*sqrt(eps2);
x1 = k1*a;
x2 = k2*a;

j1 = sbesselj(l,x1);
j1_d = sbesselj_d(l,x1).*x1 + j1;
y1 = sbessely(l,x1);
y1_d = sbessely_d(l,x1).*x1 + y1;
j2 = sbesselj(l,x2);
j2_d = sbesselj_d(l,x2).*x2 + j2;
y2 = sbessely(l,x2);
y2_d = sbessely_d(l,x2).*x2 + y2;

if k == 1 % TE
  M = product([y2_d,-j2_d,-y2,j2],[j1,j1_d,y1,y1_d]);
else % TM
  M = product([eps1.*y2_d,-eps1.*j2_d,-y2,j2],[j1,eps2.*j1_d,y1,eps2.*y1_d]);
end
