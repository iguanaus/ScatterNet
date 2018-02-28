function y = sbesselj_d(m,x)
y = (m*sbesselj(m-1,x)-(m+1)*sbesselj(m+1,x))/(2*m+1);