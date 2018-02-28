function y = product(a,b)
% take the product of two sets of 2-by-2 matrices written in 1-by-4 format, where the two sets of matrices are
% A = [ [a(:,1) a(:,3)]; [a(:,2) a(:,4)] ];
% B = [ [b(:,1) b(:,3)]; [b(:,2) b(:,4)] ];
y = [a(:,1).*b(:,1)+a(:,3).*b(:,2),a(:,2).*b(:,1)+a(:,4).*b(:,2),a(:,1).*b(:,3)+a(:,3).*b(:,4),a(:,2).*b(:,3)+a(:,4).*b(:,4)];
