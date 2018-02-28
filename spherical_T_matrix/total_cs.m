function sigma = total_cs(a,omega,eps,order)
%TOTAL_CS Total cross section of a spherical multi-layer particle.
%   TOTAL_CS(k,l,a,omega,eps) returns a N-by-2 matrix containing the total
%   scattering cross section in first column and total absorption cross section
%   in second column, where 'total' here means summing over TE and TM for
%   l = 1,2,3.
%
%   The input a is a 1-by-K row vector specifying the thickness for each layer
%   of the particle, starting from the inner-most layer. So a(1) is the radius
%   of the core, a(2) is the thickness of the first coating (NOT its radius),
%   etc.
%
%   The input omega is a N-by-1 column vector specifying the frequencies at
%   which to evaluate the cross sections.
%
%   The input eps is a N-by-(K+1) matrix specifying the relative permittivity,
%   such that eps(:,1) are for the core at the frequencies given by omega,
%   eps(:,2) for the first coating, etc, and eps(:,K+1) for the medium where
%   the particle sits in.
%
%   Unit convention: suppose the input a is in unit of nm, then the returned
%   cross sections are in unit of nm^2, and the input omega is in unit of
%   2*pi/lambda, where lambda is free-space wavelength in unit of nm. The same
%   goes when a is in some other unit of length.

%   2012 Wenjun Qiu @ MIT

if nargin < 4
	order = 3;
end

sigma = 0;
for l = 1:order
	sigma = sigma + spherical_cs(1,l,a,omega,eps) + spherical_cs(2,l,a,omega,eps);
end

