function sigma = spherical_cs(k,l,a,omega,eps)
%SPHERICAL_CS Single-channel cross section of a spherical multi-layer particle.
%   SPHERICAL_CS(k,l,a,omega,eps) returns a N-by-2 matrix containing the
%   partial scattering cross section in first column and partial absorption
%   cross section in second column.
%
%   The input k is an integer specifiying the polarization. It can be 1
%   (TE polarization) or 2 (TM polarization).
%
%   The input l is an integer specifying the angular momentum. It can be
%   1, 2, 3, ...
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

M = spherical_TM2(k,l,a,omega,eps);
tmp = M(:,1)./M(:,2);
R = (tmp - 1i)./(tmp + 1i);
coef = (2*l+1)*pi/2./omega.^2./eps(:,end);
sigma = repmat(coef,1,2).*[abs(1-R).^2 1-abs(R).^2];
