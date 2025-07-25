function [alfa,beta] = intersectionRatio(Ellipsoid1,Ellipsoid2)
%intersectionRatio Evaluates if two allipsoids need to be merged int oa
%single cluster.
%   This function evaluates if 2 interse4cted ellipsoids are able to be
%   merged into a single cluster. To do so, first the intersection is
%   computed, then de volume of each ellipsoid is calculated.
% The ratio between each ellipsoid and the intersection ios returned as a
% parameter.
E1 = ellipsoidConvertion(Ellipsoid1);
E2 = ellipsoidConvertion(Ellipsoid2);
Einter = intersection_ia(E1,E2);
%We obtain the shapoe of the matrix
Qi = parameters(Einter);
Q1 = parameters(E1);
Q2 = parameters(E2);
%Compute the determinant
Vi = det(Qi);
V1 = det(Q1);
V2 = det(Q2);
alfa = Vi/V1;
beta = Vi/V2;
end

