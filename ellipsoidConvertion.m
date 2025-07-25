function ellipsoidToolbox = ellipsoidConvertion(Ellipsoid)
%EllipsoidConvertion Translation between my ellipsoids and the toolbox
%   First we extract the center, rotation matrix and the positive definite
%   matrix
q = Ellipsoid{1};
P = Ellipsoid{2};
R = Ellipsoid{3};
% Compute the ellipsoidal matrix
Q = round(R*inv(P)*R',10);
ellipsoidToolbox = ellipsoid(q,Q);
end

