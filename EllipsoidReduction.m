function EllipsoidsNew = EllipsoidReduction(EllipsoidsOld,Threshold)
%   This Function reduces the number of ellipsoids by means of the
%   intersection of each ellipsoid. Here are all the functions neddesd to
%   apply the reduction of ellipsoids

% First I look for the number of ellipsoids at the end of the training
% process
NE = size(EllipsoidsOld,1);
% Then I compare the first ellipsoid againste all the other to si if there
% are any ohter that intersects the ellipsoid
IX = ones(1,NE);
i = 1;
while i < NE
    C = EllipsoidsOld(i,:);
    for j = i+1:NE
        Cb = EllipsoidsOld(j,:);
        if myIntersect(C,Cb)
            [A,B] = intersectionRatio(C,Cb);
            if(A>Threshold)||(B>Threshold)
                IX(j) = 0;
                Ep = EllipsoidSum(C,Cb);
                Epop = sumOptimization(Ep,C,Cb);
                [q,Q]=parameters(Epop);
                EllipsoidsOld{i,1} = q;
                EllipsoidsOld{i,2} = inv(Q);
                EllipsoidsOld{i,3} = eye(2);
                EllipsoidsOld{i,4} = [];
                C = EllipsoidsOld(i,:);
            end
        end
    end
    cont = 1;
    for k = 1:NE
        if IX(k) == 1
            EllipsoidsBeta{cont,1} = EllipsoidsOld{k,1};
            EllipsoidsBeta{cont,2} = EllipsoidsOld{k,2};
            EllipsoidsBeta{cont,3} = EllipsoidsOld{k,3};
            EllipsoidsBeta{cont,4} = EllipsoidsOld{k,4};
            cont = cont + 1;
        end
    end
    NE = sum(IX);
    IX = ones(1,NE);
    i=i+1;
    EllipsoidsOld=EllipsoidsBeta;
    EllipsoidsBeta={};
end
%Evaluate the intersection degree. If si greater than a threshold, merge
%ellipsoids
EllipsoidsNew = EllipsoidsOld;
end

