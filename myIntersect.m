function Chi = myIntersect(E1,E2)
    C1 = ellipsoid(E1{1},round(E1{3}*inv(E1{2})*E1{3}',5));
    C2 = ellipsoid(E2{1},round(E2{3}*inv(E2{2})*E2{3}',5));
    Chi = intersect(C1,C2);
end