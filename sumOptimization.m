function Ellipsoid = sumOptimization(Esum,E1,E2)
%SUMOPTIMIZATION This function optimize the sum of two ellipsoids 
% As the sum of several ellipsoids increases the size of the cluster, we
% need to reduce the are od the ellipsoid in order to preserve a good
% quality cluster.
EsumB = ellipsoidConvertion(Esum);
E1B = ellipsoidConvertion(E1);
E2B = ellipsoidConvertion(E2);
Inter1 = intersection_ia(EsumB,E1B);
Inter2 = intersection_ia(EsumB,E2B);

while (E1B <= Inter1)&&(E2B <= Inter2)
    Esum{2}=Esum{2}*1.1;
    EsumB = ellipsoidConvertion(Esum);
    Inter1 = intersection_ia(EsumB,E1B);
    Inter2 = intersection_ia(EsumB,E2B);
end
Ellipsoid = ellipsoidConvertion(Esum);
end

