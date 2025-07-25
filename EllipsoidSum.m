function Ep = EllipsoidSum(E1,E2)
    q1 = E1{1};
    q2 = E2{1};
    ele = (q1-q2)/sqrt((q1-q2)'*(q1-q2));
    Q1 = round(inv(E1{3}*E1{2}*E1{3}'),5);
    Q2 = round(inv(E2{3}*E2{2}*E2{3}'),5);
    RIGHT = (sqrt(ele'*(Q1*ele))+sqrt(ele'*(Q2*ele)));
    LEFT = (Q1/sqrt(ele'*(Q1*ele))+Q2/sqrt(ele'*(Q2*ele)));
    Qp = RIGHT*LEFT;
    Unit = det(Q1)+det(Q2);
    qp = (det(Q1)*q1+det(Q2)*q2)/Unit;
    Ep{1,:}=qp;
    Ep{2,:}=inv(Qp); % I take the inverse of the ellipsoid for the toolbox
    %to be able of plotting the result :)
    Ep{3,:}=eye(2);
    Ep{4,:}=[];
end
