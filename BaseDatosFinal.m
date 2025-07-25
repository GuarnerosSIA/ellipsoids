close all
clear all

mu1 = [-10,-10];
mu2 = [10,10];
mu3 = [10,-10];

Theta1=-0.001;
Theta2=-0.002;
Theta3=-0.003;

R1 = [cos(Theta1) -sin(Theta1);
    sin(Theta1) cos(Theta1)];
R2 = [cos(Theta2) -sin(Theta2);
    sin(Theta2) cos(Theta2)];
R3 = [cos(Theta3) -sin(Theta3);
    sin(Theta3) cos(Theta3)];

t = 1000;
n = 200;

Sigma1 = [2 -1; -1 2];
Sigma2 = [2 -1; -1 2];
Sigma3 = [2 1; 1 3];

A=zeros(n,2,t);

A(:,:,1) = mvnrnd(mu1,Sigma1,n);
B(:,:,1) = mvnrnd(mu2,Sigma2,n);
C(:,:,1) = mvnrnd(mu3,Sigma3,n);

data(:,:,1)=[A(:,:,1);B(:,:,1);C(:,:,1)];

for i = 2:t
    A(:,:,i) = (A(:,:,i-1))*R1;
    B(:,:,i) = (B(:,:,i-1))*R2;
    C(:,:,i) = (C(:,:,i-1))*R3;
    data(:,:,i)=[A(:,:,i);B(:,:,i);C(:,:,i)];
end

hold on
axis equal

T = 0:t-1;

plot3(reshape(A(:,1,:),[],t),reshape(A(:,2,:),[],t),T*1,'+r')
plot3(reshape(B(:,1,:),[],t),reshape(B(:,2,:),[],t),T*1,'+g')
plot3(reshape(C(:,1,:),[],t),reshape(C(:,2,:),[],t),T*1,'+b')
figure
plot3(reshape(data(:,1,:),[],t),reshape(data(:,2,:),[],t),T*1,'+b')


