clear all
close all


%loading the data



%load('PruebaG2.mat')

%load('PruebaE.mat')

%load('LowDim.mat')

for kk = 1:100
load('PruebaKMeans.mat')


[f0,~] = size(data);

aux = data;
clear data
data.X = aux;
clear aux


%parameters
param.c = 2;
param.m=2;
param.e=1e0;
param.vis=0;
param.val=1;
%normalization
data=clust_normalize(data,'range');

%result = FCMclust(data,param);
% Esto se secomenta para mas de dos clusters

% Debo hacer una función que me permita realizar una asignación de
% pertenencia a cada uno de los clusters para poder encontrar mas de dos
% cluster


result = FCMclust(data,param);
param.c=result.data.f;

result = GGclust(data,param); 
result = validity(result,data,param);

result.data.f = real(result.data.f);

%plot(data.X(:,1),data.X(:,2),'b.',result.cluster.v(:,1),result.cluster.v(:,2),'ro');
%hold on

%draw contour-map

new.X=data.X;
eval=clusteval(new,result,param);
%result.validity

% Computing the precision


[~,Idx] = max(result.data.f');
N = length(Idx);
Acc = 0;
for i = 1:N
    if i <=1024 && Idx(i) ==2
        Acc = Acc + 1;
    end
    if i >1024 && Idx(i) ==1
        Acc = Acc + 1;
    end
end


disp('Accuracy')
disp(100*(Acc/N))

end
