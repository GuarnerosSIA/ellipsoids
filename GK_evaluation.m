clear all
close all

%loading the data

load('PruebaKMeans.mat')
%load('PruebaG2.mat')
%load('PruebaE.mat')
%load('LowDim.mat')

aux = data;
clear data
data.X = aux;
clear aux
for i = 1:100
%parameters
param.c=2;
param.m=2;
param.e=1e-6;
param.ro=ones(1,param.c);
param.val=1;
%normalization
data=clust_normalize(data,'range');

result = GKclust(data,param);
result = validity(result,data,param);

plot(data.X(:,1),data.X(:,2),'b.',result.cluster.v(:,1),result.cluster.v(:,2),'ro');
hold on
%draw contour-map
new.X=data.X;
eval=clusteval(new,result,param);
result.validity

[~,Idx] = max(result.data.f');
N = length(Idx);
Acc = 0;
for i = 1:N
    if i <= 1024 && Idx(i) ==2
        Acc = Acc + 1;
    end
    if i > 1024 && Idx(i) ==1
        Acc = Acc + 1;
    end
end

disp('Accuracy')
disp(100*(Acc/N))

end