% Prueba Elipses

close all
clear all
rng(1);
%% Los eignevalores se claulan con base en la longitud de los semi ejes de las elipses
%L1 = 1/3.5^2;
dd = 10;
L1 = 1/dd^2;
% L2 = 1/3.5^2;
L2 = 1/dd^2;
% Dimension
dim = 2;
%Se carga la base de datos
mu1 = [-15,10];

Theta1=0.01;

R1 = [cos(Theta1) -sin(Theta1);
    sin(Theta1) cos(Theta1)];

t = 10;
n = 200;

Sigma1 = [8 -2; -2 2];


data = mvnrnd(mu1,Sigma1,n);


nn = size(data,1);
%Se establecen los parametros de la primera elipse 
q = mean(data)';%Centro
Q=[L1 0;0 L2];%Matriz de los ejes
theta=0;%Angulo de rotacion
R = [cosd(theta) -sind(theta);
sind(theta) cosd(theta)];%Matriz de rotacion
%% Se almacenan los parametros de la primera ellipse 
E{1,1} = q;
E{1,2} = Q;
E{1,3} = R;
E{1,4} = q;

%% Clasificador
Classifier = my_classifier();

figure(1)
plotellipsoid(E(1,:))
hold on
grid on
axis equal

%%
for i = 1:nn%1351%Numero de datos
    
%     j = randi(1351);
    j = randi(nn);
    nd = data(j,:);
    new_c = 0;%bander para la creacion de nuevos clusters
    dist = [];%distancia de evaluacion de pertenencia a una ellipsoide
    distB = [];%distancai para la evaluacion de la ellipse mas cercana
    cont = 1; %solo se crea un nuevo cluster si no se encuentra en ningun cluster
    NE = size(E,1);
    for k = 1:NE
        M = ellipsoidConstraint(E(k,:),nd);
        if M <= 1
            plot(nd(1),nd(2),'b.')
            E{k,4} = [E{k,4} , nd'];
%             nd = 0;
%             break
        else
            new_c = new_c + 1;
%             distB(cont) = optimizationL(nd,E(k,:));
%             cont = cont + 1;
        end
    end
    if new_c == NE
        for k = 1:NE
            distB(k) = optimizationL(nd,E(k,:));
        if distB(k)>dd
            distB(k) = dd;
        end
        end
        E(NE + 1,:) = createEllipsoid(nd,min(distB));
        plotellipsoid(E(NE + 1,:))
    end
end
figure
hold on
axis equal
grid on
E = ellipsoidsDep(E,50);%50
%% Entrenamiento
NE = size(E,1);
figure
hold on
axis equal
grid on
E(1,:)=BP_ellipsoids(E(1,:),500,1);%500

E(1,4) = E(1,1);

figure
hold on 
axis equal
grid on
plotellipsoid(E(1,:))
plot(data(:,1),data(:,2),'r.')


%Classificacion



%% FUNCIONES
%Eliminar ellipses basadas en SSE
function [Ellipsoids_new, Classifier] = eliminateClusterSSE(Ellipsoids_old,SSE, threshold,Classifier)
    nClusters = size(Ellipsoids_old,1);
    Ellipsoids_new = {};
    cont = 1;
    for i = 1:nClusters
        if(SSE(i) < threshold)
            Ellipsoids_new{cont,1} = Ellipsoids_old{i,1};
            Ellipsoids_new{cont,2} = Ellipsoids_old{i,2};
            Ellipsoids_new{cont,3} = Ellipsoids_old{i,3};
            Ellipsoids_new{cont,4} = Ellipsoids_old{i,4};
            plotellipsoid(Ellipsoids_new(cont,:))
            cont = cont + 1;
        end
    end
    NE = size(Ellipsoids_new,1);
    for k = 1:NE
        Classifier.datos_clases{k} = Ellipsoids_new{k,4};
        Classifier.Ellipsoids{1,k}=Ellipsoids_new{k,1};
        Classifier.Ellipsoids{2,k}=Ellipsoids_new{k,2};
        Classifier.Ellipsoids{3,k}=Ellipsoids_new{k,3};
        Classifier.clases{k} = k;
    end
end

% Imprimir una elipse
function plotellipsoid(E)
q = E{1,1};
Q = E{1,2};
R = E{1,3};
ellipsoidObject = ellipsoid(q,round(R*inv(Q)*R',10));
plot(ellipsoidObject)
end
% Evaluar la pertenencia a una elipsoide
function membership=ellipsoidConstraint(E,instance)
    q = E{1,1};
    Q = E{1,2};
    R = E{1,3};
    d = instance-q';
    membership = d*R*Q*R'*d';
end
% Funcion para encontrar la distancia entre un punto y los borde de una
% ellipse
function distance = optimizationL(pointX,Ellipsoid)
    q = Ellipsoid{1,1};
    Q = Ellipsoid{1,2};
    R = Ellipsoid{1,3};
    pointX = pointX - q';
    dim = size(pointX,2);
    fun=@(alp)pointX*inv(eye(2)+alp*R*Q*R')*R*Q*R'*inv(eye(2)+alp*R*Q*R')'*pointX' - 1;
    alp = fzero(fun,[0 10000]);
    Xmin = inv(eye(dim) + alp*R*Q*R')*pointX';
    distance = sqrt((pointX - Xmin')*(pointX - Xmin')');
end
% Funcion para crear una nueva elipse
function newEllipsoid = createEllipsoid(center,axislength)
    dim = size(center,2);
    newEllipsoid{1,1} = center';
    newEllipsoid{1,2} = eye(dim)*(1/(axislength)^2);
    newEllipsoid{1,3} = eye(dim);
    newEllipsoid{1,4} = center';
end
% Funcion para depurar las ellipsoides
function Ellipsoids_new = ellipsoidsDep(Ellipsoids_old,threshold)
    Ellipsoids_new = {};
    cont = 1;
    N_E = size(Ellipsoids_old,1);
    for i = 1:N_E
        instances = Ellipsoids_old{i,4};
        if size(instances,2) > threshold
            Ellipsoids_new{cont,1} = Ellipsoids_old{i,1};
            Ellipsoids_new{cont,2} = Ellipsoids_old{i,2};
            Ellipsoids_new{cont,3} = Ellipsoids_old{i,3};
            Ellipsoids_new{cont,4} = Ellipsoids_old{i,4};
            plotellipsoid(Ellipsoids_new(cont,:))
            plot(instances(1,:),instances(2,:),'b.')
            cont = cont + 1;
        end
    end
end
% Función para obtener desviacion estandar
function standardeviation = stdE(Ellipsoid,plane)
    q = Ellipsoid{1,1};
    Q = Ellipsoid{1,2};
    R = Ellipsoid{1,3};
    Eig_vec = R(:,plane);
    Data = Ellipsoid{1,4};
    [~, instances] = size(Data);
    Shift = -q'*Eig_vec;
    sum=0;
    for i = 1:instances
%         sum = sum + ((Coefficients*Data(:,i)+Shift)/sqrt(Coefficients*Coefficients'))^2;
        sum = sum + (P_plane(Eig_vec,Shift,Data(:,i)))^2;
    end
    standardeviation = sqrt(sum/(instances-1));
end
% función pra el calculo de la deistancia de un punto a un plano
function distance = P_plane(Eigenvector,shift,Data)
    distance = (Eigenvector'*Data+shift)/sqrt(Eigenvector'*Eigenvector);
end
% Funcion de entrenamiento
function Ellipsoid_new = BP_ellipsoids(Ellipsoid,iterations,plane)
Q = Ellipsoid{1,2};
    R = Ellipsoid{1,3};
    Data = Ellipsoid{1,4};
    q = mean(Data,2);
    Ellipsoid{1,1} = q;
    %%Evaluar desviacion estajndfar
    desvEstX1 = stdE(Ellipsoid,1)
    desvEstX2 = stdE(Ellipsoid,2)
    Eig_vec = R(:,1);
    [dim, instances] = size(Data);
%     theta = atand(-Eig_vec(2,1)/Eig_vec(1,1));
%     theta = atan2d(-Eig_vec(2,1),Eig_vec(1,1));
    theta = atan2d(Eig_vec(2,1),Eig_vec(1,1));
    Shift = -q'*Eig_vec;
    Loss = [];
    for i = 1:iterations
        Sum = 0;
        desvEst = stdE(Ellipsoid,1);
        DAccum = 0;
%         for j = 1:instances
%             d = P_plane(Eig_vec,Shift,Data(:,j));
%             Sum = Sum + 0.5*d*(1/instances)*(desvEst^(-1))*((sqrt(tand(theta)^2+1)*(Data(1,j)*secd(theta)^2))...
%                 -0.5*((tand(theta)*Data(1,j) + Data(2,j) + Shift/Eig_vec(1))*(tand(theta)^2+1)^(-1/2)*(2*tand(theta)*secd(theta)^2)))...
%                 /(tand(theta)^2+1);
%             WatchSum(i,j) = Sum;
%             WatchDist(i,j) = d;
%             DAccum = DAccum + abs(d);
%         end
        for j = 1:instances
            d = P_plane(Eig_vec,Shift,Data(:,j));
            Sum = Sum + 0.5*d*(1/instances)*(desvEst^(-1))*((sqrt(tand(theta)^2+1)*(Data(2,j)*secd(theta)^2))...
                -0.5*((tand(theta)*Data(2,j) + Data(1,j) + Shift/Eig_vec(1))*(tand(theta)^2+1)^(-1/2)*(2*tand(theta)*secd(theta)^2)))...
                /(tand(theta)^2+1);
            WatchSum(i,j) = Sum;
            WatchDist(i,j) = d;
            DAccum = DAccum + abs(d);
        end
        theta = theta + Sum;
        WatchSumFinal(i)=Sum;
        WatchTheta(i)=theta;
        WatchDev1(i)=stdE(Ellipsoid,1);
        WatchDev2(i)=stdE(Ellipsoid,2);
        WatchDistFinal(i)=DAccum;
        R = [cosd(theta) -sind(theta);
        sind(theta) cosd(theta)];
        Ellipsoid{1,3} = R;
        Eig_vec = R(:,1);
        Shift = -q'*Eig_vec;
        Loss(:,i) = [stdE(Ellipsoid,1);theta];
    end
 


    for i = 1:dim
        Q(i,i) = 1/(2.5*stdE(Ellipsoid,i))^2;
    end
    Ellipsoid_new{1,1} = q;
    Ellipsoid_new{1,2} = Q;
    Ellipsoid_new{1,3} = R;
    Ellipsoid_new{1,4} = Data;
    plot(Data(1,:),Data(2,:),'b.')
end
