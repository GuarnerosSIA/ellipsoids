%% Definitive ellipsoid Classification
close all
clear all
%% Hyperparameters of the program
% Analisis de sensibilidad. En este caso se van a realizar cambios en los
% valores que definen las elipses para poder determinar que tanto cambian
% los valores de clasificación si se realiza un cambio en la precision del
% algoritmo. La semilla sera fija para que se pueda evaluar el cambio
% rng(1);

% Tamaño de las elipsioides iniciales
dd = 3;
% Numero de datos dentro de la elipse
in_data = 60;
% Numero asociado a la desviación estandar
gamma_par = 3;
L1 = 1/dd^2;
L2 = 1/dd^2;
% Xi_+
xi_p = 0.2222;
% Dimension
dim = 2;
%The database is loaded
Mparam = 3;
% load BaseFinalExtended.mat
% load('PruebaKMeans.mat')
% load('PruebaG2.mat')
load('PruebaE.mat')
% load('LowDimCorrected.mat')
% data=data_aux;

% data=data/10;


% Clases
% Labels = {0,150,300,450,600,750,900,1050,1200,1350};
% Labels = {0,1000,2000};
% Labels = {0,1024,2048};
Labels = {0,2000,4000,6000,6100,6200,6300,6400,6500};
results_XXX = zeros(10,2);
% Evaluation number
tic
for xxx = 1:1

ClassificationSecure = [];

%The size of the database is measure to generate the first aleatory pattern
%to be introduced to the ellipsoid algorithm
nn = size(data,1);
j = randi(nn);
%Default ellipsoid parameters: Center, Eigenvalues and rotation angle
q = data(j,:)';
Q=[L1 0;0 L2];
theta=0;
R = [cosd(theta) -sind(theta);
sind(theta) cosd(theta)];
%% The parameters are stored in an structure
E{1,1} = q;
E{1,2} = Q;
E{1,3} = R;
E{1,4} = q;
%%%%%%%%%%%
figure
plotellipsoid(E(1,:))
hold on
grid on
axis equal
%%%%%%%%%%%
% tic
for MM = 1:Mparam
%This cycle is for the retarining of the ellipsoids after the refinement
%process is executed
    for i = 1:nn
    %In this cycle, the patterns are presented to the algorithm in an
    %aleatory order. Then it is evaluated if the new pattern belongs to
    %some ellipsoid. Notice that pattern can be in different ellipsoids,
    %thus, each pattern is evalñuated against all the existen ellipsoids
        j = randi(nn);
        nd = data(j,:);
        new_c = 0;%Flag to evaluate if the pattern is oputside every ellipsoid
        dist = [];%
        distB = [];%Variable for the identification of the closes ellipsoid
        NE = size(E,1);
        for k = 1:NE
        %This cycle seeks the memebership of the pattern to any ellipsoid
            M = ellipsoidConstraint(E(k,:),nd);
            if M <= 1
%%                plot(nd(1),nd(2),'b.','MarkerSize',20)
                E{k,4} = [E{k,4} , nd'];%The pattern is assigned to the ellipsoid
            else
                new_c = new_c + 1;
            end
        end
    % Ithe pattern is outside every ellipsoids, a new ellipsoid is created
    % based on the distance to the closest ellipsoid, in this way, the
    % ellipsoids at furst are separated
        if new_c == NE
            for k = 1:NE
                distB(k) = optimizationL(nd,E(k,:));
                if distB(k)>dd
                    distB(k) = dd;
                end
            end
%             if min(distB)>0.5%%Este fi lo cabao de añadir
            E(NE + 1,:) = createEllipsoid(nd,min(distB));
%%            plotellipsoid(E(NE + 1,:))
%             end
        end
    end
%%    figure
%%    hold on
%%    axis equal
%%    grid on
    %The First filter eliminates the ellipsoids that conatins few patterns
    %inside them.
    E = ellipsoidsDep(E,in_data);
%% Entrenamiento
    NE = size(E,1);
%%    figure
%%    hold on
%%    axis equal
%%    grid on
    % For each ellipsoid, the backpropagation algorithm is applied in order
    % to optimize the standard deviation. The center is set as the mean od
    % the patterns inside the llipsoid
    for k = 1:NE
        E(k,:)=BP_ellipsoids(E(k,:),100,gamma_par);%500
%%        plotellipsoid(E(k,:))
    end
%%    figure
%%    hold on 
%%    axis equal
%%    grid on
    %The Last depuration is about the ellipsoid quality. If the ellipsoid
    %presents a poor quality given by the ellipsoid equation, then the
    %ellipsoid is eliminated.
    E = qualityEDep(E,xi_p,0.0);
end
% toc
%% Ellipsoid Reduction
E = EllipsoidReduction(E,0.25);%Estaba en 0.2
%% Clasificación de los clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clasificación Segura
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For each of the labels, the patterns are asigned a number depending on the
%closest ellipsoid. Then if the distance to the closest ellipsoid is less
%than 1, the pattern is inside the ellipsoid and is asigned the
%corresponding index.

Membership=[];
DataArrayS = cell(1,size(E,1));
Ix = zeros(1,size(data,1));
Ix_true = zeros(1,size(data,1));
ClassificationSecure = zeros(size(Labels,2)-1,size(E,1));

for j = 1:size(Labels,2)-1
    for k = Labels{j}+1:Labels{j+1}
        Ix_true(k) = j;
        u = data(k,:);
        for i = 1:(size(E,1))
            Cluster = E(i,:);
            Membership(i)= ellipsoidConstraint(Cluster,u);
        end
        [Value,Index] = min(Membership);
        if Value <= 1
            Ix(k) = Index;
        end
    end
end

[conf_matrix,labels_true_unique,labels_pred_unique] = build_confusion_matrix(Ix_true,Ix);
cost_matrix = -conf_matrix;
[matching, uR, uC] = matchpairs(cost_matrix, 1e6);


total_matched = 0;

for k = 1:size(matching, 1)
    i = matching(k,1);
    j = matching(k,2);
    
    if ismember(0,Ix)
        ttt = 1;
    else
        ttt = 0;
    end

    if i > ttt
        total_matched = total_matched + conf_matrix(i, j);
    end
end

accuracy = total_matched / length(Ix_true); 

disp('FinalAccuracy:')
results_XXX(xxx,1) = accuracy;
disp(accuracy)
IndexSum = [];
% Start plotting properties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
for i = 1:(size(E,1))
    Labelts = strcat('$C_',num2str(i),'$');
    plot(data(Ix==i,1),data(Ix==i,2),'MarkerSize',20,'Marker','.',...
    'LineStyle','none');
    for j = 1:size(Labels,2)-1
        IndexSum(i,j) = sum(Ix(Labels{j}+1:Labels{j+1})==i);
        %For each label, the sum of the index is compute
    end
end
plot(data(Ix==0,1),data(Ix==0,2),'MarkerSize',20,'Marker','.',...
    'LineStyle','none','Color',[0 0 0]);
% Create ylabel
ylabel('$x_2$','FontSize',30,'Interpreter','latex');
% Create xlabel
xlabel('$x_1$','FontSize',30,'Interpreter','latex');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'DataAspectRatio',[1 1 1],'FontSize',20,'PlotBoxAspectRatio',...
    [434 342.3 342.3],'TickLabelInterpreter','latex');
% End Plotting properties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adding the statistics of the clasification accuracy




%Evaluating the accuracy. 
TOTAL=0;
Redundant=[];
for i = 1:size(Labels,2)-1
    Total = -Labels{i}+Labels{i+1};
    [Accuracy,IIxx] = max(IndexSum(:,i));    
    TOTAL = TOTAL + Accuracy;
    Accuracy = Accuracy/Total*100;
%     disp('Accuracy')
%     disp(Accuracy)
    if(i <= size(E,1))
        plotellipsoid(E(i,:))
    end
end

final_accuracy = TOTAL/size(data,1);

% disp(final_accuracy)

% Verify if the ellipsoids number is equal to two
if size(E) > 2
    disp(size(E))
    disp('THERE IS SOMETHING BAD IN HERE')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clasificacion por distancia 2 B|
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

Membership=[];
IxB = zeros(1,size(data,1));
IxB_true = zeros(1,size(data,1));

for j = 1:size(Labels,2)-1
    for k = Labels{j}+1:Labels{j+1}
        u = data(k,:);
        for i = 1:(size(E,1))
            Cluster = E(i,:);
            Membership(i)= ellipsoidConstraint(Cluster,u);
        end
        [~,Index] = min(Membership);
        IxB(k)=Index;
        IxB_true(k) = j;
    end
end

[conf_matrix,labels_true_unique,labels_pred_unique] = build_confusion_matrix(IxB_true,IxB);
cost_matrix = -conf_matrix;
[matching, uR, uC] = matchpairs(cost_matrix, 1e6);

total_matched = 0;

for k = 1:size(matching, 1)
    i = matching(k,1);
    j = matching(k,2);
    total_matched = total_matched + conf_matrix(i, j);
end

accuracy = total_matched / length(Ix_true); 
disp('FinalAccuracy:')
disp(accuracy)
results_XXX(xxx,2) = accuracy;

IndexSumB = [];
for i = 1:(size(E,1))
    plot(data(IxB==i,1),data(IxB==i,2),'MarkerSize',20,'Marker','.',...
    'LineStyle','none')
    for j = 1:size(Labels,2)-1
        IndexSumB(i,j) = sum(IxB(Labels{j}+1:Labels{j+1})==i);
    end
end
TOTALB=0;
for i = 1:size(Labels,2)-1
    Total = -Labels{i}+Labels{i+1};
    Accuracy = max(IndexSumB(:,i));
    TOTALB = TOTALB + Accuracy;
    Accuracy = Accuracy/Total*100;
%     disp('Accuracy')
%     disp(Accuracy)
    if(i <= size(E,1))
        plotellipsoid(E(i,:))
    end
end

final_accuracyB = TOTALB/size(data,1);
% disp('FinalAccuracy:')
% disp(final_accuracyB)


% Create ylabel
ylabel('$x_2$','FontSize',30,'Interpreter','latex');
% Create xlabel
xlabel('$x_1$','FontSize',30,'Interpreter','latex');
grid(axes1,'on');
% Set the remaining axes properties
set(axes1,'DataAspectRatio',[1 1 1],'FontSize',20,'PlotBoxAspectRatio',...
    [434 342.3 342.3],'TickLabelInterpreter','latex');


load('labels_ld.mat')
end
toc
%% FUNCIONES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotellipsoid(E)
%Plotting each ellipsoid
q = E{1,1};
Q = E{1,2};
R = E{1,3};
%% ellipsoidObject = ellipsoid(q,round(R*inv(Q)*R',10));
%% plot(ellipsoidObject,'r','LineWidth',5)
end
function membership=ellipsoidConstraint(E,instance)
%Take the internal parameters of an ellipsoid and applied the ellipsoid
%equation
    q = E{1,1};
    Q = E{1,2};
    R = E{1,3};
    d = instance-q';
    membership = d*R*Q*R'*d';
end
function distance = optimizationL(pointX,Ellipsoid)
%Optimization procedure for finding the distance between the boundary of an
%ellipsoid and a given point
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

function newEllipsoid = createEllipsoid(center,axislength)
%Create an ellipsoid based on a given distance
    dim = size(center,2);
    newEllipsoid{1,1} = center';
    newEllipsoid{1,2} = eye(dim)*(1/(axislength)^2);
    %print(dim)
    %print(axislength)
    newEllipsoid{1,2} = eye(dim)*(1/(axislength)^2);
    newEllipsoid{1,3} = eye(dim);
    newEllipsoid{1,4} = center';
end

function Ellipsoids_new = ellipsoidsDep(Ellipsoids_old,threshold)
% If an ellipsoid have less than athreshold of values inside them, the
% ellipsoid is eliminated. The remaining ellipsoids are save for the next
% step
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
%%            plotellipsoid(Ellipsoids_new(cont,:))
%%            plot(instances(1,:),instances(2,:),'b.','MarkerSize',20)
            cont = cont + 1;
        end
    end
end

function standardeviation = stdE(Ellipsoid,plane)
%This function computes the standar deviation of the ellipsoid along a
%given semi axis. The standar deviation is compute based on the
%eigenvectors given by the rotation matrix. Once we compute the
%eigenvector, we can calculate the diatnce, sum all the diustance and
%obtain the standar deviation algon a semiaxis
    q = Ellipsoid{1,1};
    Q = Ellipsoid{1,2};
    R = Ellipsoid{1,3};
    Eig_vec = R(:,plane);
    Data = Ellipsoid{1,4};
    [~, instances] = size(Data);
    Coefficients = Eig_vec';
    Shift = -q'*Eig_vec;
    sum=0;
    for i = 1:instances
        sum = sum + (P_plane(Eig_vec,Shift,Data(:,i)))^2;
    end
    standardeviation = sqrt(sum/(instances-1));
end
function distance = P_plane(Eigenvector,shift,Data)
    distance = (Eigenvector'*Data+shift)/sqrt(Eigenvector'*Eigenvector);
end
function Ellipsoid_new = BP_ellipsoids(Ellipsoid,iterations,gamma_par)
%This function first takes the parameter of the llipsoid
Q = Ellipsoid{1,2};
    R = Ellipsoid{1,3};
    Data = Ellipsoid{1,4};
    q = mean(Data,2);
    Ellipsoid{1,1} = q;
%Evaluates the standard deviation  of the each semiaxis, obtain the value
%of the angle theta
    desvEstX1 = stdE(Ellipsoid,1);
    desvEstX2 = stdE(Ellipsoid,2);
    Eig_vec = R(:,1);
    [dim, instances] = size(Data);
    theta = atan2d(Eig_vec(2,1),Eig_vec(1,1));
    Shift = -q'*Eig_vec;
    Loss = [];
    for i = 1:iterations
        %For each pattern the derivative with respect to the angle are
        %computed, if the optimization is to reduce the standar deviation
        %algon the first axis, the sign is negative, otherwise, the sign is
        %positive. The parameters of the ellipsoid are updated
        Sum = 0;
        desvEst = stdE(Ellipsoid,1);
        DAccum = 0;
        for j = 1:instances
            d = P_plane(Eig_vec,Shift,Data(:,j));
            Xdata = Data(:,j)-q;
            U = Xdata(1,1) + tand(theta)*Xdata(2,1);
            V = sqrt(tand(theta)^2 + 1);
            dU = Xdata(2,1)*secd(theta)^2;
            dV = (0.5*V^(-1))*(2*tand(theta)*secd(theta)^2);
            Sum = Sum + d*(1/instances)*(desvEst^(-1))*(V*dU-U*dV)/(V^2);
        end
        theta = theta + Sum;
        WatchSumFinal(i)=Sum;
        WatchTheta(i)=theta;
        WatchDev1(i)=stdE(Ellipsoid,1);
        WatchDev2(i)=stdE(Ellipsoid,2);
        R = [cosd(theta) -sind(theta);
        sind(theta) cosd(theta)];
        Ellipsoid{1,3} = R;
        Eig_vec = R(:,1);
        Shift = -q'*Eig_vec;
        Loss(:,i) = [stdE(Ellipsoid,1);theta];
    end
% The standar deviation of each axis is computed and the semi axis are
% updated to this value
    for i = 1:dim
        Q(i,i) = 1/(gamma_par*stdE(Ellipsoid,i))^2;
    end
    Ellipsoid_new{1,1} = q;
    Ellipsoid_new{1,2} = Q;
    Ellipsoid_new{1,3} = R;
    Ellipsoid_new{1,4} = Data;
%%    plot(Data(1,:),Data(2,:),'b.','MarkerSize',20)
end

function Ellipsoids_new = qualityEDep(Ellipsoids_old,ThresholdUp,ThresholdDown)
%If the quality of the ellipsoid is below a given threshold, the ellipsoid
%si preserved, otherwise is eliminated
    Ellipsoids_new = {};
    cont = 1; 
    N_E = size(Ellipsoids_old,1);
    for i = 1:N_E
        Q = ellipsoidQuality(Ellipsoids_old(i,:));
            disp(Q)
        if  Q < ThresholdUp && ThresholdDown < Q
            Ellipsoids_new{cont,1} = Ellipsoids_old{i,1};
            Ellipsoids_new{cont,2} = Ellipsoids_old{i,2};
            Ellipsoids_new{cont,3} = Ellipsoids_old{i,3};
            Ellipsoids_new{cont,4} = [];
            cont = cont + 1;
        end
    end
end

function Q = ellipsoidQuality(Ellipsoid)
%The quality of the ellipsoid is computed based on the ellipsoid equation
    Q = 0;
    Data = Ellipsoid{1,4};
    N = size(Data,2);
    for i = 1:N
        u = Data(:,i)';
        Q = Q + ellipsoidConstraint(Ellipsoid,u)/N;
    end
end


%%% Confussion matrix
function [conf_matrix, labels_true_unique, labels_pred_unique] = build_confusion_matrix(labels_true, labels_pred)
    labels_true_unique = unique(labels_true);
    labels_pred_unique = unique(labels_pred);
    n_true = length(labels_true_unique);
    n_pred = length(labels_pred_unique);
    conf_matrix = zeros(n_pred, n_true);

    for i = 1:n_pred
        for j = 1:n_true
            cluster_id = labels_pred_unique(i);
            true_id = labels_true_unique(j);
            conf_matrix(i,j) = sum(labels_pred == cluster_id & labels_true == true_id);
        end
    end
end


