% Social Ski-Driver (SSD) optimization algorithm
% More details about the algorithm are in [please cite the original paper (below)]
% Alaa Tharwat, Thomas Gabel, "Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm"
% Neural Computing and Applications, pp. 1-14, 2019
% Tharwat, A. & Gabel, T. Neural Comput & Applic (2019). https://doi.org/10.1007/s00521-019-04159-z
clc
clear all

Max_iterations=50;  % Maximum Number of Iterations
correction_factor = 2.0; % Correction factor
inertia = 1.0; % Ineritia Coeffecient
swarm_size = 20; % Number of particles
LB=[-5.12 -5.12]; % Lower Boundaries
UB=[5.12 5.12];   % Upper Boundaries
xrange=UB(1)-LB(1);
yrange=UB(2)-LB(2);
Dim=size(UB,2);
Runs=20;
for r=1:Runs
    %     clear swarm
    swarm(:, 1, 1)=rand(1,swarm_size)*xrange+LB(1);
    swarm(:, 1, 2)=rand(1,swarm_size)*yrange+LB(2);
    
    % Initial best value so far
    swarm(:, 4, 1) = inf;          

    % Initial velocity
    swarm(:, 2, :) = 0;             
    for iter = 1 : Max_iterations    
        % Calculating fitness value for all particles
        for i = 1 : swarm_size
            swarm(i, 1, 1) = swarm(i, 1, 1) + swarm(i, 2, 1);     %update x position
            swarm(i, 1, 2) = swarm(i, 1, 2) + swarm(i, 2, 2);     %update y position        
            
            % Update Position Bounds
            swarm(i, 1, 1) = max(swarm(i, 1, 1),LB(1));
            swarm(i, 1, 1) = min(swarm(i, 1, 1),UB(1));
            swarm(i, 1, 2) = max(swarm(i, 1, 2),LB(2));
            swarm(i, 1, 2) = min(swarm(i, 1, 2),UB(2));
            
            % The fitness function (Rastrigin)
            x = swarm(i, 1, 1);
            y = swarm(i, 1, 2);
            Fval(i)=20+x.^2-10.*cos(2.*3.14159.*x)+y.^2-10.*cos(2.*3.14159.*y); % evaluate the obejective function
            if Fval(i) < swarm(i, 4, 1)                 % if new position is better
                swarm(i, 3, 1) = swarm(i, 1, 1);    % Update the position of the first dimension
                swarm(i, 3, 2) = swarm(i, 1, 2);    % Update the position of the second dimension
                swarm(i, 4, 1) = Fval(i);              % Update best value
            end
        end
        a=2-iter*((2)/Max_iterations); % a decreases linearly fron 2 to 0
        
        % Search for the global best solution
        [temp, gbest] = min(swarm(:, 4, 1));        % global best position
         
        % Calculate the mean of the best three solutions in each dimension
        [aa, ind]=sort(Fval);
        for i=1:Dim
            M(i)=mean(swarm(ind(1:3), 1, i));    
        end
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        A1=2*a*r1-a; 
        C1=2*r2; 
        
        % Updating velocity vectors
        for i = 1 : swarm_size
            if (rand<0.5)  %  Use Sine function to move
                for j=1:Dim
                    swarm(i, 2, j) = (a)*sin(rand)*(swarm(i, 3, j) - swarm(i, 1, j))+(2-a)*sin(rand)*(M(j)- swarm(i, 1, j));   %x velocity component
                end
            else        %  Use Coine function to move
                for j=1:Dim
                    swarm(i, 2, j) = (a)*cos(rand)*(swarm(i, 3, j) - swarm(i, 1, j))+(2-a)*cos(rand)*(M(j)- swarm(i, 1, j));   %x velocity component
                end
            end
        end
        
        % Store the best fitness valuye in the convergence curve
        ConvergenceCurve(iter,1)=swarm(gbest,4,1);
        disp(['Iterations No. ' int2str(iter) ' , the best fitness value is ' num2str(gbest)]);
    end
    HistoryConvCurve{r}=ConvergenceCurve;
    clear ConvergenceCurve
end

for i= 1: Runs
    plot(HistoryConvCurve{i}) ; hold on   
end
title('Convergence Curve, different runs')
xlabel('Iterations')
ylabel('Fitness Value')
