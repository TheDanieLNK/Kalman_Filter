% Daniel Nkemelu
function KalmanFilterWithoutAC

clear;
clc;

% declaring vectors and matrices for the true model
TF_k = [0.96, 0.04; 0, 1];
TH_k = [1, 0;0, 1];
TSW = [0.03 0; 0 0.02];
TSV = [1.5 0; 0 1.5];
Tb_k = [0; 0];

% declaring vectors and matrices for the Kalman model; for estimation
F_k = [0.95, 0.05; 0, 1];
H_k = [1, 0;0, 1];
SW = [0.04 0; 0 0.01];
SV = [4 0; 0 1];
b_k = [0; 0];

N = 300;                    % number of time steps
M = length(F_k);            % gives us the n-dimension of our Kalman Filter

X = zeros(M, N);            % create a 2x300 matrix for our internal and external temperatures
Xh = X;                     % create a similar matrix as X for our internal temperature estimates

P = zeros(2,2,N);           % our matrix of covariance matrices
X(:,1) = [23; 20];          % initial temperature state value
Xh(:,1) = [25, 25];         % initial estimate state value
P(:,:,1) = [1 0; 0 1];      % initial covariance matrix

Z = zeros(2, N);            % create a matrix for our measurements
control = 0;                % initial control state
controls = zeros(N,1);      % create a matrix of the control states
controls(1) = control;

% initial values for our markov chain
alpha = 0.25;                  
beta = 0.5;
markovMatrix = [1-alpha, alpha; beta, 1-beta];
state = 1;

% we iterate N time steps
for n = 1:N-1
    % store sine functions in variables
    funcSin1 = [0; 0.14 * sin((2 * pi * n) / 300)];
    funcSin2 = [0; 0.1 * sin((2 * pi * n) / 300)];
    
    % generate random noise
    W = [normrnd(0, 0.03); normrnd(0, 0.02)];
    V = [normrnd(0, 1.5); normrnd(0, 1.5)];
    
    % get temperature values from true model
    X(:,n+1) = TF_k * X(:,n) + funcSin1 + Tb_k * control + W;
    % get sensor measurement
    Z(:,n) = TH_k * X(:,n+1) + V;    
     
    % entering state 1 means that time is advanced, so we predict
    if(state == 1)
        % prediction
        Xh(:,n+1) = F_k * Xh(:,n) + funcSin2 + b_k * control;
        % calculate and store covariance matrix
        S = F_k * P(:,:,n) * F_k' + TSW;
        P(:,:,n+1) = S;
        % determine next state
        state = discrete(markovMatrix(state,:));
    
    % entering state 2 means there is a new measurement and there should be
    % a fusion step
    else
        % if there are multiple measurements in a row, they should be
        % processed in order and only the last estimate plotted
        
        while state == 2
            % get value of the control
            u_k = control;
            
            % generate random noise
            w_k = [normrnd(0, 0.03); normrnd(0, 0.02)];
            v_k = [normrnd(0,1.5); normrnd(0,1.5)];
            
            % compute the control equations for both the true and Kalman
            % model
            Tbk_uk  = Tb_k * u_k;
            bk_uk = b_k * u_k;
            
            % get temperature values from true model
            X(:,n+1) = TF_k* X(:,n) + funcSin1 + Tbk_uk + w_k;
            
            % compute new sensor measurement
            Z(:,n) = H_k * X(:,n+1) + v_k;
            
            % use system equations to compute new covariance matrix
            S = F_k * P(:,:,n) * F_k' + SW;
            K = S * H_k' * (H_k * S * H_k' + SV) ^(-1); 
            P(:,:,n+1) = (eye(2)-K*H_k)*S;

            % prediction
            Xh(:,n+1) = (F_k - K*H_k *F_k)* Xh(:,n) - K * H_k * funcSin2 + funcSin2 - K * H_k * bk_uk + bk_uk + K*Z(:,n);
            % determine next state
            state = discrete(markovMatrix(state,:));
        end
    end
    
    % compute the probability that the internal temperature is more than 28 degrees
    % if this exceeds 10%, then in the next step let uk = 1 otherwise uk = 0.
    estimatedInternalTemp = 0.95 * Xh(1,n) + 0.05 * Xh(2,n);
    estimatedVar = P(1,1,n+1);
    
    if qfunc((28 - estimatedInternalTemp)/sqrt(estimatedVar)) > 0.1
        control = 1;
    else
        control = 0;
    end   
    
    % store value of the controls
    controls(n+1) = control;
end

% plotting values
plot(X(1,:), 'b')                             % true model internal temperature
hold on

ylim([0, 40]);
legend('Actual External Temperature', ...
        'Actual Internal Temperature with AC', ...
        'Actual Internal Temperature without AC');
xlabel('Time steps')
ylabel('Temperature')
title('Kalman Filter showing 300 steps')

end
