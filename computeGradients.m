function [W1_gradients, W2_gradients, W3_gradients, Bias_L2_gradient, Bias_L3_gradient, J_train] = computeGradients(Theta1,Theta2,Theta3, ...
                                   bias_L1, bias_L2, bias_L3, ...
                                   X_train, y_train, lambda)

%One Hot Encoding
y_oht = (double(y_train)./[1 2 3 4 5 6 7 8 9 10]) == 1;

% Weights gradients matrices initilization to accumalate gradients
W1_gradients = zeros(size(Theta1));
W2_gradients = zeros(size(Theta2));
W3_gradients = zeros(size(Theta3));

% Bias gradients initilization
Bias_L2_gradient = 0;
Bias_L3_gradient = 0;

X_train = double(X_train);
m = size(X_train, 1);

%% Forward propogation 
% Layer 1, input layer
a1 = [(ones(m, 1)*bias_L1) X_train];

%Layer 2
z2 = a1 * Theta1';
a2 = [(ones(m, 1)*bias_L2) sigmoid(z2)];
    
%Layer 3
z3 = a2 * Theta2';
a3 = [(ones(m, 1)*bias_L3) sigmoid(z3)];
    
%Layer 4
z4 = a3 * Theta3';
a4 = sigmoid(z4);

%Cost Function
J_train = -1*sum(sum(y_oht.*log(a4)+(1-y_oht) .* log(1-a4(y_oht))));
J_train = (J_train/m) + lambda*(sum(sum(Theta1(:,2:end).^2))...
                            +sum(sum(Theta2(:,2:end).^2))...
                            +sum(sum(Theta3(:,2:end).^2)))/(2*m);

%% Backpropogation
for trainExample = 1:m
    % Layer 4, output layer
    z4b = z4(trainExample,:)';
    a4b = a4(trainExample,:)';
    
    % Layer 3
    z3b = z3(trainExample,:)';
    a3b = a3(trainExample,:)';
    
    % Layer 2
    z2b = z2(trainExample,:)';
    a2b = a2(trainExample,:)';
    
    % Layer 1, Input layer
    a1b = a1(trainExample,:)';
    
    weights_L4_delta = a4b - y_oht(trainExample,:)';
%     weights_L4_deltaA = a4' - y_oht';
%     delta_L3A = Theta3'*(weights_L4_delta).*[ones(size(z3,1),1)  sigDerivative(z3)]';

    delta_L3 = (Theta3'*weights_L4_delta).*[1;sigDerivative(z3b)];
    bias_L3_delta = delta_L3(1:1);
    weights_L3_delta = delta_L3(2:end);
    
    delta_L2 = (Theta2'*weights_L3_delta).*[1;sigDerivative(z2b)];
    bias_L2_delta = delta_L2(1:1);
    weights_L2_delta = delta_L2(2:end);

    W1_gradients = W1_gradients + weights_L2_delta*a1b';
	W2_gradients = W2_gradients + weights_L3_delta*a2b';
    W3_gradients = W3_gradients + weights_L4_delta*a3b';
    
    Bias_L2_gradient = Bias_L2_gradient + bias_L2_delta;
    Bias_L3_gradient = Bias_L3_gradient + bias_L3_delta;
end

% Apply regularization
W1_gradients = W1_gradients + lambda*([zeros(size(Theta1,1),1),Theta1(:,2:end)]); %to not change j=0 wali column
W2_gradients = W2_gradients + lambda*([zeros(size(Theta2,1),1),Theta2(:,2:end)]);
W3_gradients = W3_gradients + lambda*([zeros(size(Theta3,1),1),Theta3(:,2:end)]);

end
