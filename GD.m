function [Theta1, Theta2, Theta3, bias_L2, bias_L3, J_train] = GD(Theta1,Theta2,Theta3, ...
                                   bias_L1, bias_L2, bias_L3, ...
                                   X, y, lambda)
alpha = 0.9; %Learning rate
epochs = 320; % Epochs
J_train = []; % To log values of Cost Function J
m = size(X, 1); % number of training examples
% Implementation of mini batch gradient descent
    for iter = 1: epochs

        for batchNum = 1: 8 
         a = 1 + (5000*(batchNum-1));
         b = 5000*batchNum;
         X_train = X(a:b,:);
         y_train = y(a:b,:);

         % Forward and back propogation is done in the computeGradients.m file
         [W1_gradients, W2_gradients, W3_gradients,...
                Bias2_gradient, Bias3_gradient, J] = computeGradients(Theta1,...
                Theta2,Theta3, bias_L1, bias_L2, bias_L3, ...
                                       X_train, y_train, lambda);

            % Theta Update Rule
            Theta1 = Theta1 - (alpha*W1_gradients/m); 
            Theta2 = Theta2 - (alpha*W2_gradients/m); 
            Theta3 = Theta3 - (alpha*W3_gradients/m); 

            % Bias Update Rule
            bias_L2 = bias_L2 - (alpha*Bias2_gradient/m); 
            bias_L3 = bias_L3 - (alpha*Bias3_gradient/m);

            % Log the cost func values
            J_train(iter) = J;

             fprintf('Current iter: %i,  batchNum: %i, CostFuncVal: %f\n', iter, batchNum,J_train(iter));

             if isinf(J_train(iter))
                fprintf('Cost Function Value INCREASING. Please re-adjust learning rate parameter...\n'); 
                fprintf('Current Learning Rate is %f...\n', alpha);
                pause;
             end
    % save('try6_23_dec_weights', 'Theta1', 'Theta2', 'Theta3', 'bias_L2', 'bias_L3')
        end

    end
end