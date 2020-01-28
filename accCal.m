function rslt = accCal(Theta1, Theta2, Theta3, bias_L1, bias_L2, bias_L3, X, y)
X = (double(X));

m = size(X, 1);

% Perform forward propogation
a1 = [(ones(m, 1)*bias_L1) X];
    
z2 = a1 * Theta1';
a2 = [(ones(m, 1)*bias_L2) sigmoid(z2)];
    
z3 = a2 * Theta2';
a3 = [(ones(m, 1)*bias_L3) sigmoid(z3)];
    
z4 = a3 * Theta3';
a4 = sigmoid(z4);

% This step is explained in 2.4.1 of the attached project report
[~, predictVector] = max(a4, [], 2); % https://www.mathworks.com/help/matlab/ref/max.html#d117e888441

%Calculate Accuracy
rslt = mean(double(predictVector == y))* 100;


end
