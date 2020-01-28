 function rslt = prediction(Image, W1, W2, W3, bias_L1, bias_L2, bias_L3)

     % Normalize the input image
    X = double(Image)./255;

    m = size(X, 1);

    %% Perform forward propogation
    a1 = [(ones(m, 1)*bias_L1) X];

    z2 = a1 * W1';
    a2 = [(ones(m, 1)*bias_L2) sigmoid(z2)];

    z3 = a2 * W2';
    a3 = [(ones(m, 1)*bias_L3) sigmoid(z3)];

    z4 = a3 * W3';
    a4 = sigmoid(z4);

    strPred = "";

    [~, predictedLabel] = max(a4, [], 2); % https://www.mathworks.com/help/matlab/ref/max.html#d117e888441

    if predictedLabel == 10
        strPred = "airplane";
    elseif predictedLabel == 1 
         strPred = "automobile";	
    elseif predictedLabel == 2 
         strPred = "bird";	
    elseif predictedLabel == 3
         strPred = "cat";	
    elseif predictedLabel == 4 
         strPred = "deer";	
    elseif predictedLabel == 5 
         strPred = "dog";	
    elseif predictedLabel == 6 
         strPred = "frog";	
    elseif predictedLabel == 7 
         strPred = "horse";	
    elseif predictedLabel == 8 
         strPred = "ship";	
    elseif predictedLabel == 9 
         strPred = "truck";	
    end

    fprintf('\nPredicted Label is: %s\n', strPred);
 end