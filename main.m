%% Machine Learning Project 03
clear ; close all; clc

%% =========== Part 1: Loading Dataset =============
% % Load Training Data
fprintf('Loading Data ...\n')

% Path where CIFAR-10 dataset is stored
path = "C:\Users\Umer\Desktop\ML_proj3";

dataBatch1 = load( path + '\cifar-10-batches-mat\data_batch_1.mat'); % Load the training batch 01
dataBatch2 = load( path + '\cifar-10-batches-mat\data_batch_2.mat'); % Load the training batch 02
dataBatch3 = load( path + '\cifar-10-batches-mat\data_batch_3.mat'); % Load the training batch 03
dataBatch4 = load( path + '\cifar-10-batches-mat\data_batch_4.mat'); % Load the training batch 04
dataBatch5 = load( path + '\cifar-10-batches-mat\data_batch_5.mat'); % Load the training batch 05

test_batch1 = load( path + '\cifar-10-batches-mat\test_batch.mat'); % Load the test batch

% To combine all the training batches into a single Batch
% Dividing the feature vector by 255 to normalise the data
train_x = ([double(dataBatch1.data); double(dataBatch2.data); double(dataBatch3.data); double(dataBatch4.data); double(dataBatch5.data)])./255;
train_y = [dataBatch1.labels; dataBatch2.labels; dataBatch3.labels; dataBatch4.labels; dataBatch5.labels];

% The training batch now contains combine labels and the data
trainingData = [double(train_y), train_x];
% Spliting trainingData into Training Set and CV Set

rng('default'); %to ensure constant seed of random gen each time the code runs
[m,n] = size(trainingData) ;% get the size of data matrix
P = 0.80 ; %Spliting 80-20
rnd = randperm(m)  ; %Take the row number vector and randomize the row number in it

%Get the Training set
data_train = trainingData(rnd(1:round(P*m)),:) ; %get 80% of the data

%Get the CV set
data_cv = trainingData(rnd(round(P*m)+1:end),:) ; %get 20% of the data

X_train = data_train(:, 2:end); % get feature vectors 
y_train = data_train(:, 1); % get label vector

X_cv = data_cv(:, 2:end); % get feature vectors 
y_cv = data_cv(:, 1); % get label vector

X_test = double([test_batch1.data])./255; % Dividing the Test feature vector by 255 to normalise the data
y_test = [test_batch1.labels];

% Replacing '0' labels in the test/train/CV set with a label '10'
for i = 1: length(y_train)
    for j = 1: length(y_train(i))
        if (y_train(i,j) == 0)
            y_train(i,j) = 10;
        end
    end
end
for i = 1: length(y_test)
    for j = 1: length(y_test(i))
        if (y_test(i,j) == 0)
            y_test(i,j) = 10;
        end
    end
end
for i = 1: length(y_cv)
    for j = 1: length(y_cv(i))
        if (y_cv(i,j) == 0)
            y_cv(i,j) = 10;
        end
    end
end

% % MNIST DATASET
% MNIST = load('mnist.mat');
% 
% X_train = double(MNIST.trainX)/255;
% y_train = MNIST.trainY' ;
% 
% X_test = double(MNIST.testX)/255;
% y_test = MNIST.testY';
% 
% for index = 1: length(y_train)
%     if (y_train(index) == 0)
%         y_train(index) = 10;
%     end
% end
% 
% for index = 1: length(y_test)
%     if (y_test(index) == 0)
%         y_test(index) = 10;
%     end
% end
% 


fprintf('Program paused. Press enter to continue.\n');
pause;
%% =========== Part 2: Init Params  =============
input_layer_size  = size(X_train, 2);  % Number of neurons in the first layer ...
                                       %will be equal to the number of features in the feature vector
                                       % which is 3072
first_hidden_layer_size = 256;   %  hidden units
sec_hidden_layer_size = 64;   %  hidden units
output_layer_size = 10;          % 10 labels - from 1 to 10 (note that we have mapped "0" to label 10)

% Initializing Weights Matrices
fprintf('Initializing Weights Matrices and Bias Units ...\n')

init_W1 = initW(input_layer_size, first_hidden_layer_size);
init_W2 = initW(first_hidden_layer_size, sec_hidden_layer_size);
init_W3 = initW(sec_hidden_layer_size, output_layer_size);

init_bias_L1 = 0;
init_bias_L2 = 0;
init_bias_L3 = 0;

%% =========== Part 3: NN Training  =============
fprintf('Neural Network Training... \n')

lambda = 0;

% Perform forward and back propogation
[W1, W2, W3, bias_L2, bias_L3, J_train] = GD(init_W1, init_W2, init_W3, ...
                                   init_bias_L1, init_bias_L2, init_bias_L3, ...
                                   X_train, y_train, lambda);

bias_L1 = init_bias_L1;

% To Plot the convergence graph, Gradient Descent
figure(1);
plot(1:numel(J_train), J_train, '-r', 'LineWidth', 1.5);
legend('Train')
xlabel('Number of epochs')
ylabel('Error')

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 4: Prediction/Accuracy Calculation  =============

accTraining = accCal(W1, W2, W3, bias_L1, bias_L2, bias_L3, X_train, y_train);
accTest = accCal(W1, W2, W3, bias_L1, bias_L2, bias_L3, X_test, y_test);
accCV = accCal(W1, W2, W3, bias_L1, bias_L2, bias_L3, X_cv, y_cv);

fprintf('\nTraining Set Accuracy: %f\n', accTraining);
fprintf('\nTest Set Accuracy: %f\n', accTest);
fprintf('\nCV Set Accuracy: %f\n', accCV);
% save('save_vars', 'W1', 'W2', 'W3', 'bias_L2', 'bias_L3', 'J_train')
