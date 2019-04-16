clear all;
close all;
clc;

addpath ../Datasets/cifar-10-batches-mat/;

N = 10000; % Number of images
d = 3072; % The dimensionality of each image
K = 10; % Number of labels

% Load all datasets.
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat', d, N, K); 
[trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat', d, N, K); 
[trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat', d, N, K); 
[trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat', d, N, K); 
[trainAndValX, trainAndValY, trainAndValy] = LoadBatch('data_batch_5.mat', d, N, K);
[testX, testY, testy] = LoadBatch('test_batch.mat', d, N, K);

% Use 5000 images for validation.
% trainX5 = trainAndValX(:, 1:5000);
% trainY5 = trainAndValY(:, 1:5000);
% trainy5 = trainAndValy(1:5000);
% valX = trainAndValX(:, 5001:10000);
% valY = trainAndValY(:, 5001:10000);
% valy = trainAndValy(5001:10000);

% Use 1000 images for validation.
trainX5 = trainAndValX(:, 1:9000);
trainY5 = trainAndValY(:, 1:9000);
trainy5 = trainAndValy(1:9000);
valX = trainAndValX(:, 9001:10000);
valY = trainAndValY(:, 9001:10000);
valy = trainAndValy(9001:10000);

trainX = horzcat(trainX, trainX2, trainX3, trainX4, trainX5);
trainY = horzcat(trainY, trainY2, trainY3, trainY4, trainY5);
trainy = vertcat(trainy, trainy2, trainy3, trainy4, trainy5);

m = 50; % Number of nodes in the hidden layer.

n_batch = 100;
n = 49000; % Total number of training samles from all batches.
n_s = 2 * floor(n / n_batch);

n_epochs = 8;

eta_min = 1e-5;
eta_max = 1e-1;

% l_min = -5;
% l_max = -3;

for test=1:1
    % l = l_min + (l_max - l_min)* rand(1, 1);
    % lambda = 10^l;
    lambda = 3.7431e-05;
  
    [W, b] = InitParams(K, d, m);
    [Wstar, bstar] = MiniBatchGD(trainX, trainY, n_batch, eta_min ,eta_max, n_s, n_epochs, W, b, lambda, n, valX, valY, trainy, valy);
    
    % Measure perfomance on the validation set.
    acc = ComputeAccuracy(valX, valy, Wstar, bstar);
    disp(lambda + " & " + acc + " \\");
    disp("\hline");
    
    testAccuracy = ComputeAccuracy(testX, testy, Wstar, bstar);
    disp(testAccuracy);
end


% % Relative error checks
% [W, b] = InitParams(K, d, m);
% W{1} = W{1}(:,1:20);
% lambda = 1;
% [P, H] = EvaluateClassifier(trainX(1:20,1:100), W, b);
% 
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:20,1:100), trainY(:,1:100), W, b, lambda, 1e-5);
% 
% [grad_W, grad_b] = ComputeGradients(trainX(1:20,1:100), trainY(:,1:100), P, H, W, lambda);
% 
% error1 = ComputeRelativeError(grad_b{1}, ngrad_b{1});
% disp(error1);
% 
% error2 = ComputeRelativeError(grad_b{2}, ngrad_b{2});
% disp(error2);
% 
% error3 = ComputeRelativeError(grad_W{1}, ngrad_W{1});
% disp(error3);
% 
% error4 = ComputeRelativeError(grad_W{2}, ngrad_W{2});
% disp(error4);


function [X, Y, y] = LoadBatch(filename, d, N, K)
    A = load(filename);
    X = reshape(A.data', d, N); % X size d x N. N  = 10000 and d = 3072
    X = double(X);
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X,2)]);
    
    y = A.labels + 1; % To encode labels between 1-10.
    Y = zeros(K, N);
    for i = 1:length(y) 
       Y(y(i), i) = 1;
    end
end


function [W, b] = InitParams(K, d, m)
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    
    W = cell(2, 1);
    b = cell(2, 1);
    
    std1 = 1/sqrt(d);
    std2 = 1/sqrt(m);
    
    W1 = std1.*randn(m, d);
    W2 = std2.*randn(K, m);
    
    b{1} = b1;
    b{2} = b2;
    
    W{1} = W1;
    W{2} = W2;
   
end

% Copied function
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);

            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end


function [P, h] = EvaluateClassifier(X, W, b)
    s1 = W{1}*X + b{1};
    h = max(0, s1);
    s = W{2}*h + b{2};
    P = exp(s)./sum(exp(s));
end

% Compute cost and loss.
function [J, c] = ComputeCost(X, Y, W, b, lambda)
    
    [P, h] = EvaluateClassifier(X, W, b);
    l = zeros(1,size(Y, 2));
    
    for i = 1:size(Y, 2)
        l(i) = -log(Y(:,i)'*P(:,i));
    end
    c = sum(l)/size(Y, 2);
    J = c + lambda * (sum(sum(W{1}.^2)) + sum(sum(W{2}.^2)));
end 
 
% Function that computes the accuracy of the nets
function acc = ComputeAccuracy(X, y, W, b)
    [P, h] = EvaluateClassifier(X, W, b);
    [~, index] = max(P);
    I = find(index(:) == y);
    acc = length(I)/length(X);
end 


function [grad_W, grad_b] = ComputeGradients(X, Y, P, H, W, lambda)
    [d, N] = size(X);
    grad_W = cell(2, 1);
    grad_b = cell(2, 1);
    
    vector_of_ones = ones(N, 1);

    G = -1*(Y-P);
    dLdW2 = (1/N)*(G*transpose(H));
    dLdb2 = (1/N)*(G*vector_of_ones);
    G = transpose(W{2})*G;
    H = H > 0;
    G = G.*H;
    
    dLdW1 = (1/N)*(G*transpose(X));
    dLdb1 = (1/N)*(G*vector_of_ones);
    
    grad_W{1} = dLdW1 + 2*lambda*W{1};
    grad_W{2} = dLdW2 + 2*lambda*W{2};
    grad_b{1} = dLdb1;
    grad_b{2} = dLdb2;

end

% Compute relative error between numerically computed gradient
% value and analytically computed gradient value.
function [rel_error] = ComputeRelativeError(g_a, g_n)
    eps = 1e-6;
    rel_error = norm(g_a-g_n)/max(eps, (norm(g_a) + norm(g_n)));
end

% 
function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch, eta_min ,eta_max, n_s, n_epochs, W, b, lambda, N, valX, valY, y, valy)
    l = 0;
    t = 0;
    
    for i=1:n_epochs
        
%     Plot stuff 
%     [J, c] = ComputeCost(X, Y, W, b, lambda);
%     [valJ, valc] = ComputeCost(valX, valY, W, b, lambda);
%     acc = ComputeAccuracy(X, y, W, b);
%     acc_val = ComputeAccuracy(valX, valy, W, b);
%     x(i) = t;
% 
%     trainingLoss(i) = c;
%     validationLoss(i) = valc;
%     trainingCost(i) = J;
%     validationCost(i) = valJ;
%     trainAcc(i) = acc;
%     valAcc(i) = acc_val;
        
        for j=1:N/n_batch
            t = t + 1;
            if 2*l*n_s <= t && t <= (2*l + 1)*n_s
                eta = eta_min + ((t - 2*l*n_s)/n_s)*(eta_max - eta_min);
            else
                eta = eta_max - ((t - (2*l + 1)*n_s)/n_s)*(eta_max - eta_min);
            end

            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            [P, h] = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, W, lambda);
            W{1} = W{1} - eta * grad_W{1};
            b{1} = b{1} - eta * grad_b{1};
            W{2} = W{2} - eta * grad_W{2};
            b{2} = b{2} - eta * grad_b{2};    
        end    
            
        if(t >= 2*n_s)
            l = 1;
        end

        if(t >= 4*n_s)
            l = 2;
        end
        
        if(t >= 6*n_s)
            l = 3;
        end

    end
    
%     [J, c] = ComputeCost(X, Y, W, b, lambda);
%     [valJ, valc] = ComputeCost(valX, valY, W, b, lambda);
%     acc = ComputeAccuracy(X, y, W, b);
%     acc_val = ComputeAccuracy(valX, valy, W, b);
%     x(i + 1) = t;
% 
%     trainingCost(i + 1) = J;
%     validationCost(i + 1) = valJ;
% 
%     trainingLoss(i + 1) = c;
%     validationLoss(i + 1) = valc;
% 
%     trainAcc(i + 1) = acc;
%     valAcc(i + 1) = acc_val;
% 
%     figure
%     plot(1:1:n_epochs + 1, trainingLoss)
%     xlabel('epochs')
%     ylabel('loss')
%     legend('training')
%     xlim([1, n_epochs]);
%     title('Loss Plot')
    
    Wstar{1} = W{1};
    Wstar{2} = W{2};
  
    bstar{1} = b{1};
    bstar{2} = b{2};
end 



