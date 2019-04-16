clear all;
close all;
clc;

addpath Datasets/cifar-10-batches-mat/;

N = 10000; % Number of images
d = 3072; % The dimensionality of each image
K = 10; % Number of labels

% Parameter settings to test:
% Setting 1:
lambda = 0;
n_epochs = 40;
n_batch = 100;
eta = .1;

% Setting 2:
% lambda = 0;
% n_epochs = 40;
% n_batch = 100;
% eta = .01;

% Setting 3:
% lambda = .1;
% n_epochs = 40;
% n_batch = 100;
% eta = .01;

% Setting 4:
% lambda = 1;
% n_epochs = 40;
% n_batch = 100;
% eta = .01;

% Load all datasets.
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat', d, N, K); 
[valX, valY, valy] = LoadBatch('data_batch_2.mat', d, N, K);
[testX, testY, testy] = LoadBatch('test_batch.mat', d, N, K);

% Uncomment for debugging by setting the seed.
% rng(400); 

% Initialize W and b to have Gaussian random values with
% zero mean and standard deviation 0.01.
[W, b] = InitParams(K, d);


% Code to test relative error:
%
P = EvaluateClassifier(trainX(1:20, 1), W(:,1:20), b);
[grad_W, grad_b] = ComputeGradients(trainX(1:20, 1), trainY(:, 1), P, W(:, 1:20), lambda);
% disp(grad_b);
%
% [ngrad_b, ngrad_W] = ComputeGradsNum(trainX(:, 1), trainY(:, 1), W, b, lambda, 1e-6);
% disp(ngrad_b);
%
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:20, 1), trainY(:, 1),W(:,1:20), b, lambda, 1e-6);
% disp(ngrad_b);
%
error = ComputeRelativeError(grad_b, ngrad_b);
disp(error);

% Run the learning algorithm and compute the accuracy.
[Wstar, bstar] = MiniBatchGD(trainX, trainY, n_batch, eta, n_epochs, W, b, lambda, N, valX, valY);
acc = ComputeAccuracy(testX, testy, Wstar, bstar);
% disp(acc);

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
% montage(s_im, 'Size', [1,10]);

% Step 1
% Function that reads in the data from CIFAR-10 batch file
% and returns the image and label data in spearate files.
function [X, Y, y] = LoadBatch(filename, d, N, K)
    A = load(filename);
    X = reshape(A.data', d, N); % X size d x N. N  = 10000 and d = 3072
    X = double(X);
    X = X/255;
    y = A.labels + 1; % To encode labels between 1-10.
    Y = zeros(K, N);
    for i = 1:length(y) 
       Y(y(i), i) = 1;
    end
end

% Step 2
% W size: K x d, 10 x 3072
% b size: K x 1, 10 x 1
% Zero mean and std dev 0.01
function [W, b] = InitParams(K, d)
    a = 0.01;
    W = a.*randn(K, d);
    b = a.*randn(K, 1);
end

% Step 3
% Function that evaluates the network function.
% Equation 1: s = Wx + b
% Equation 2: p = SOFTMAX(s) = exp(s)/1^T exp(s)
% Each column of P contains the probability for each
% label for the image in the corresponding column of X.
% P has size K x n.
function P = EvaluateClassifier(X, W, b)
    s = W*X + b;
    P = exp(s)./sum(exp(s));
end

% Step 4
% Function that computes the cost function given by
% equation 5.
function J = ComputeCost(X, Y, W, b, lambda)
    p = EvaluateClassifier(X, W, b);
    l = zeros(1,size(Y, 2));
    
    for i = 1:size(Y, 2)
        l(i) = -log(Y(:,i)'*p(:,i));
    end
    
    J = sum(l)/size(Y, 2) + lambda * sum(sum(W.^2));
end 

function [l, J] = ComputeCostWithSVMLoss(X, Y, W, b, lambda)
    s = W*X + b;
    l = zeros(size(s));
    
    totalSum = 0;
    for j = 1:size(Y, 2)
        for y = 1:size(Y, 2)
            if(j ~= y)
                totalSum = totalSum + max(0, s(j) - s(y) + 1);
            end 
        end
    end
    
    J = totalSum/size(Y, 2) + lambda * sum(sum(W.^2));
    
end 

% Step 5 
% Function that computes the accuracy of the nets
% predictions by equation 4.
% k* = arg max {p1, ..., pk}
function acc = ComputeAccuracy(X, y, W, b)
    p = EvaluateClassifier(X, W, b);
    [~, index] = max(p);
    I = find(index(:) == y);
    acc = length(I)/length(X);
end 

% Step 6
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    [~, N] = size(X);
    ones_vector = ones(N, 1);
    g = -(Y - P);
    grad_W = (g*X')/N + 2 * lambda * W;
    grad_b = (g*ones_vector)/N;
end


% Compute relative error between numerically computed gradient
% value and analytically computed gradient value.
function [rel_error] = ComputeRelativeError(g_a, g_n)
    eps = 1e-6;
    rel_error = norm(g_a-g_n)/max(eps, (norm(g_a) + norm(g_n)));
end


% Step 7
function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch, eta, n_epochs, W, b, lambda, N, valX, valY)
    trainingLoss = zeros(1, n_epochs);
    validationLoss = zeros(1, n_epochs);
    for i=1:n_epochs
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end  
        J = ComputeCost(X, Y, W, b, lambda);
        valJ = ComputeCost(valX, valY, W, b, lambda);
      
        trainingLoss(i) = J;
        validationLoss(i) = valJ;
    end
%     figure
%     plot(1:1:n_epochs, trainingLoss, 1:1:n_epochs, validationLoss)
%     xlabel('epochs')
%     ylabel('loss')
%     legend('training loss', 'validation loss')
%     xlim([1,n_epochs])
%     title('lambda=1,nepochs=40,nbatch=100,eta=.01')
    
    Wstar = W;
    bstar = b;
end 


% Copied function 1
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)
    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c) / h;
    end
end 

% Copied function 2
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end
end
