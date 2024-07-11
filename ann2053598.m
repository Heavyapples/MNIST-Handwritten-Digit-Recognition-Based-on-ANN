function [Yn,On,Yt,Ot] = ann2053598(ID, N_ep, lr, bp, u, v, w, cf)
    % Load MNIST train and test datasets
    mnist_train = readtable('MNIST_train_1000.csv');
    mnist_test = readtable('MNIST_test_100.csv');

    % Convert the labels into binary format: ID vs not-ID
    ID_array = arrayfun(@(x) str2double(x), num2str(ID));
    Yn = double(any(ismember(table2array(mnist_train(:, 1)), ID_array), 2));
    Yt = double(any(ismember(table2array(mnist_test(:, 1)), ID_array), 2));

    % Normalize the pixel values
    Xn = table2array(mnist_train(:, 2:end)) / 255;
    Xt = table2array(mnist_test(:, 2:end)) / 255;

    % Initialize the weights and biases
    [weights, biases] = initialize_network(u, v, w);

    % Train the network
    for epoch = 1:N_ep
        [outputs, activations] = forward_propagation(Xn, weights, biases);
        cost = compute_cost(Yn, outputs, cf);
        disp(['Epoch ', num2str(epoch), ', Cost: ', num2str(cost)]);

        % Different backprop methods based on 'bp'
        if bp == 1 % heuristic backprop
            [d_weights, d_biases] = backward_propagation(Xn, Yn, outputs, activations, weights, biases);
        else % calculus-based backprop
            [d_weights, d_biases] = calculus_based_backward_propagation(Xn, Yn, outputs, activations, weights, biases);
        end

        for layer = 1:length(weights)
            weights{layer} = weights{layer} - lr * d_weights{layer};
            biases{layer} = biases{layer} - lr * d_biases{layer};
        end
    end

    On = predict(Xn, weights, biases);
    Ot = predict(Xt, weights, biases);

    confusion_matrix_train = compute_confusion_matrix(Yn, On);
    confusion_matrix_test = compute_confusion_matrix(Yt, Ot);

    disp('Confusion matrix for the training data:');
    disp(confusion_matrix_train);
    disp('Confusion matrix for the test data:');
    disp(confusion_matrix_test);
end

function [weights, biases] = initialize_network(u, v, w)
    weights = {sqrt(1 / 784) * randn(u, 784), sqrt(1 / u) * randn(v, u), sqrt(1 / v) * randn(w, v), sqrt(1 / w) * randn(1, w)};
    biases = {zeros(u, 1), zeros(v, 1), zeros(w, 1), zeros(1, 1)};
end

function [outputs, activations] = forward_propagation(X, weights, biases)
    activations = cell(1, length(weights));
    outputs = cell(1, length(weights));
    
    for layer = 1:length(weights)
        if layer == 1
            activations{layer} = weights{layer} * X' + repmat(biases{layer}, 1, size(X, 1));
        else
            activations{layer} = weights{layer} * outputs{layer - 1} + repmat(biases{layer}, 1, size(outputs{layer - 1}, 2));
        end
        
        outputs{layer} = sigmoid(activations{layer});
    end
end

function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function [d_weights, d_biases] = backward_propagation(X, Y, outputs, activations, weights, biases)
    d_weights = cell(1, length(weights));
    d_biases = cell(1, length(biases));
    
    deltas = cell(1, length(outputs));

    % Calculate the delta for the output layer
    deltas{end} = 2 * (outputs{end} - Y') .* sigmoid_derivative(activations{end});

    % Calculate the delta for the hidden layers
    for layer = length(outputs) - 1:-1:1
        deltas{layer} = (weights{layer + 1}' * deltas{layer + 1}) .* sigmoid_derivative(activations{layer});
    end

    % Calculate the derivatives of the weights and biases
    for layer = 1:length(weights)
        if layer == 1
            d_weights{layer} = (deltas{layer} * X) / size(X, 1);
        else
            d_weights{layer} = (deltas{layer} * outputs{layer - 1}') / size(outputs{layer - 1}, 2);
        end
        d_biases{layer} = mean(deltas{layer}, 2);
    end
end

function g_prime = sigmoid_derivative(z)
    g_prime = sigmoid(z) .* (1 - sigmoid(z));
end

function cost = compute_cost(Y, outputs, cf)
    switch cf
        case 1 % Total Squared Error
            cost = sum((Y - outputs{end}).^2) / 2;
        case 2 % Cross-Entropy
            cost = -sum(Y.*log(outputs{end}) + (1-Y).*log(1-outputs{end}));
        otherwise
            error('Invalid cost function. Use 1 for TSE or 2 for CE.');
    end
end

function pred = predict(X, weights, biases)
    [~, activations] = forward_propagation(X, weights, biases);
    % The predicted value is 1 if the activation value is greater than 0.5, otherwise it is 0
    pred = double(activations{end}' > 0.5);
end

function confusion_matrix = compute_confusion_matrix(Y, pred)
    Y = Y(:);
    pred = pred(:);
    TP = sum((Y == 1) & (pred == 1));
    FP = sum((Y == 0) & (pred == 1));
    FN = sum((Y == 1) & (pred == 0));
    TN = sum((Y == 0) & (pred == 0));
    confusion_matrix = [TP, FP; FN, TN];
end

function [d_weights, d_biases] = calculus_based_backward_propagation(X, Y, outputs, activations, weights, biases)
    d_weights = cell(1, length(weights));
    d_biases = cell(1, length(biases));

    deltas = cell(1, length(outputs));

    % Calculate the delta for the output layer
    deltas{end} = (outputs{end} - Y') .* sigmoid_derivative(activations{end});

    % Calculate the delta for the hidden layers
    for layer = length(outputs) - 1:-1:1
        deltas{layer} = (weights{layer + 1}' * deltas{layer + 1}) .* sigmoid_derivative(activations{layer});
    end

    % Calculate the derivatives of the weights and biases
    for layer = 1:length(weights)
        if layer == 1
            d_weights{layer} = (deltas{layer} * X) / size(X, 1);
        else
            d_weights{layer} = (deltas{layer} * outputs{layer - 1}') / size(outputs{layer - 1}, 2);
        end
        d_biases{layer} = mean(deltas{layer}, 2);
    end
end
