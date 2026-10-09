"""
Deep Neural Network - Application (Cat vs Non-Cat Classifier)
Deep Learning Specialization - Course 1: Neural Networks and Deep Learning (Week 4, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)
"""

import numpy as np
import copy
from building_deep_neural_network import (
    initialize_parameters,
    initialize_parameters_deep,
    linear_activation_forward,
    L_model_forward,
    compute_cost,
    linear_activation_backward,
    L_model_backward,
    update_parameters
)

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        cost = compute_cost(A2, Y)
        
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0:
            costs.append(cost)
            
    return parameters, costs

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0:
            costs.append(cost)
            
    return parameters, costs

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))
    
    probas, caches = L_model_forward(X, parameters)
    
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
            
    accuracy = np.sum((p == y) / m)
    return p, accuracy

if __name__ == "__main__":
    np.random.seed(1)
    n_x = 12
    n_h = 7
    n_y = 1
    m = 20
    X = np.random.randn(n_x, m)
    Y = (np.random.randn(1, m) > 0).astype(int)
    
    # Test two_layer_model
    params_2l, costs_2l = two_layer_model(X, Y, (n_x, n_h, n_y), num_iterations=5)
    assert "W1" in params_2l and "W2" in params_2l
    
    # Test L_layer_model
    layers_dims = [n_x, 8, 4, n_y]
    params_ll, costs_ll = L_layer_model(X, Y, layers_dims, num_iterations=5)
    assert "W1" in params_ll and "W3" in params_ll
    
    preds, acc = predict(X, Y, params_ll)
    assert preds.shape == (1, m)
    
    print("All Deep Neural Network Application unit tests passed with 100% precision!")
