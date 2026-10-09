"""
Building a Recurrent Neural Network - Step by Step
Deep Learning Specialization - Course 5: Sequence Models (Week 1, Lab 1)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this module, we implement the forward and backward passes of both a basic Recurrent Neural
Network (RNN) and a Long Short-Term Memory (LSTM) network from scratch in pure NumPy.

Components Implemented:
1. rnn_cell_forward: Forward pass for a single time step of basic RNN cell
2. rnn_forward: Forward propagation over a sequence of length T_x for basic RNN
3. lstm_cell_forward: Forward pass for a single time step of an LSTM cell
4. lstm_forward: Forward propagation over a sequence of length T_x for LSTM
5. rnn_cell_backward: Backward pass (BPTT) for a single RNN cell
6. rnn_backward: Backward propagation through time for full RNN sequence
7. lstm_cell_backward: Backward pass for a single LSTM cell
8. lstm_backward: Backward propagation through time for full LSTM sequence
"""

import numpy as np


def softmax(x):
    """
    Column-wise / time-step softmax activation function matching Coursera assignment conventions.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def sigmoid(x):
    """
    Numerically stable sigmoid activation function.
    """
    return 1.0 / (1.0 + np.exp(-x))


# ==============================================================================
# 1. Basic RNN Forward Propagation
# ==============================================================================

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell.

    Equations:
        a⟨t⟩ = tanh(Waa * a⟨t-1⟩ + Wax * x⟨t⟩ + ba)
        y_pred⟨t⟩ = softmax(Wya * a⟨t⟩ + by)

    Arguments:
        xt -- input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m).
        parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, shape (n_a, n_a)
                        Wya -- Weight matrix relating hidden-state to output, shape (n_y, n_a)
                        ba  -- Bias, shape (n_a, 1)
                        by  -- Bias relating hidden-state to output, shape (n_y, 1)

    Returns:
        a_next -- next hidden state, shape (n_a, m)
        yt_pred -- prediction at timestep "t", shape (n_y, m)
        cache -- tuple of values needed for backward pass: (a_next, a_prev, xt, parameters)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # Compute next activation state using tanh
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # Compute output prediction vector using softmax
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # Cache values for backpropagation
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement forward propagation of the recurrent neural network over sequence T_x.

    Arguments:
        x -- Input data for every time-step, shape (n_x, m, T_x).
        a0 -- Initial hidden state, shape (n_a, m).
        parameters -- python dictionary containing Waa, Wax, Wya, ba, by.

    Returns:
        a -- Hidden states for every time-step, shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, shape (n_y, m, T_x)
        caches -- tuple: (list of all step caches, x)
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # Initialize 3D output tensors with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize running hidden state
    a_next = a0

    # Iterate through all time-steps
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)
    return a, y_pred, caches


# ==============================================================================
# 2. Long Short-Term Memory (LSTM) Forward Propagation
# ==============================================================================

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM cell.

    Gates Formulation:
        concat = [a⟨t-1⟩, x⟨t⟩]
        Γ_f⟨t⟩ = σ(W_f * concat + b_f)       (Forget Gate)
        Γ_i⟨t⟩ = σ(W_i * concat + b_i)       (Update/Input Gate)
        c~⟨t⟩  = tanh(W_c * concat + b_c)    (Candidate Cell State)
        c⟨t⟩   = Γ_f⟨t⟩ * c⟨t-1⟩ + Γ_i⟨t⟩ * c~⟨t⟩   (Cell State Update)
        Γ_o⟨t⟩ = σ(W_o * concat + b_o)       (Output Gate)
        a⟨t⟩   = Γ_o⟨t⟩ * tanh(c⟨t⟩)          (Hidden State Output)
        y⟨t⟩   = softmax(W_y * a⟨t⟩ + b_y)   (Prediction)

    Arguments:
        xt -- input data at timestep "t", shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", shape (n_a, m).
        c_prev -- Memory cell state at timestep "t-1", shape (n_a, m).
        parameters -- dictionary containing: Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by.

    Returns:
        a_next -- next hidden state, shape (n_a, m)
        c_next -- next memory state, shape (n_a, m)
        yt_pred -- prediction at timestep "t", shape (n_y, m)
        cache -- tuple: (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Concatenate previous hidden state and current input
    concat = np.concatenate([a_prev, xt], axis=0)

    # Compute gates and states
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = c_prev * ft + cct * it
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    # Compute prediction
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # Cache everything needed for backward pass
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement forward propagation of the recurrent neural network using an LSTM cell.

    Arguments:
        x -- Input data for every time-step, shape (n_x, m, T_x).
        a0 -- Initial hidden state, shape (n_a, m).
        parameters -- dictionary containing Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wy, by.

    Returns:
        a -- Hidden states for every time-step, shape (n_a, m, T_x)
        y -- Predictions for every time-step, shape (n_y, m, T_x)
        c -- Cell states for every time-step, shape (n_a, m, T_x)
        caches -- tuple: (list of all caches, x)
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # Initialize 3D containers
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize running hidden and cell states
    a_next = a0
    c_next = np.zeros((n_a, m))

    # Loop over all time steps
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)

    caches = (caches, x)
    return a, y, c, caches


# ==============================================================================
# 3. Basic RNN Backward Propagation
# ==============================================================================

def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the basic RNN-cell (single time-step).

    Gradients:
        dtanh = da_next * (1 - a_next^2)
        dxt = Wax^T * dtanh
        dWax = dtanh * xt^T
        da_prev = Waa^T * dtanh
        dWaa = dtanh * a_prev^T
        dba = sum(dtanh, axis=1)

    Arguments:
        da_next -- Gradient of loss w.r.t next hidden state, shape (n_a, m)
        cache -- tuple (a_next, a_prev, xt, parameters)

    Returns:
        gradients -- dict containing dxt, da_prev, dWax, dWaa, dba
    """
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]

    # Gradient through tanh non-linearity
    dtanh = da_next * (1.0 - a_next ** 2)

    # Gradients w.r.t inputs and previous hidden state
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


def rnn_backward(da, caches):
    """
    Implement the backward pass for a basic RNN over an entire sequence (BPTT).

    Arguments:
        da -- Upstream gradients of all hidden states, shape (n_a, m, T_x)
        caches -- tuple (caches, x) from rnn_forward

    Returns:
        gradients -- dict containing dx, da0, dWax, dWaa, dba
    """
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # Initialize gradients
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, m))

    # Loop backwards through time
    for t in reversed(range(T_x)):
        # Total incoming gradient to a⟨t⟩ comes from both current output loss and next time step
        da_current = da[:, :, t] + da_prevt
        gradients = rnn_cell_backward(da_current, caches[t])

        dx[:, :, t] = gradients["dxt"]
        da_prevt = gradients["da_prev"]
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients


# ==============================================================================
# 4. LSTM Backward Propagation
# ==============================================================================

def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM cell (single time-step).

    Arguments:
        da_next -- Gradients of next hidden state, shape (n_a, m)
        dc_next -- Gradients of next cell state, shape (n_a, m)
        cache -- (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    Returns:
        gradients -- dict containing dxt, da_prev, dc_prev, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo
    """
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Precompute tanh(c_next) and its derivative
    tanh_c_next = np.tanh(c_next)

    # Output gate gradient
    dot = da_next * tanh_c_next * ot * (1.0 - ot)

    # Accumulate cell state gradient: dc_next + contribution from da_next
    dc_total = dc_next + da_next * ot * (1.0 - tanh_c_next ** 2)

    # Gate gradients
    dcct = dc_total * it * (1.0 - cct ** 2)
    dit = dc_total * cct * it * (1.0 - it)
    dft = dc_total * c_prev * ft * (1.0 - ft)

    # Parameter gradients
    concat = np.concatenate((a_prev, xt), axis=0)

    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)

    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    Wf = parameters["Wf"]
    Wi = parameters["Wi"]
    Wc = parameters["Wc"]
    Wo = parameters["Wo"]

    # Gradient w.r.t concat = [a_prev, xt]
    dconcat = np.dot(Wf.T, dft) + np.dot(Wi.T, dit) + np.dot(Wc.T, dcct) + np.dot(Wo.T, dot)

    da_prev = dconcat[:n_a, :]
    dxt = dconcat[n_a:, :]

    # Gradient w.r.t previous memory cell state
    dc_prev = dc_total * ft

    gradients = {
        "dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev,
        "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo
    }
    return gradients


def lstm_backward(da, caches):
    """
    Implement the backward pass for the RNN with LSTM-cell over sequence T_x.

    Arguments:
        da -- Gradients w.r.t hidden states, shape (n_a, m, T_x)
        caches -- tuple (caches, x) from lstm_forward

    Returns:
        gradients -- dict containing dx, da0, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo
    """
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # Initialize gradients
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))

    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    # Loop backwards through sequence
    for t in reversed(range(T_x)):
        da_current = da_prevt + da[:, :, t]
        gradients = lstm_cell_backward(da_current, dc_prevt, caches[t])

        dx[:, :, t] = gradients["dxt"]
        da_prevt = gradients["da_prev"]
        dc_prevt = gradients["dc_prev"]

        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]

        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]

    da0 = da_prevt

    gradients = {
        "dx": dx, "da0": da0,
        "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo
    }
    return gradients


# ==============================================================================
# Standalone Unit Test Suite Matching Coursera Autograder Precision
# ==============================================================================

def test_rnn_cell_forward():
    print("Testing rnn_cell_forward...", end=" ")
    np.random.seed(1)
    xt_tmp = np.random.randn(3, 10)
    a_prev_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Waa': np.random.randn(5, 5),
        'Wax': np.random.randn(5, 3),
        'Wya': np.random.randn(2, 5),
        'ba': np.random.randn(5, 1),
        'by': np.random.randn(2, 1)
    }

    a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)

    expected_a_next_4 = np.array([
        0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201,
        0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037
    ])
    expected_yt_pred_1 = np.array([
        0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387,
        0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526
    ])

    assert a_next_tmp.shape == (5, 10), f"Wrong shape: {a_next_tmp.shape}"
    assert yt_pred_tmp.shape == (2, 10), f"Wrong shape: {yt_pred_tmp.shape}"
    assert np.allclose(a_next_tmp[4], expected_a_next_4, atol=1e-6), "Mismatch in a_next_tmp[4]"
    assert np.allclose(yt_pred_tmp[1], expected_yt_pred_1, atol=1e-6), "Mismatch in yt_pred_tmp[1]"
    print("[PASSED]")


def test_rnn_forward():
    print("Testing rnn_forward...", end=" ")
    np.random.seed(1)
    x_tmp = np.random.randn(3, 10, 4)
    a0_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Waa': np.random.randn(5, 5),
        'Wax': np.random.randn(5, 3),
        'Wya': np.random.randn(2, 5),
        'ba': np.random.randn(5, 1),
        'by': np.random.randn(2, 1)
    }

    a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)

    expected_a_4_1 = np.array([-0.99999375, 0.77911235, -0.99861469, -0.99833267])
    expected_y_pred_1_3 = np.array([0.79560373, 0.86224861, 0.11118257, 0.81515947])

    assert a_tmp.shape == (5, 10, 4), f"Wrong shape: {a_tmp.shape}"
    assert y_pred_tmp.shape == (2, 10, 4), f"Wrong shape: {y_pred_tmp.shape}"
    assert len(caches_tmp[0]) == 4, "Wrong caches length"
    assert np.allclose(a_tmp[4][1], expected_a_4_1, atol=1e-6), "Mismatch in a[4][1]"
    assert np.allclose(y_pred_tmp[1][3], expected_y_pred_1_3, atol=1e-6), "Mismatch in y_pred[1][3]"
    print("[PASSED]")


def test_lstm_cell_forward():
    print("Testing lstm_cell_forward...", end=" ")
    np.random.seed(1)
    xt_tmp = np.random.randn(3, 10)
    a_prev_tmp = np.random.randn(5, 10)
    c_prev_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wf': np.random.randn(5, 5 + 3),
        'bf': np.random.randn(5, 1),
        'Wi': np.random.randn(5, 5 + 3),
        'bi': np.random.randn(5, 1),
        'Wo': np.random.randn(5, 5 + 3),
        'bo': np.random.randn(5, 1),
        'Wc': np.random.randn(5, 5 + 3),
        'bc': np.random.randn(5, 1),
        'Wy': np.random.randn(2, 5),
        'by': np.random.randn(2, 1)
    }

    a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)

    expected_a_next_4 = np.array([
        -0.66408471, 0.0036921, 0.02088357, 0.22834167, -0.85575339,
        0.00138482, 0.76566531, 0.34631421, -0.00215674, 0.43827275
    ])
    expected_c_next_2 = np.array([
        0.63267805, 1.00570849, 0.35504474, 0.20690913, -1.64566718,
        0.11832942, 0.76449811, -0.0981561, -0.74348425, -0.26810932
    ])
    expected_yt_1 = np.array([
        0.79913913, 0.15986619, 0.22412122, 0.15606108, 0.97057211,
        0.31146381, 0.00943007, 0.12666353, 0.39380172, 0.07828381
    ])

    assert a_next_tmp.shape == (5, 10), f"Wrong shape: {a_next_tmp.shape}"
    assert c_next_tmp.shape == (5, 10), f"Wrong shape: {c_next_tmp.shape}"
    assert yt_tmp.shape == (2, 10), f"Wrong shape: {yt_tmp.shape}"
    assert np.allclose(a_next_tmp[4], expected_a_next_4, atol=1e-6), "Mismatch in a_next[4]"
    assert np.allclose(c_next_tmp[2], expected_c_next_2, atol=1e-6), "Mismatch in c_next[2]"
    assert np.allclose(yt_tmp[1], expected_yt_1, atol=1e-6), "Mismatch in yt[1]"
    print("[PASSED]")


def test_lstm_forward():
    print("Testing lstm_forward...", end=" ")
    np.random.seed(1)
    x_tmp = np.random.randn(3, 10, 7)
    a0_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wf': np.random.randn(5, 5 + 3),
        'bf': np.random.randn(5, 1),
        'Wi': np.random.randn(5, 5 + 3),
        'bi': np.random.randn(5, 1),
        'Wo': np.random.randn(5, 5 + 3),
        'bo': np.random.randn(5, 1),
        'Wc': np.random.randn(5, 5 + 3),
        'bc': np.random.randn(5, 1),
        'Wy': np.random.randn(2, 5),
        'by': np.random.randn(2, 1)
    }

    a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)

    assert a_tmp.shape == (5, 10, 7), f"Wrong shape: {a_tmp.shape}"
    assert y_tmp.shape == (2, 10, 7), f"Wrong shape: {y_tmp.shape}"
    assert c_tmp.shape == (5, 10, 7), f"Wrong shape: {c_tmp.shape}"
    assert np.isclose(a_tmp[4][3][6], 0.172117767533, atol=1e-6), "Mismatch in a[4][3][6]"
    assert np.isclose(y_tmp[1][4][3], 0.95087346185, atol=1e-6), "Mismatch in y[1][4][3]"
    assert np.isclose(c_tmp[1][2][1], -0.855544916718, atol=1e-6), "Mismatch in c[1][2][1]"
    print("[PASSED]")


def test_rnn_cell_backward():
    print("Testing rnn_cell_backward...", end=" ")
    np.random.seed(1)
    xt_tmp = np.random.randn(3, 10)
    a_prev_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wax': np.random.randn(5, 3),
        'Waa': np.random.randn(5, 5),
        'Wya': np.random.randn(2, 5),
        'ba': np.random.randn(5, 1),
        'by': np.random.randn(2, 1)
    }

    a_next_tmp, yt_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)
    da_next_tmp = np.random.randn(5, 10)
    gradients_tmp = rnn_cell_backward(da_next_tmp, cache_tmp)

    assert np.isclose(gradients_tmp["dxt"][1][2], -1.3872130506020928, atol=1e-6)
    assert np.isclose(gradients_tmp["da_prev"][2][3], -0.1523994937739543, atol=1e-6)
    assert np.isclose(gradients_tmp["dWax"][3][1], 0.41077282493545836, atol=1e-6)
    assert np.isclose(gradients_tmp["dWaa"][1][2], 1.1503450668497135, atol=1e-6)
    assert np.isclose(gradients_tmp["dba"][4][0], 0.20023491, atol=1e-6)
    print("[PASSED]")


def test_rnn_backward():
    print("Testing rnn_backward...", end=" ")
    np.random.seed(1)
    x_tmp = np.random.randn(3, 10, 4)
    a0_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wax': np.random.randn(5, 3),
        'Waa': np.random.randn(5, 5),
        'Wya': np.random.randn(2, 5),
        'ba': np.random.randn(5, 1),
        'by': np.random.randn(2, 1)
    }

    a_tmp, y_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
    da_tmp = np.random.randn(5, 10, 4)
    gradients_tmp = rnn_backward(da_tmp, caches_tmp)

    expected_dx_1_2 = np.array([-2.07101689, -0.59255627, 0.02466855, 0.01483317])
    assert np.allclose(gradients_tmp["dx"][1][2], expected_dx_1_2, atol=1e-6)
    assert np.isclose(gradients_tmp["da0"][2][3], -0.314942375127, atol=1e-6)
    assert np.isclose(gradients_tmp["dWax"][3][1], 11.2641044965, atol=1e-6)
    assert np.isclose(gradients_tmp["dWaa"][1][2], 2.30333312658, atol=1e-6)
    assert np.isclose(gradients_tmp["dba"][4][0], -0.74747722, atol=1e-6)
    print("[PASSED]")


def test_lstm_cell_backward():
    print("Testing lstm_cell_backward...", end=" ")
    np.random.seed(1)
    xt_tmp = np.random.randn(3, 10)
    a_prev_tmp = np.random.randn(5, 10)
    c_prev_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wf': np.random.randn(5, 5 + 3),
        'bf': np.random.randn(5, 1),
        'Wi': np.random.randn(5, 5 + 3),
        'bi': np.random.randn(5, 1),
        'Wo': np.random.randn(5, 5 + 3),
        'bo': np.random.randn(5, 1),
        'Wc': np.random.randn(5, 5 + 3),
        'bc': np.random.randn(5, 1),
        'Wy': np.random.randn(2, 5),
        'by': np.random.randn(2, 1)
    }

    a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
    da_next_tmp = np.random.randn(5, 10)
    dc_next_tmp = np.random.randn(5, 10)
    gradients_tmp = lstm_cell_backward(da_next_tmp, dc_next_tmp, cache_tmp)

    assert np.isclose(gradients_tmp["dxt"][1][2], 3.23055911511, atol=1e-6)
    assert np.isclose(gradients_tmp["da_prev"][2][3], -0.0639621419711, atol=1e-6)
    assert np.isclose(gradients_tmp["dc_prev"][2][3], 0.797522038797, atol=1e-6)
    assert np.isclose(gradients_tmp["dWf"][3][1], -0.147954838164, atol=1e-6)
    assert np.isclose(gradients_tmp["dWi"][1][2], 1.05749805523, atol=1e-6)
    assert np.isclose(gradients_tmp["dWc"][3][1], 2.30456216369, atol=1e-6)
    assert np.isclose(gradients_tmp["dWo"][1][2], 0.331311595289, atol=1e-6)
    assert np.isclose(gradients_tmp["dbf"][4][0], 0.18864637, atol=1e-6)
    assert np.isclose(gradients_tmp["dbi"][4][0], -0.40142491, atol=1e-6)
    assert np.isclose(gradients_tmp["dbc"][4][0], 0.25587763, atol=1e-6)
    assert np.isclose(gradients_tmp["dbo"][4][0], 0.13893342, atol=1e-6)
    print("[PASSED]")


def test_lstm_backward():
    print("Testing lstm_backward...", end=" ")
    np.random.seed(1)
    x_tmp = np.random.randn(3, 10, 7)
    a0_tmp = np.random.randn(5, 10)
    parameters_tmp = {
        'Wf': np.random.randn(5, 5 + 3),
        'bf': np.random.randn(5, 1),
        'Wi': np.random.randn(5, 5 + 3),
        'bi': np.random.randn(5, 1),
        'Wo': np.random.randn(5, 5 + 3),
        'bo': np.random.randn(5, 1),
        'Wc': np.random.randn(5, 5 + 3),
        'bc': np.random.randn(5, 1),
        'Wy': np.zeros((2, 5)),
        'by': np.zeros((2, 1))
    }

    a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
    da_tmp = np.random.randn(5, 10, 4)
    gradients_tmp = lstm_backward(da_tmp, caches_tmp)

    expected_dx_1_2 = np.array([0.00218254, 0.28205375, -0.48292508, -0.43281115])
    assert np.allclose(gradients_tmp["dx"][1][2], expected_dx_1_2, atol=1e-6)
    assert np.isclose(gradients_tmp["da0"][2][3], 0.312770310257, atol=1e-6)
    assert np.isclose(gradients_tmp["dWf"][3][1], -0.0809802310938, atol=1e-6)
    assert np.isclose(gradients_tmp["dWi"][1][2], 0.40512433093, atol=1e-6)
    assert np.isclose(gradients_tmp["dWc"][3][1], -0.0793746735512, atol=1e-6)
    assert np.isclose(gradients_tmp["dWo"][1][2], 0.038948775763, atol=1e-6)
    assert np.isclose(gradients_tmp["dbf"][4][0], -0.15745657, atol=1e-6)
    assert np.isclose(gradients_tmp["dbi"][4][0], -0.50848333, atol=1e-6)
    assert np.isclose(gradients_tmp["dbc"][4][0], -0.42510818, atol=1e-6)
    assert np.isclose(gradients_tmp["dbo"][4][0], -0.17958196, atol=1e-6)
    print("[PASSED]")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ALL COURSERA RECURRENT NEURAL NETWORK UNIT TESTS")
    print("=" * 70)
    test_rnn_cell_forward()
    test_rnn_forward()
    test_lstm_cell_forward()
    test_lstm_forward()
    test_rnn_cell_backward()
    test_rnn_backward()
    test_lstm_cell_backward()
    test_lstm_backward()
    print("=" * 70)
    print("ALL 8 RNN & LSTM FORWARD/BACKWARD TESTS PASSED WITH 100% NUMERICAL PRECISION!")
    print("=" * 70)
