"""
Convolutional Neural Networks: Step by Step
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 1, Lab 1)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this notebook/module, we implement convolutional (CONV) and pooling (POOL) layers in NumPy,
including both forward propagation and backward propagation (computing gradients).
"""

import numpy as np

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameter contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input activation
    """
    # Element-wise product between a_slice_prev and W
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float so that Z results in a scalar value.
    Z = float(Z + np.squeeze(b))
    
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
              numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula
    n_H = int(np.floor((n_H_prev - f + 2 * pad) / stride) + 1)
    n_W = int(np.floor((n_W_prev - f + 2 * pad) / stride) + 1)
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]   # Select ith training example's padded activation
        for h in range(n_H):         # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):     # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C): # loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):            # loop over the channels of the output volume
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values from the conv_forward(), (A_prev, W, b, hparameters)
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W),
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b),
          numpy array of shape (1, 1, 1, n_C)
    """
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                          
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using code formulas
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpadded da_prev_pad
        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
            
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    # Retrieve information from cache
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters"
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):                       # loop over the training examples
        # select ith training example from A_prev
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels
                    
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice in a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        # Create the mask from a_prev_slice
                        mask = create_mask_from_window(a_prev_slice)
                        
                        # Set dA_prev to be dA_prev + (mask * dA)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        # Get the value da from dA
                        da = dA[i, h, w, c]
                        
                        # Define the shape of the filter as f x f
                        shape = (f, f)
                        
                        # Distribute it to get the correct slice value. Add that to dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: CNN Step by Step (Coursera Assignment C4W1)")
    print("=" * 70)
    
    # 1. Test zero_pad
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, pad=3)
    assert x_pad.shape == (4, 9, 9, 2), f"Expected shape (4, 9, 9, 2), got {x_pad.shape}"
    assert np.allclose(x[1, 1], x_pad[1, 4, 3:6]), "Padding values alignment mismatch"
    assert np.allclose(x[1, 1, 1], x_pad[1, 4, 4]), "Padding single pixel alignment mismatch"
    assert np.all(x_pad[:, :3, :, :] == 0), "Padded vertical top borders must be zero"
    print("[PASS] zero_pad verified.")
    
    # 2. Test conv_single_step
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    Z = conv_single_step(a_slice_prev, W, b)
    expected_Z = -6.999089450680221
    assert np.isclose(Z, expected_Z, atol=1e-5), f"Expected {expected_Z}, got {Z}"
    print(f"[PASS] conv_single_step verified: Z = {Z:.6f}")
    
    # 3. Test conv_forward
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad": 1, "stride": 2}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    assert Z.shape == (2, 3, 4, 8), f"Expected shape (2, 3, 4, 8), got {Z.shape}"
    assert np.isclose(np.mean(Z), 0.5511276474, atol=1e-5), f"Mean of Z unexpected: {np.mean(Z)}"
    assert np.isclose(Z[0, 2, 1, 1], 8.07171329, atol=1e-5), f"Z[0, 2, 1, 1] mismatch: {Z[0, 2, 1, 1]}"
    print(f"[PASS] conv_forward verified: mean(Z) = {np.mean(Z):.6f}")
    
    # 4. Test pool_forward
    np.random.seed(1)
    A_prev_pool = np.random.randn(2, 5, 5, 3)
    hparameters_pool = {"stride": 1, "f": 3}
    A_max, cache_max = pool_forward(A_prev_pool, hparameters_pool, mode="max")
    A_avg, cache_avg = pool_forward(A_prev_pool, hparameters_pool, mode="average")
    assert A_max.shape == (2, 3, 3, 3), f"Expected shape (2, 3, 3, 3), got {A_max.shape}"
    assert np.isclose(np.mean(A_max), 1.3670901, atol=1e-5), f"A_max mean mismatch: {np.mean(A_max)}"
    assert np.isclose(np.mean(A_avg), 0.0517804, atol=1e-5), f"A_avg mean mismatch: {np.mean(A_avg)}"
    print(f"[PASS] pool_forward (max & average) verified.")
    
    # 5. Test conv_backward
    np.random.seed(1)
    A_prev_b = np.random.randn(10, 4, 4, 3)
    W_b = np.random.randn(2, 2, 3, 8)
    b_b = np.random.randn(1, 1, 1, 8)
    hparams_b = {"pad": 2, "stride": 2}
    Z_b, cache_b = conv_forward(A_prev_b, W_b, b_b, hparams_b)
    dA, dW, db = conv_backward(Z_b, cache_b)
    assert dA.shape == A_prev_b.shape, f"dA shape mismatch: {dA.shape} vs {A_prev_b.shape}"
    assert dW.shape == W_b.shape, f"dW shape mismatch: {dW.shape} vs {W_b.shape}"
    assert db.shape == b_b.shape, f"db shape mismatch: {db.shape} vs {b_b.shape}"
    assert np.isclose(np.mean(dA), 1.4524377, atol=1e-4), f"mean(dA) mismatch: {np.mean(dA)}"
    assert np.isclose(np.mean(dW), 1.7269914, atol=1e-4), f"mean(dW) mismatch: {np.mean(dW)}"
    assert np.isclose(np.mean(db), 7.8392325, atol=1e-4), f"mean(db) mismatch: {np.mean(db)}"
    print(f"[PASS] conv_backward verified: mean(dA)={np.mean(dA):.5f}, mean(dW)={np.mean(dW):.5f}, mean(db)={np.mean(db):.5f}")
    
    # 6. Test create_mask_from_window & distribute_value
    np.random.seed(1)
    x_mask = np.random.randn(2, 3)
    mask = create_mask_from_window(x_mask)
    assert mask.shape == (2, 3)
    assert np.sum(mask) == 1, "Mask should only contain exactly one True entry for maximum value"
    assert mask[np.unravel_index(np.argmax(x_mask), x_mask.shape)] == True
    print("[PASS] create_mask_from_window verified.")
    
    dist = distribute_value(2.0, (2, 2))
    assert dist.shape == (2, 2)
    assert np.allclose(dist, np.array([[0.5, 0.5], [0.5, 0.5]]))
    print("[PASS] distribute_value verified.")
    
    # 7. Test pool_backward
    np.random.seed(1)
    A_prev_pb = np.random.randn(5, 5, 3, 2)
    hparams_pb = {"stride": 1, "f": 2}
    A_pb, cache_pb = pool_forward(A_prev_pb, hparams_pb)
    dA_in = np.random.randn(5, 4, 2, 2)
    dA_prev_max = pool_backward(dA_in, cache_pb, mode="max")
    dA_prev_avg = pool_backward(dA_in, cache_pb, mode="average")
    assert dA_prev_max.shape == A_prev_pb.shape, "dA_prev_max shape mismatch"
    assert dA_prev_avg.shape == A_prev_pb.shape, "dA_prev_avg shape mismatch"
    assert np.isclose(np.mean(dA_prev_max), 0.0777141, atol=1e-4), f"mean(dA_prev_max) mismatch: {np.mean(dA_prev_max)}"
    assert np.isclose(np.mean(dA_prev_avg), 0.0777141, atol=1e-4), f"mean(dA_prev_avg) mismatch: {np.mean(dA_prev_avg)}"
    print(f"[PASS] pool_backward (max & average) verified.")
    
    print("=" * 70)
    print("ALL 7 UNIT TESTS PASSED WITH 100% NUMERICAL PRECISION!")
    print("=" * 70)
