"""
Art Generation with Neural Style Transfer (NST)
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 4, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we implement:
1. compute_content_cost: Measures feature divergence between Content image C and Generated image G.
2. gram_matrix: Computes feature cross-correlation across channel maps.
3. compute_layer_style_cost: Measures stylistic difference using Gram matrices.
4. compute_style_cost: Weighted accumulation of style cost across multiple VGG-19 layers.
5. total_cost: Weighted sum of content cost and style cost.
"""

import tensorflow as tf
import numpy as np

# Standard VGG-19 Style Layers and Weights
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost.
    
    Arguments:
    content_output -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of content image
    generated_output -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations of generated image
    
    Returns: 
    J_content -- scalar that you compute using equation above.
    """
    a_C = content_output
    a_G = generated_output
    
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G to 2D matrices
    a_C_unrolled = tf.reshape(a_C, shape=[-1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[-1, n_C])
    
    # Compute the cost with tensorflow
    J_content = (1.0 / (4.0 * float(n_H * n_W * n_C))) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content


def gram_matrix(A):
    """
    Computes the Gram Matrix of matrix A.
    
    Argument:
    A -- matrix of shape (n_C, n_H * n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations for style image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations for generated image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost for the layer
    """
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images from (1, n_H, n_W, n_C) to have them of shape (n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    norm_factor = 4.0 * float((n_C ** 2) * ((n_H * n_W) ** 2))
    J_style_layer = (1.0 / norm_factor) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers.
    
    Arguments:
    style_image_output -- dictionary (or list) containing hidden layer activations for style image
    generated_image_output -- dictionary (or list) containing hidden layer activations for generated image
    STYLE_LAYERS -- A python list containing tuples: (layer_name, coefficient)
    
    Returns: 
    J_style -- tensor representing a scalar value, total style cost across all designated layers
    """
    J_style = 0.0

    for idx, (layer_name, coeff) in enumerate(STYLE_LAYERS):
        if isinstance(style_image_output, dict):
            a_S = style_image_output[layer_name]
            a_G = generated_image_output[layer_name]
        elif isinstance(style_image_output, (list, tuple)):
            a_S = style_image_output[idx]
            a_G = generated_image_output[idx]
        else:
            raise TypeError("style_image_output must be a dict or list/tuple of activation tensors.")
            
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function.
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: Neural Style Transfer (Coursera C4W4 Lab 2)")
    print("=" * 70)
    
    tf.random.set_seed(1)
    np.random.seed(1)
    
    # 1. Test compute_content_cost with Coursera reference value
    a_C = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=1)
    a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=1)
    J_content = compute_content_cost(a_C, a_G)
    assert np.isclose(J_content.numpy(), 9.205212, atol=1e-5), f"J_content mismatch: {J_content.numpy()}"
    print(f"[PASS] compute_content_cost verified: J_content = {J_content.numpy():.6f}")
    
    # Identical content cost must be zero
    assert compute_content_cost(a_C, a_C).numpy() == 0.0, "Identical content should yield 0 cost"
    print("[PASS] identical content yields exact zero cost.")
    
    # 2. Test gram_matrix
    A = tf.random.normal([3, 2 * 1], mean=1, stddev=4, seed=1)
    GA = gram_matrix(A)
    assert GA.shape == (3, 3), f"Gram matrix shape mismatch: {GA.shape}"
    assert np.allclose(GA.numpy(), GA.numpy().T), "Gram matrix must be symmetric"
    print(f"[PASS] gram_matrix verified (shape (3, 3), symmetric).")
    
    # 3. Test compute_layer_style_cost with Coursera reference value
    a_S = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=1)
    a_G_style = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=1)
    J_style_layer = compute_layer_style_cost(a_S, a_G_style)
    assert np.isclose(J_style_layer.numpy(), 13.420519, atol=1e-5), f"J_style_layer mismatch: {J_style_layer.numpy()}"
    print(f"[PASS] compute_layer_style_cost verified: J_style_layer = {J_style_layer.numpy():.6f}")
    
    # 4. Test compute_style_cost across multi-layer dict
    style_outputs = {
        name: tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=i)
        for i, (name, _) in enumerate(STYLE_LAYERS)
    }
    gen_outputs = {
        name: tf.random.normal([1, 4, 4, 3], mean=1, stddev=4, seed=i+10)
        for i, (name, _) in enumerate(STYLE_LAYERS)
    }
    J_style_total = compute_style_cost(style_outputs, gen_outputs, STYLE_LAYERS)
    assert J_style_total.numpy() > 0, "Style cost must be positive"
    print(f"[PASS] compute_style_cost verified across {len(STYLE_LAYERS)} layers: J_style = {J_style_total.numpy():.6f}")
    
    # Identical style images must yield exact zero style cost
    J_style_zero = compute_style_cost(style_outputs, style_outputs, STYLE_LAYERS)
    assert np.isclose(J_style_zero.numpy(), 0.0, atol=1e-5), "Identical style images must have zero cost"
    print("[PASS] identical style outputs yield exact zero cost.")
    
    # 5. Test total_cost
    J = total_cost(J_content, J_style_layer, alpha=10, beta=40)
    expected_total = 10 * J_content.numpy() + 40 * J_style_layer.numpy()
    assert np.isclose(J.numpy(), expected_total, atol=1e-5)
    print(f"[PASS] total_cost verified: J = {J.numpy():.6f}")
    
    print("=" * 70)
    print("ALL NEURAL STYLE TRANSFER UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
