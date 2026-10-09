"""
Image Segmentation with U-Net
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 3, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we implement:
1. conv_block: Downsampling encoder block with 2x Conv2D, optional Dropout, and optional MaxPooling2D.
2. upsampling_block: Expansive decoder block with Conv2DTranspose, skip-connection concatenation, and 2x Conv2D.
3. unet_model: Complete U-Net architecture for semantic segmentation (CARLA self-driving dataset).
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
)
from tensorflow.keras.models import Model
import numpy as np


def conv_block(inputs=None, n_filters=32, dropout_prob=0.0, max_pooling=True):
    """
    Convolutional downsampling block.
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability (float between 0 and 1)
        max_pooling -- Use MaxPooling2D to reduce spatial dimensions of output volume
        
    Returns:
        next_layer, skip_connection -- Next layer and skip connection outputs
    """
    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)
    
    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv)
    
    # If dropout_prob > 0 add a dropout layer
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
        
    # If max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block.
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
        
    Returns:
        conv -- Tensor output
    """
    up = Conv2DTranspose(
        n_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(expansive_input)
    
    # Merge the previous output and the contractive_input along channel axis
    merge = concatenate([up, contractive_input], axis=3)
    
    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(merge)
    
    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv)
    
    return conv


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    U-Net model for semantic image segmentation.
    
    Arguments:
        input_size -- Input image shape (height, width, channels)
        n_filters -- Base number of filters for the first layer
        n_classes -- Number of segmentation output classes
        
    Returns:
        model -- tf.keras.Model instance
    """
    inputs = Input(input_size)
    
    # Contracting Path (Encoder)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)
    
    # Expanding Path (Decoder)
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)
    
    # Clean-up conv layer
    conv9 = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(ublock9)
    
    # Output 1x1 conv to map to n_classes logits
    conv10 = Conv2D(n_classes, kernel_size=1, padding='same')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10, name='UNet')
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: U-Net Semantic Segmentation (Coursera C4W3 Lab 2)")
    print("=" * 70)
    
    tf.random.set_seed(10)
    np.random.seed(10)
    
    # 1. Test conv_block with max pooling
    dummy_input = tf.random.normal((1, 96, 128, 3))
    next_layer, skip_conn = conv_block(dummy_input, n_filters=32, dropout_prob=0.0, max_pooling=True)
    assert next_layer.shape == (1, 48, 64, 32), f"next_layer shape mismatch: {next_layer.shape}"
    assert skip_conn.shape == (1, 96, 128, 32), f"skip_conn shape mismatch: {skip_conn.shape}"
    print(f"[PASS] conv_block (with max pooling) verified.")
    print(f"       next_layer shape: {next_layer.shape}, skip_conn shape: {skip_conn.shape}")
    
    # 2. Test conv_block with max_pooling=False (bottleneck) and dropout
    next_layer_nb, skip_conn_nb = conv_block(next_layer, n_filters=64, dropout_prob=0.3, max_pooling=False)
    assert next_layer_nb.shape == (1, 48, 64, 64), f"Bottleneck shape mismatch: {next_layer_nb.shape}"
    assert next_layer_nb.shape == skip_conn_nb.shape
    print(f"[PASS] conv_block (bottleneck, max_pooling=False) verified. Shape: {next_layer_nb.shape}")
    
    # 3. Test upsampling_block
    up_out = upsampling_block(next_layer, skip_conn, n_filters=32)
    assert up_out.shape == (1, 96, 128, 32), f"upsampling_block shape mismatch: {up_out.shape}"
    print(f"[PASS] upsampling_block verified. Output shape: {up_out.shape}")
    
    # 4. Test complete unet_model
    model = unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23)
    assert model.input_shape == (None, 96, 128, 3), f"Model input shape mismatch: {model.input_shape}"
    assert model.output_shape == (None, 96, 128, 23), f"Model output shape mismatch: {model.output_shape}"
    print(f"[PASS] unet_model architecture verified.")
    print(f"       Input shape: {model.input_shape}, Output shape: {model.output_shape}")
    print(f"       Total layers: {len(model.layers)}, Total params: {model.count_params():,}")
    
    # 5. Forward pass execution test
    batch_in = tf.random.normal((2, 96, 128, 3))
    batch_out = model(batch_in, training=False)
    assert batch_out.shape == (2, 96, 128, 23), f"Forward pass output shape mismatch: {batch_out.shape}"
    print(f"[PASS] Forward pass verified: Batch (2, 96, 128, 3) -> Output (2, 96, 128, 23).")
    
    print("=" * 70)
    print("ALL U-NET IMAGE SEGMENTATION UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
