"""
Convolution Model Application (Keras / TensorFlow)
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 1, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we build and train a ConvNet in TensorFlow for the SIGNS dataset:
CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE (Softmax)
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np


def convolutional_model(input_shape=(64, 64, 3), classes=6):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    input_shape -- shape of the images of the dataset: (64, 64, 3)
    classes -- number of classification categories (6 for SIGNS dataset)
    
    Returns:
    model -- TF Keras model (object instance)
    """
    input_img = Input(shape=input_shape)
    
    # CONV2D: 8 filters 4x4, stride with 1, padding 'SAME'
    Z1 = Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_img)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(Z1)
    
    # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(P1)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(Z2)
    
    # FLATTEN
    F = Flatten()(P2)
    
    # DENSE: 6 neurons with softmax activation
    outputs = Dense(units=classes, activation='softmax')(F)
    
    model = Model(inputs=input_img, outputs=outputs, name='ConvolutionalModel')
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: Convolution Model Application (Coursera C4W1 Lab 2)")
    print("=" * 70)
    
    conv_model = convolutional_model((64, 64, 3), classes=6)
    assert conv_model.input_shape == (None, 64, 64, 3), f"Input shape mismatch: {conv_model.input_shape}"
    assert conv_model.output_shape == (None, 6), f"Output shape mismatch: {conv_model.output_shape}"
    
    # Forward pass test
    test_data = tf.random.normal((4, 64, 64, 3))
    preds = conv_model(test_data)
    assert preds.shape == (4, 6), f"Preds shape mismatch: {preds.shape}"
    assert np.allclose(np.sum(preds.numpy(), axis=1), [1.0, 1.0, 1.0, 1.0], atol=1e-5), "Softmax output must sum to 1.0"
    
    print(f"[PASS] Convolutional Model verified.")
    print(f"Total Parameters: {conv_model.count_params():,}")
    print("=" * 70)
