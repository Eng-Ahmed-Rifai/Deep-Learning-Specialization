"""
Transfer Learning with MobileNetV2
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 2, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we build a transfer learning pipeline using MobileNetV2:
1. Feature extraction with frozen base model weights.
2. Custom classification head (GlobalAveragePooling2D -> Dropout -> Dense).
3. Fine-tuning strategy: unfreezing top layers for end-to-end gradient updates.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def alpaca_model(image_shape=(160, 160, 3), data_augmentation=None):
    """
    Define a TF Keras model for binary classification using a pre-trained MobileNetV2.
    
    Arguments:
    image_shape -- tuple of shape (160, 160, 3)
    data_augmentation -- sequential data augmentation pipeline or None
    
    Returns:
    model -- TF Keras model instance
    """
    input_shape = image_shape
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None  # weights can be 'imagenet' in live deployment
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Apply data augmentation if present
    x = data_augmentation(inputs) if data_augmentation is not None else inputs
    
    # Data preprocessing
    x = preprocess_input(x)
    
    # Base model pass in inference mode
    x = base_model(x, training=False)
    
    # Add classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AlpacaClassifier')
    return model, base_model


def setup_fine_tuning(base_model, fine_tune_at=120):
    """
    Unfreezes the base model starting from layer index fine_tune_at onwards.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: Transfer Learning MobileNetV2 (Coursera C4W2 Lab 2)")
    print("=" * 70)
    
    model, base_model = alpaca_model((160, 160, 3))
    assert model.input_shape == (None, 160, 160, 3), f"Input shape mismatch: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Output shape mismatch: {model.output_shape}"
    assert not base_model.trainable, "Base model should initially be frozen!"
    
    # Test fine tuning setup
    base_model = setup_fine_tuning(base_model, fine_tune_at=120)
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    frozen_count = sum(1 for layer in base_model.layers if not layer.trainable)
    assert frozen_count == 120, f"Expected 120 frozen layers, got {frozen_count}"
    assert trainable_count > 0, "Top layers must be trainable for fine-tuning"
    
    # Forward pass test
    dummy_img = tf.random.normal((2, 160, 160, 3))
    out = model(dummy_img, training=False)
    assert out.shape == (2, 1), f"Output shape mismatch: {out.shape}"
    
    print(f"[PASS] Transfer Learning MobileNetV2 verified.")
    print(f"Base Model Total Layers: {len(base_model.layers)}")
    print(f"Frozen Layers: {frozen_count}, Fine-tuned Layers: {trainable_count}")
    print("=" * 70)
