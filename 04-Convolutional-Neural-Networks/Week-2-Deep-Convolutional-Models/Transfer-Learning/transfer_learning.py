"""
Transfer Learning with MobileNetV2
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 2, Lab 2)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we build a transfer learning pipeline using MobileNetV2:
1. data_augmenter: Sequential pipeline for data augmentation (RandomFlip + RandomRotation).
2. alpaca_model: Transfer learning model using pre-trained MobileNetV2 with custom classification head.
3. setup_fine_tuning: Fine-tuning strategy unfreezing top layers for end-to-end gradient updates.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, RandomFlip, RandomRotation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (160, 160)


def data_augmenter():
    """
    Create a Sequential model composed of 2 layers for data augmentation.
    
    Returns:
        tf.keras.Sequential
    """
    data_augmentation = Sequential(name='data_augmentation')
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    return data_augmentation


def alpaca_model(image_shape=IMG_SIZE, data_augmentation=None, weights=None):
    """
    Define a TF Keras model for binary classification using a pre-trained MobileNetV2.
    
    Arguments:
    image_shape -- tuple of shape (160, 160) or (160, 160, 3)
    data_augmentation -- sequential data augmentation pipeline or None
    weights -- base model weights ('imagenet' or None)
    
    Returns:
    model -- TF Keras model instance
    """
    input_shape = image_shape + (3,) if len(image_shape) == 2 else image_shape
    
    if data_augmentation is None:
        data_augmentation = data_augmenter()
        
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Apply data augmentation
    x = data_augmentation(inputs)
    
    # Data preprocessing
    x = preprocess_input(x)
    
    # Base model pass in inference mode
    x = base_model(x, training=False)
    
    # Add classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='AlpacaClassifier')
    model.base_model = base_model
    return model


def setup_fine_tuning(base_model, fine_tune_at=120):
    """
    Unfreezes the base model starting from layer index fine_tune_at onwards.
    
    Arguments:
    base_model -- base convolutional network (MobileNetV2)
    fine_tune_at -- layer index from which layers are trained
    
    Returns:
    base_model -- updated model with configured trainability
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: Transfer Learning MobileNetV2 (Coursera C4W2 Lab 2)")
    print("=" * 70)
    
    # 1. Test data_augmenter
    augmenter = data_augmenter()
    assert len(augmenter.layers) == 2, f"Expected 2 layers in augmenter, got {len(augmenter.layers)}"
    assert isinstance(augmenter.layers[0], RandomFlip), "First layer must be RandomFlip"
    assert isinstance(augmenter.layers[1], RandomRotation), "Second layer must be RandomRotation"
    print(f"[PASS] data_augmenter verified (RandomFlip + RandomRotation).")
    
    # 2. Test alpaca_model with IMG_SIZE
    model = alpaca_model(image_shape=IMG_SIZE, data_augmentation=augmenter, weights=None)
    assert model.input_shape == (None, 160, 160, 3), f"Input shape mismatch: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Output shape mismatch: {model.output_shape}"
    base_model = model.base_model
    assert not base_model.trainable, "Base model should initially be frozen!"
    print(f"[PASS] alpaca_model constructed successfully with frozen MobileNetV2 base.")
    
    # 3. Test fine-tuning configuration
    base_model = setup_fine_tuning(base_model, fine_tune_at=120)
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    frozen_count = sum(1 for layer in base_model.layers if not layer.trainable)
    assert frozen_count == 120, f"Expected 120 frozen layers, got {frozen_count}"
    assert trainable_count > 0, "Top layers must be trainable for fine-tuning"
    print(f"[PASS] setup_fine_tuning verified: {frozen_count} frozen, {trainable_count} trainable.")
    
    # 4. Forward pass test
    dummy_img = tf.random.normal((2, 160, 160, 3))
    out = model(dummy_img, training=False)
    assert out.shape == (2, 1), f"Output shape mismatch: {out.shape}"
    print(f"[PASS] Forward pass verified: Batch (2, 160, 160, 3) -> Output (2, 1).")
    
    print("=" * 70)
    print("ALL TRANSFER LEARNING UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
