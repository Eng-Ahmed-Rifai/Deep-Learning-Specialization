# Lab 2: Trigger Word Detection

> **Course 5: Sequence Models (Week 3, Lab 2)** — *DeepLearning.AI*

## Overview
Builds an audio wake-word detection model capable of detecting the activation keyword `"activate"` within a continuous stream of background noise and negative words.

## Pipeline & Architecture
1. **Audio Pre-processing:** Converts 10-second 44.1 kHz audio clips into spectrogram features via Short-Time Fourier Transform (STFT), yielding $T_x = 5511$ time bins with 101 frequency bands.
2. **Synthetic Data Synthesis:** Randomly inserts recordings of positive keywords and negative keywords into background recordings (cafes, streets) while recording time stamps.
3. **Target Label Smoothing:** To avoid extreme class imbalance ($y=1$ only at the exact instant the word finishes), consecutive $1$s are generated for $50$ consecutive time steps following the trigger word.
4. **Model Architecture:**
   - **1D Convolutional Layer:** Stride $s=4$ reduces temporal sequence length from $5511$ down to $T_y = 1375$ time steps while extracting spectral frequency patterns.
   - **Batch Normalization & Dropout (0.8).**
   - **2x GRU Layers:** Recurrent units processing temporal context.
   - **TimeDistributed Dense Layer with Sigmoid:** Emits wake-word probability $\in [0, 1]$ at every time step.
