# Course 4: Convolutional Neural Networks

> **DeepLearning.AI / Andrew Ng**  
> Part of the [Deep Learning Specialization](../README.md)  
> Implementation & Verification by: **Eng. Ahmed Rifai (100.00% Verified)**

---

## 🎯 Course Description
In this fourth course of the Deep Learning Specialization, you will understand how computer vision has evolved and become familiar with its exciting applications such as autonomous driving, face recognition, reading radiology images, and neural style transfer.

By the end of this course, you will be able to:
- Build a Convolutional Neural Network from scratch, including convolution and pooling layers (forward and backward passes).
- Implement multi-class image classification with modern deep architectures.
- Use Residual Networks (ResNets) to optimize networks beyond 50 layers without degradation.
- Understand and implement state-of-the-art object detection algorithms like YOLO (You Only Look Once) using anchor boxes and non-max suppression.
- Apply Siamese networks and Triplet Loss to one-shot face recognition and verification.
- Synthesize artistic images using Neural Style Transfer with Gram matrices and deep feature representations.

---

## 📂 Course Structure & Modules

| Module | Topic | Lab / Key Implementation | Status |
| :--- | :--- | :--- | :---: |
| **[Week 1](./Week-1-Foundations-of-CNNs/)** | Foundations of CNNs | [`cnn_step_by_step.py`](./Week-1-Foundations-of-CNNs/Convolution-Model-Step-by-Step/cnn_step_by_step.py) | **100% Tested** |
| **[Week 2](./Week-2-Deep-Convolutional-Models/)** | Deep Convolutional Models | [`resnets.py`](./Week-2-Deep-Convolutional-Models/ResNets/resnets.py) (ResNet-50) | **100% Tested** |
| **[Week 3](./Week-3-Object-Detection/)** | Object Detection | YOLO Car Detection & Non-Max Suppression | **Complete** |
| **[Week 4](./Week-4-Special-Applications/)** | Special Applications | FaceNet Triplet Loss & Neural Style Transfer | **Complete** |

---

## 🧠 Master Mathematical Reference

### 1. Convolution Output Dimensions
For input $(n_H^{[l-1]}, n_W^{[l-1]})$, filter size $f$, padding $p$, and stride $s$:

$$n_H^{[l]} = \left\lfloor \frac{n_H^{[l-1]} + 2p - f}{s} \right\rfloor + 1, \quad n_W^{[l]} = \left\lfloor \frac{n_W^{[l-1]} + 2p - f}{s} \right\rfloor + 1$$

### 2. Residual Connection Formulation
$$\mathcal{H}(x) = \mathcal{F}(x) + x \implies a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$

### 3. YOLO Objective & Loss
$$y = [p_c, b_x, b_y, b_h, b_w, c_1, \dots, c_C]^T$$

### 4. Triplet Loss (FaceNet)
$$\mathcal{L}(A, P, N) = \max\left(\|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha, 0\right)$$

### 5. Neural Style Transfer Total Loss
$$J(G) = \alpha J_{\text{content}}(C, G) + \beta J_{\text{style}}(S, G)$$
