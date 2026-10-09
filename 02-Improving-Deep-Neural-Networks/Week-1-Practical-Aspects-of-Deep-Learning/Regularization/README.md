# Regularization in Deep Neural Networks

## Overview
Deep neural networks with high capacity have a tendency to overfit the training dataset, especially when the number of parameters is large relative to the training samples. This module implements:
1. **L2 Regularization (Weight Decay / Frobenius Norm Regularization)**
2. **Dropout Regularization (Inverted Dropout)**

All implementations achieved **100.00% grade perfection** on the Coursera / DeepLearning.AI autograder.

---

## 1. L2 Regularization

### Cost Function
The standard cross-entropy cost function is augmented with the squared Frobenius norm of all weight matrices:
$$J_{\text{regularized}}(W, b) = J(W, b) + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$
where:
$$\|W^{[l]}\|_F^2 = \sum_{i} \sum_{j} (W_{ij}^{[l]})^2$$

### Gradient Updates
For each layer $l$, the gradient with respect to $W^{[l]}$ includes the regularization term:
$$\frac{\partial J_{\text{reg}}}{\partial W^{[l]}} = \frac{\partial J}{\partial W^{[l]}} + \frac{\lambda}{m} W^{[l]}$$
Because $\frac{\lambda}{m} W^{[l]}$ pulls weights towards zero, this is often called **weight decay**.

---

## 2. Inverted Dropout

### Forward Propagation
For each layer $l \in \{1, 2\}$, we generate a random boolean mask $D^{[l]}$ with retention probability `keep_prob`:
$$D^{[l]} = \text{np.random.rand}(A^{[l]}.\text{shape}) < \text{keep\_prob}$$
$$A^{[l]} = A^{[l]} * D^{[l]}$$
$$A^{[l]} = \frac{A^{[l]}}{\text{keep\_prob}}$$

*Note on autograder precision:* $D^{[l]}$ must retain its boolean mask datatype (`dtype=bool`) inside the cache tuple to pass deeplearning autograder assertion checks.

### Backward Propagation
During backward propagation, the gradients are shut down using the exact same mask stored in cache:
$$dA^{[l]} = dA^{[l]} * D^{[l]}$$
$$dA^{[l]} = \frac{dA^{[l]}}{\text{keep\_prob}}$$

---

## Summary of Results
| Model | Train Accuracy | Test Accuracy | Overfitting? |
| :--- | :---: | :---: | :---: |
| Baseline (Non-regularized) | 94.8% | 91.5% | High variance |
| L2 Regularization ($\lambda = 0.7$) | 93.8% | 93.0% | Balanced |
| Inverted Dropout (`keep_prob = 0.86`) | 92.9% | 95.0% | Best generalization |
