# Course 1: Neural Networks and Deep Learning

> **DeepLearning.AI / Andrew Ng**  
> Part of the [Deep Learning Specialization](../README.md)

---

## 🎯 Course Description
In this first course of the Deep Learning Specialization, you will study the foundational concept of neural networks and deep learning. By the end, you will be familiar with the significant technological trends driving the rise of deep learning, build, train, and apply fully connected deep neural networks, implement efficient (vectorized) neural networks, and identify key parameters in a neural network's architecture.

---

## 📂 Course Structure & Modules

| Module | Topic | Lab / Key Implementation | Status |
| :--- | :--- | :--- | :---: |
| **[Week 1](./Week-1-Introduction-to-Deep-Learning/)** | Introduction to Deep Learning | Structural drivers of DL, Data vs Compute scaling | Complete |
| **[Week 2](./Week-2-Neural-Networks-Basics/)** | Neural Networks Basics | [`logistic_regression_nn_mindset.py`](./Week-2-Neural-Networks-Basics/logistic_regression_nn_mindset.py) | **100% Tested** |
| **[Week 3](./Week-3-Shallow-Neural-Networks/)** | Shallow Neural Networks | 2-Layer Neural Network (Planar Data Classification) | Complete |
| **[Week 4](./Week-4-Deep-Neural-Networks/)** | Deep Neural Networks | Building your Deep Neural Network: Step by Step | Complete |

---

## 🧠 Master Mathematical Reference

### Matrix Dimension Rules (Andrew Ng's Golden Formula)
For any layer $l \in \{1, \dots, L\}$:
- **Weights:** $W^{[l]}$ has shape $(n^{[l]}, n^{[l-1]})$
- **Biases:** $b^{[l]}$ has shape $(n^{[l]}, 1)$
- **Linear intermediate:** $Z^{[l]}$ has shape $(n^{[l]}, m)$
- **Activation output:** $A^{[l]}$ has shape $(n^{[l]}, m)$
- **Gradients:** $dW^{[l]}$ has shape $(n^{[l]}, n^{[l-1]})$, $db^{[l]}$ has shape $(n^{[l]}, 1)$

### Forward Propagation Equations (Vectorized)
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$
*(where $A^{[0]} = X$)*

### Backward Propagation Equations (Vectorized)
$$dZ^{[l]} = dA^{[l]} * {g^{[l]}}'(Z^{[l]})$$
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$
$$db^{[l]} = \frac{1}{m} \text{np.sum}(dZ^{[l]}, \text{axis}=1, \text{keepdims}=\text{True})$$
$$dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$$
