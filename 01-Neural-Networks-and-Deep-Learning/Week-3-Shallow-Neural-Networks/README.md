# Week 3: Shallow Neural Networks (One Hidden Layer)

> **Course 1: Neural Networks and Deep Learning** — *DeepLearning.AI / Andrew Ng*

---

## 1. Network Architecture (2-Layer Network)

By convention in deep learning, the input layer is layer $0$ (not counted in the network depth). A network with an input layer, one hidden layer, and an output layer is termed a **2-layer neural network**.

- **Layer 0 (Input):** $a^{[0]} = x \in \mathbb{R}^{n_x}$
- **Layer 1 (Hidden):** $n^{[1]}$ neurons. Computes $z^{[1]} = W^{[1]} a^{[0]} + b^{[1]}$ and $a^{[1]} = g^{[1]}(z^{[1]})$
- **Layer 2 (Output):** $n^{[2]} = 1$ neuron. Computes $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$ and $a^{[2]} = \sigma(z^{[2]}) = \hat{y}$

---

## 2. Activation Functions Comparison

| Activation | Formula $g(z)$ | Derivative $g'(z)$ | Typical Use Case | Pros & Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $g(z)(1 - g(z))$ | Binary output layer only | Slow convergence; vanishing gradient for $|z| > 4$. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - (g(z))^2$ | Hidden layers in shallow networks | Strictly superior to sigmoid for hidden layers because it is zero-centered (mean $\approx 0$). |
| **ReLU** | $\max(0, z)$ | $1$ if $z > 0$, else $0$ | Modern standard for hidden layers | Eliminates vanishing gradient for $z > 0$; extremely fast computation. |
| **Leaky ReLU** | $\max(0.01z, z)$ | $1$ if $z > 0$, else $0.01$ | Hidden layers with "dying ReLU" issues | Prevents dead neurons when $z < 0$. |

---

## 3. Why Random Initialization is Mandatory (Symmetry Breaking)

If you initialize weights to zero:
$$W^{[1]} = \mathbf{0}, \quad W^{[2]} = \mathbf{0}$$

Every hidden unit computes the exact same value:
$$a_1^{[1]} = a_2^{[1]} = \dots = a_{n_h}^{[1]} = g(0)$$

And during backpropagation:
$$dz_1^{[1]} = dz_2^{[1]} = \dots$$

All hidden units remain completely identical throughout training, regardless of how many neurons are added.

### Correct Initialization:
```python
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
```
*(Weights are multiplied by $0.01$ to prevent large initial activations that saturate sigmoid or tanh, which would cause tiny gradients).*
