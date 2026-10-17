# 🧠 Deep Learning Specialization — Master Study Guide

**Author:** Eng. Ahmed Rifai  
**Instructor:** Andrew Ng (DeepLearning.AI / Stanford University)  
**Target:** First-Principles Mathematical Formulations, Backpropagation Derivations, Optimization Equations, Computer Vision Architectures, and Sequence Transformers.

---

## 📑 Table of Contents
1. [Course 1: Neural Networks and Deep Learning](#course-1-neural-networks-and-deep-learning)
2. [Course 2: Improving Deep Neural Networks (Hyperparameters, Tuning & Regularization)](#course-2-improving-deep-neural-networks)
3. [Course 3: Structuring Machine Learning Projects](#course-3-structuring-machine-learning-projects)
4. [Course 4: Convolutional Neural Networks (CNNs)](#course-4-convolutional-neural-networks)
5. [Course 5: Sequence Models & Transformer Architectures](#course-5-sequence-models--transformer-architectures)

---

## Course 1: Neural Networks and Deep Learning

### 1. Vectorized Forward Propagation
For layer $l \in [1, L]$ with $m$ training examples:
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$
where:
- $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
- $b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$ (broadcast across $m$ columns)
- $A^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$, with $A^{[0]} = X \in \mathbb{R}^{n_x \times m}$.

### 2. Loss & Cost Functions
- **Binary Cross-Entropy (Logistic Regression)**:
  $$\mathcal{L}(a, y) = -(y \log a + (1 - y) \log (1 - a))$$
  $$J(W, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$$

### 3. Vectorized Backward Propagation
Chain rule equations computed from output layer backwards:
$$dZ^{[L]} = A^{[L]} - Y$$
For hidden layer $l = L-1, \dots, 1$:
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$
$$db^{[l]} = \frac{1}{m} \sum_{i=1}^m dZ^{[l](i)} \quad (\text{axis}=1, \text{keepdims}=\text{True})$$
$$dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$$
$$dZ^{[l-1]} = dA^{[l-1]} * {g^{[l-1]}}'(Z^{[l-1]})$$

---

## Course 2: Improving Deep Neural Networks

### 1. Weight Initialization
- **Vanishing/Exploding Gradients**: Deep networks suffer when weights are too large or too small.
- **Xavier (Glorot) Initialization** (for Tanh): $\text{Var}(W) = \frac{1}{n_{in}}$ or $\frac{2}{n_{in} + n_{out}}$.
- **He Initialization** (for ReLU):
  $$W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{[l-1]}}}\right)$$

### 2. Regularization Strategies
- **L2 Regularization (Weight Decay)**:
  $$J_{reg} = J + \frac{\lambda}{2m} \sum_l \|W^{[l]}\|_F^2$$
  $$dW^{[l]} = dW^{[l]} + \frac{\lambda}{m} W^{[l]}$$
- **Inverted Dropout**:
  During training:
  ```python
  d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
  a3 = (a3 * d3) / keep_prob  # Scaled by keep_prob to keep expected value intact
  ```
  During test/inference: No dropout applied.

### 3. Optimization Algorithms
- **Momentum**: Exponentially weighted averages of past gradients:
  $$v_{dW} = \beta v_{dW} + (1 - \beta) dW, \quad W = W - \alpha v_{dW}$$
- **RMSprop**: Scales learning rate inversely with root mean square of gradients:
  $$s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) (dW)^2, \quad W = W - \frac{\alpha}{\sqrt{s_{dW}} + \epsilon} dW$$
- **Adam (Adaptive Moment Estimation)**:
  Combines Momentum ($\beta_1 \approx 0.9$) and RMSprop ($\beta_2 \approx 0.999$) with bias correction:
  $$v_{corrected} = \frac{v}{1 - \beta_1^t}, \quad s_{corrected} = \frac{s}{1 - \beta_2^t}$$
  $$W = W - \frac{\alpha}{\sqrt{s_{corrected}} + \epsilon} v_{corrected}$$

### 4. Batch Normalization
Normalizes the pre-activations $Z^{[l]}$ across the mini-batch:
$$\mu = \frac{1}{m} \sum z^{(i)}, \quad \sigma^2 = \frac{1}{m} \sum (z^{(i)} - \mu)^2$$
$$z_{norm}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\tilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$$
Learned parameters $\gamma$ and $\beta$ allow the network to represent any activation scale.

---

## Course 3: Structuring Machine Learning Projects

### 1. Orthogonalization & Evaluation Metrics
- Single-number evaluation metric (e.g., F1-score combining precision and recall).
- **Satisficing vs Optimizing Metrics**: Maximize accuracy (optimizing) subject to inference latency $< 100\text{ms}$ (satisficing).

### 2. Bias-Variance Diagnosis & Human-Level Performance (HLP)
- **Bayes Optimal Error** $\approx$ Human-Level Performance (HLP).
- **Avoidable Bias**: $\text{Training Error} - \text{HLP}$. Fix via: bigger model, longer training, advanced optimization.
- **Variance**: $\text{Dev Error} - \text{Training Error}$. Fix via: regularization (L2, dropout), more training data, data augmentation.
- **Data Mismatch**: $\text{Train-Dev Error}$ vs $\text{Dev Error}$ (occurs when training on web images and deploying on mobile user photos).

---

## Course 4: Convolutional Neural Networks (CNNs)

### 1. ConvNet Foundations
- Output dimensions for input $(n_H \times n_W)$ with filter $(f \times f)$, padding $p$, and stride $s$:
  $$n_H^{out} = \left\lfloor \frac{n_H + 2p - f}{s} \right\rfloor + 1$$
- Total parameters in a convolutional layer with $K$ filters of size $(f \times f \times n_C)$:
  $$\text{Params} = (f \cdot f \cdot n_C + 1) \cdot K$$

### 2. Classic & Modern Architectures
- **LeNet-5 & AlexNet**: Pioneered Conv-Pool-Conv-Pool-FC pipelines.
- **VGG-16**: Simplicity of design using exclusively $3 \times 3$ convolutions with stride 1 and $2 \times 2$ max pool.
- **ResNet (He et al.)**: Skip connections bypass 2 or more layers:
  $$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$
  Ensures gradients propagate smoothly, enabling training of networks with $>100$ layers.
- **Inception (GoogLeNet)**: Computes $1 \times 1, 3 \times 3, 5 \times 5$ convolutions and max pooling in parallel. Uses $1 \times 1$ convolutions for dimensionality reduction (bottleneck layers).

### 3. Object Detection (YOLO — You Only Look Once)
- Divides image into $S \times S$ grid cells.
- Target vector per grid cell: $y = [p_c, b_x, b_y, b_h, b_w, c_1, c_2, \dots, c_k]^T$.
- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
- **Non-Max Suppression (NMS)**: Eliminates duplicate overlapping bounding boxes for the same object based on class probability scores and IoU threshold ($>0.5$).
- **Anchor Boxes**: Enables grid cells to detect multiple overlapping objects of different aspect ratios.

---

## Course 5: Sequence Models & Transformer Architectures

### 1. Recurrent Architectures (RNN, GRU, LSTM)
- **Vanilla RNN**: $a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$.
- **Long Short-Term Memory (LSTM)**: Solves vanishing gradients across long sequences via dedicated cell state $c^{\langle t \rangle}$ controlled by 3 gates:
  - **Forget Gate**: $\Gamma_f = \sigma(W_f [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f)$
  - **Update Gate**: $\Gamma_u = \sigma(W_u [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u)$
  - **Candidate State**: $\tilde{c}^{\langle t \rangle} = \tanh(W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)$
  - **Cell Update**: $c^{\langle t \rangle} = \Gamma_f * c^{\langle t-1 \rangle} + \Gamma_u * \tilde{c}^{\langle t \rangle}$
  - **Output Gate**: $\Gamma_o = \sigma(W_o [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o), \quad a^{\langle t \rangle} = \Gamma_o * \tanh(c^{\langle t \rangle})$

### 2. Self-Attention & Transformers
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- Eliminates sequential recurrent bottlenecks, enabling full sequence parallelization during training.
