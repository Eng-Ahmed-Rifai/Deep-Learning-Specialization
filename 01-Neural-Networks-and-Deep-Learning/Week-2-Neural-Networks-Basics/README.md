# Week 2: Neural Networks Basics & Logistic Regression Mindset

> **Course 1: Neural Networks and Deep Learning** — *DeepLearning.AI / Andrew Ng*  
> Implementation: [`logistic_regression_nn_mindset.py`](./logistic_regression_nn_mindset.py)

---

## 1. Overview & Learning Objectives

Week 2 introduces the fundamental building blocks of modern deep learning by framing **Logistic Regression** not merely as an econometric or statistical regression technique, but as the simplest possible neural network: a **single neuron** with a sigmoid activation function.

### Key Objectives:
- Master matrix dimensions and vectorized notation for training sets.
- Understand forward propagation and backward propagation via computation graphs.
- Derive the binary cross-entropy loss function from Maximum Likelihood Estimation (MLE).
- Apply calculus chain rule to derive analytical gradients without manual loops.
- Leverage NumPy vectorization (SIMD) and internalize broadcasting rules.
- Eliminate Rank-1 arrays to prevent silent dimensional bugs.

---

## 2. Mathematical Notation & Data Representation

In deep learning notation established by Andrew Ng, a dataset of $m$ training examples is arranged with features along rows and training examples along columns:

| Notation | Dimension | Description |
| :--- | :--- | :--- |
| $n_x$ (or $n$) | $\mathbb{N}$ | Number of input features per example (e.g. $64 \times 64 \times 3 = 12,288$ for RGB images). |
| $m$ | $\mathbb{N}$ | Number of training examples in the batch. |
| $X$ | $\mathbb{R}^{n_x \times m}$ | Input feature matrix: $X = \begin{bmatrix} \vert & \vert & & \vert \\ x^{(1)} & x^{(2)} & \dots & x^{(m)} \\ \vert & \vert & & \vert \end{bmatrix}$ |
| $Y$ | $\mathbb{R}^{1 \times m}$ | Ground truth labels vector: $Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \dots & y^{(m)} \end{bmatrix}, \quad y^{(i)} \in \{0, 1\}$ |
| $w$ | $\mathbb{R}^{n_x \times 1}$ | Weight parameter vector (column vector). |
| $b$ | $\mathbb{R}$ (scalar) | Bias parameter. |
| $A$ | $\mathbb{R}^{1 \times m}$ | Model activation probabilities: $A = \begin{bmatrix} a^{(1)} & a^{(2)} & \dots & a^{(m)} \end{bmatrix}$ |

---

## 3. Logistic Regression as a Single Neuron

A single artificial neuron computes a linear transformation followed by a non-linear activation:

```
Inputs: x_1, x_2, ..., x_n
           \   |   /
            v  v  v
      [ z = w^T x + b ]  -->  Linear step
             |
             v
      [ a = sigma(z)  ]  -->  Activation step
             |
             v
         Output: a = y_hat
```

### 3.1 The Sigmoid Activation Function

The sigmoid function maps any real number $\mathbb{R}$ to the open interval $(0, 1)$, allowing the output to be interpreted as a conditional probability $\hat{y} = P(y = 1 \mid x)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Key Properties:
- As $z \to \infty$, $\sigma(z) \to 1$.
- As $z \to -\infty$, $\sigma(z) \to 0$.
- At $z = 0$, $\sigma(0) = \frac{1}{1 + 1} = 0.5$.

#### Derivative of Sigmoid:
$$\frac{d\sigma(z)}{dz} = \frac{d}{dz}\left((1 + e^{-z})^{-1}\right) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
$$= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{1 + e^{-z}} \cdot \left(1 - \frac{1}{1 + e^{-z}}\right) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$$

---

## 4. Loss Function Derivation (Maximum Likelihood Estimation)

### 4.1 Why Not Mean Squared Error (MSE)?
In linear regression, MSE $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$ is convex. However, when $\hat{y} = \sigma(w^T x + b)$, substituting sigmoid into MSE yields a **non-convex** optimization landscape populated with local minima and flat saddle regions where gradient descent gets trapped.

### 4.2 Derivation via Maximum Likelihood Estimation (MLE)
The conditional probability distribution for binary label $y \in \{0, 1\}$ given input $x$ is modeled as a Bernoulli distribution:

$$P(y \mid x) = \hat{y}^y \cdot (1 - \hat{y})^{1 - y}$$

- If $y = 1$: $P(y = 1 \mid x) = \hat{y}$
- If $y = 0$: $P(y = 0 \mid x) = 1 - \hat{y}$

Taking the natural logarithm (log-likelihood):

$$\log P(y \mid x) = y \log \hat{y} + (1 - y) \log (1 - \hat{y})$$

To minimize loss via gradient descent instead of maximizing likelihood, we define the **Loss function** $\mathcal{L}(a, y)$ as the negative log-likelihood:

$$\mathcal{L}(a, y) = -\left[ y \log a + (1 - y) \log(1 - a) \right]$$

### 4.3 The Overall Cost Function $J(w, b)$
The overall cost $J(w, b)$ is the empirical average loss across all $m$ training examples:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)}) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log a^{(i)} + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

This cost function is strictly **convex**, ensuring that gradient descent converges globally to the global minimum.

---

## 5. Computation Graph & Analytical Backpropagation

Backpropagation calculates the partial derivatives of the cost function with respect to model parameters using the multivariable calculus chain rule.

### 5.1 Step-by-Step Chain Rule Derivation

```mermaid
flowchart LR
    A["x, w, b"] --> B["z = w^T x + b"]
    B --> C["a = sigma(z)"]
    C --> D["L(a, y) = -[y log a + (1-y) log(1-a)]"]
    D -.->|"dL/da = -y/a + (1-y)/(1-a)"| C
    C -.->|"da/dz = a(1-a)"| B
    B -.->|"dL/dz = a - y"| A
```

#### Step 1: Derivative of Loss with respect to Activation $a$
$$\frac{\partial \mathcal{L}}{\partial a} = -\left( \frac{y}{a} - \frac{1 - y}{1 - a} \right) = -\frac{y(1 - a) - a(1 - y)}{a(1 - a)} = \frac{a - y}{a(1 - a)}$$

#### Step 2: Derivative of Activation $a$ with respect to Linear output $z$
$$\frac{\partial a}{\partial z} = a(1 - a)$$

#### Step 3: Derivative of Loss with respect to $z$ ($dz$)
Using the Chain Rule:
$$dz = \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} = \frac{a - y}{a(1 - a)} \cdot a(1 - a) = a - y$$

*Notice the elegant cancellation of terms! The gradient $dz$ is simply the prediction error: $a - y$.*

#### Step 4: Derivatives with respect to parameters $w$ and $b$
Since $z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$:
$$\frac{\partial z}{\partial w} = x \implies dw = \frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w} = x \cdot (a - y) = x \cdot dz$$

$$\frac{\partial z}{\partial b} = 1 \implies db = \frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial b} = 1 \cdot (a - y) = dz$$

---

## 6. Full Vectorization: Eliminating Explicit Loops

In classic CPU programming, one might write nested for-loops over all $m$ examples and $n_x$ features. In deep learning, vectorized operations exploit **SIMD (Single Instruction, Multiple Data)** parallel execution on modern CPUs (AVX-512) and GPUs (CUDA cores).

### 6.1 Vectorized Forward Propagation

$$\begin{aligned}
Z &= w^T X + b \\
&= \begin{bmatrix} w_1 & \dots & w_{n_x} \end{bmatrix} \begin{bmatrix} x_1^{(1)} & \dots & x_1^{(m)} \\ \vdots & \ddots & \vdots \\ x_{n_x}^{(1)} & \dots & x_{n_x}^{(m)} \end{bmatrix} + \begin{bmatrix} b & \dots & b \end{bmatrix} \\
&\in \mathbb{R}^{1 \times m}
\end{aligned}$$

$$A = \sigma(Z) \in \mathbb{R}^{1 \times m}$$

$$\text{cost} = -\frac{1}{m} \left( Y \log(A)^T + (1 - Y) \log(1 - A)^T \right) = -\frac{1}{m} \sum \left( Y * \log(A) + (1 - Y) * \log(1 - A) \right)$$

### 6.2 Vectorized Backward Propagation

$$dZ = A - Y = \begin{bmatrix} a^{(1)} - y^{(1)} & a^{(2)} - y^{(2)} & \dots & a^{(m)} - y^{(m)} \end{bmatrix} \in \mathbb{R}^{1 \times m}$$

$$dw = \frac{1}{m} X \, dZ^T = \frac{1}{m} \begin{bmatrix} \vert & \dots & \vert \\ x^{(1)} & \dots & x^{(m)} \\ \vert & \dots & \vert \end{bmatrix} \begin{bmatrix} dz^{(1)} \\ dz^{(2)} \\ \vdots \\ dz^{(m)} \end{bmatrix} \in \mathbb{R}^{n_x \times 1}$$

$$db = \frac{1}{m} \sum_{i=1}^m dZ^{(i)} = \frac{1}{m} \text{np.sum}(dZ) \in \mathbb{R} \text{ (scalar)}$$

---

## 7. NumPy Broadcasting Rules & Rank-1 Safety

### 7.1 The Two Rules of NumPy Broadcasting:
Two array dimensions are compatible for element-wise operations when:
1. They are **equal**, OR
2. One of the dimensions is **1**.

When one dimension is 1, NumPy automatically duplicates/stretches that axis along the other array's dimension without copying data in memory.

#### Example in Logistic Regression:
$$w^T X \quad \text{has shape } (1, m)$$
$$b \quad \text{has shape } (1, 1) \text{ or scalar } ()$$
NumPy broadcasts $b$ horizontally across all $m$ columns:
$$\begin{bmatrix} z_1 & z_2 & \dots & z_m \end{bmatrix} + b = \begin{bmatrix} z_1 + b & z_2 + b & \dots & z_m + b \end{bmatrix}$$

### 7.2 The Rank-1 Array Trap:
In Python, arrays created like `a = np.random.randn(5)` have shape `(5,)`. These are **rank-1 arrays**, behaving neither as true row vectors nor as column vectors:

```python
# DANGEROUS: Rank-1 Array
a = np.random.randn(5)      # a.shape is (5,)
print(a.T.shape)            # Still (5,)! Transpose has no effect!
print(np.dot(a, a.T))       # Outputs a scalar float instead of a (5, 5) matrix!

# SAFE & MANDATORY PRACTICE: Explicit 2D Vectors
a = np.random.randn(5, 1)   # Explicit column vector (5, 1)
print(a.T.shape)            # (1, 5) row vector!
print(np.dot(a, a.T).shape) # (5, 5) outer product matrix!
assert a.shape == (5, 1)    # Andrew Ng's Defensive Programming Rule
```

---

## 8. Summary of the Gradient Descent Optimization Loop

For each iteration $k = 1, \dots, \text{num\_iterations}$:

1. **Forward Propagation:**
   $$A = \sigma(w^T X + b)$$
   $$J = -\frac{1}{m} \sum \left( Y \log A + (1 - Y) \log (1 - A) \right)$$

2. **Backward Propagation:**
   $$dZ = A - Y$$
   $$dw = \frac{1}{m} X dZ^T$$
   $$db = \frac{1}{m} \sum dZ$$

3. **Parameter Update:**
   $$w := w - \alpha \cdot dw$$
   $$b := b - \alpha \cdot db$$
   *(where $\alpha$ is the learning rate)*

---

## 9. Verification & Coursera Test Cases

All functions implemented in `logistic_regression_nn_mindset.py` are verified against the exact unit test assertions of the Coursera AutoGrader:

| Function | Coursera Test Inputs | Expected Result | Status |
| :--- | :--- | :--- | :--- |
| `sigmoid(z)` | `z = [0, 2]` | `[0.5, 0.88079708]` | **PASSED** |
| `initialize_with_zeros(dim)` | `dim = 2` | `w.shape = (2, 1), b = 0.0` | **PASSED** |
| `propagate(w, b, X, Y)` | `w=[[1],[2]], b=2, X=..., Y=...` | `cost=5.80154532, dw=[[0.998],[2.395]], db=0.001455` | **PASSED** |
| `optimize(w, b, X, Y)` | `100 iters, lr=0.009` | `w=[[0.190],[0.122]], b=1.925359, cost=5.801545` | **PASSED** |
| `predict(w, b, X)` | `w=[[0.112],[0.231]], b=-0.3` | `predictions = [[1., 1., 0.]]` | **PASSED** |
| `model(...)` | Synthetic Binary Dataset | Monotonic loss decrease, Train > 95%, Test > 90% | **PASSED** |
| **Vectorization Benchmark** | $1,000 \text{ features} \times 100,000 \text{ samples}$ | **>1500x speedup** vs explicit for-loop | **PASSED** |
