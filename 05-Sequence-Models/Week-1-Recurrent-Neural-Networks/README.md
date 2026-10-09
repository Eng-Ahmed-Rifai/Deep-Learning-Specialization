# Week 1: Recurrent Neural Networks (RNNs, GRUs & LSTMs)

> **Course 5: Sequence Models** — *DeepLearning.AI / Andrew Ng*  
> **Learner & Implementer:** Eng. Ahmed Rifai (100.00% Verified)

---

## 1. Why Sequence Models?

Standard deep feedforward networks and CNNs fail on sequential data (text, audio, video, time-series) due to two major limitations:
1. **Variable Input and Output Lengths:** Different sentences have different numbers of words ($T_x^{(i)} \neq T_x^{(j)}$). Standard networks require fixed-sized inputs.
2. **Failure to Share Features across Positions:** A feature learned at position $t=1$ (e.g., a named entity like "Harry") must be recognizable if it appears at position $t=15$. Stacking fully-connected layers does not share weights across different positions, leading to an intractable parameter explosion.

Recurrent Neural Networks solve both issues by maintaining an internal **hidden state** that propagates information sequentially from time step $t-1$ to time step $t$, while sharing a single set of weight matrices across all time steps.

---

## 2. Standard RNN Mathematics & Architecture

```
       y⟨1⟩              y⟨2⟩              y⟨Tx⟩
        ↑                 ↑                 ↑
     [ Wya ]           [ Wya ]           [ Wya ]
        ↑                 ↑                 ↑
a⟨0⟩ → (RNN) → a⟨1⟩  →  (RNN) → a⟨2⟩  → ... → (RNN) → a⟨Tx⟩
        ↑                 ↑                 ↑
     [ Wax ]           [ Wax ]           [ Wax ]
        ↑                 ↑                 ↑
       x⟨1⟩              x⟨2⟩              x⟨Tx⟩
```

### 2.1 Forward Propagation Equations
At each time step $t \in [1, T_x]$:

$$a^{\langle t \rangle} = g_1(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$$

$$\hat{y}^{\langle t \rangle} = g_2(W_{ya} a^{\langle t \rangle} + b_y)$$

Where:
- $g_1$ is typically the $\tanh$ activation function (or $\text{ReLU}$).
- $g_2$ is the $\text{softmax}$ (multi-class) or $\sigma$ (binary) activation function.
- $x^{\langle t \rangle} \in \mathbb{R}^{n_x \times m}$, $a^{\langle t \rangle} \in \mathbb{R}^{n_a \times m}$, $\hat{y}^{\langle t \rangle} \in \mathbb{R}^{n_y \times m}$.

### 2.2 Vectorized Concatenation Notation
To simplify equations, we define matrix $W_a = [W_{aa} \mid W_{ax}] \in \mathbb{R}^{n_a \times (n_a + n_x)}$ and vector $[a^{\langle t-1 \rangle}; x^{\langle t \rangle}] \in \mathbb{R}^{(n_a + n_x) \times m}$:

$$a^{\langle t \rangle} = \tanh\left( W_a \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} + b_a \right)$$

---

## 3. Backpropagation Through Time (BPTT) & Vanishing Gradients

### 3.1 Loss Function
For a sequence where predictions occur at every time step, total cross-entropy loss is:

$$\mathcal{L}(\hat{y}, y) = \sum_{t=1}^{T_y} \mathcal{L}^{\langle t \rangle}(\hat{y}^{\langle t \rangle}, y^{\langle t \rangle}) = -\sum_{t=1}^{T_y} \sum_{k=1}^{n_y} y_k^{\langle t \rangle} \log \hat{y}_k^{\langle t \rangle}$$

### 3.2 The Vanishing Gradient Derivation
To update weights $W_{aa}$ based on the loss at time step $T$, the chain rule traverses the entire temporal sequence:

$$\frac{\partial \mathcal{L}^{\langle T \rangle}}{\partial a^{\langle 1 \rangle}} = \frac{\partial \mathcal{L}^{\langle T \rangle}}{\partial a^{\langle T \rangle}} \cdot \prod_{j=2}^T \frac{\partial a^{\langle j \rangle}}{\partial a^{\langle j-1 \rangle}}$$

The Jacobian matrix of the transition is:

$$\frac{\partial a^{\langle j \rangle}}{\partial a^{\langle j-1 \rangle}} = \text{diag}(1 - (a^{\langle j \rangle})^2) \cdot W_{aa}^T$$

If the largest eigenvalue of $W_{aa}$ is $\lambda < 1$, as $T - 1$ grows large:

$$\left\| \prod_{j=2}^T \frac{\partial a^{\langle j \rangle}}{\partial a^{\langle j-1 \rangle}} \right\| \le \|\text{diag}(\dots)\|^{T-1} \|W_{aa}\|^{T-1} \propto \lambda^{T-1} \xrightarrow{T \to \infty} 0$$

**Consequence:**
- Gradients computed from errors late in the sequence decay exponentially as they flow backwards.
- The network cannot learn long-range temporal dependencies (e.g., agreeing subject-verb agreement: *"The **cat**, which ate the fish and drank the milk, **was** full"* vs *"The **cats** ... **were** full"*).
- **Exploding Gradients:** If $\lambda > 1$, gradients explode exponentially $\to \text{NaN}$. This is solved via **Gradient Clipping**:
  $$\text{if } \|\mathbf{g}\| > \text{threshold}: \quad \mathbf{g} \leftarrow \frac{\text{threshold}}{\|\mathbf{g}\|} \mathbf{g}$$

---

## 4. Gated Recurrent Unit (GRU)

To overcome vanishing gradients, Cho et al. (2014) introduced the GRU, which introduces an explicit **memory cell** $c$ regulated by continuous gates $\in [0, 1]$.

### 4.1 GRU Gate Formulations
In a GRU, the hidden state equals the memory cell ($a^{\langle t \rangle} = c^{\langle t \rangle}$):

1. **Reset Gate ($\Gamma_r$):** Determines how much of the past memory to forget when computing the new candidate state:
   $$\Gamma_r^{\langle t \rangle} = \sigma\left( W_r [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_r \right)$$

2. **Candidate Value ($\tilde{c}^{\langle t \rangle}$):**
   $$\tilde{c}^{\langle t \rangle} = \tanh\left( W_c [\Gamma_r^{\langle t \rangle} * c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c \right)$$

3. **Update Gate ($\Gamma_u$):** Balances past memory against candidate memory:
   $$\Gamma_u^{\langle t \rangle} = \sigma\left( W_u [c^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u \right)$$

4. **Cell Memory State ($c^{\langle t \rangle}$):**
   $$c^{\langle t \rangle} = \Gamma_u^{\langle t \rangle} * \tilde{c}^{\langle t \rangle} + (1 - \Gamma_u^{\langle t \rangle}) * c^{\langle t-1 \rangle}$$

Because $\Gamma_u$ can be close to 0, $c^{\langle t \rangle} \approx c^{\langle t-1 \rangle}$, allowing information to travel hundreds of time steps without decaying!

---

## 5. Long Short-Term Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997), the LSTM decouples the **cell state** $c^{\langle t \rangle}$ from the **hidden activation** $a^{\langle t \rangle}$ and uses three separate gating mechanisms.

```
       c⟨t-1⟩ ──────[ x ]────────────────(+)───────> c⟨t⟩
                      ↑                   ↑
                    Γf⟨t⟩           Γi⟨t⟩ * c~⟨t⟩
                      │                   │
       a⟨t-1⟩ ──┬─────┼─────────┬─────────┼──────[tanh]──[ x ]──> a⟨t⟩
                │     │         │         │                ↑
                │   [ σ ]     [ σ ]    [tanh]            Γo⟨t⟩
                │   (Wf)      (Wi)      (Wc)               │
                │     ↑         ↑         ↑              [ σ ]
                │     │         │         │              (Wo)
                └─────┴─────────┴─────────┴────────────────┘
                      ↑
                     x⟨t⟩
```

### 5.1 Full LSTM Equations

1. **Forget Gate:** Controls what information to discard from the previous cell state:
   $$\Gamma_f^{\langle t \rangle} = \sigma\left( W_f [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f \right)$$

2. **Update / Input Gate:** Controls which dimensions of the cell state receive new information:
   $$\Gamma_i^{\langle t \rangle} = \sigma\left( W_i [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_i \right)$$

3. **Candidate Memory State:**
   $$\tilde{c}^{\langle t \rangle} = \tanh\left( W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c \right)$$

4. **Cell State Update (Additive Highway):**
   $$c^{\langle t \rangle} = \Gamma_f^{\langle t \rangle} * c^{\langle t-1 \rangle} + \Gamma_i^{\langle t \rangle} * \tilde{c}^{\langle t \rangle}$$

5. **Output Gate:** Controls what portion of the internal cell state is exposed as the hidden output:
   $$\Gamma_o^{\langle t \rangle} = \sigma\left( W_o [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o \right)$$

6. **Hidden State Output:**
   $$a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle} * \tanh(c^{\langle t \rangle})$$

7. **Prediction Output:**
   $$\hat{y}^{\langle t \rangle} = \text{softmax}(W_y a^{\langle t \rangle} + b_y)$$

### 5.2 The Constant Error Carousel (CEC)
In LSTM, the gradient w.r.t previous cell state contains an additive path:

$$\frac{\partial c^{\langle t \rangle}}{\partial c^{\langle t-1 \rangle}} = \Gamma_f^{\langle t \rangle}$$

When the forget gate is saturated at $\Gamma_f \approx 1$, the gradient does not multiply through weight matrices or decaying derivatives of non-linear activations; it passes back through time unattenuated.

---

## 6. Architectural Comparison: RNN vs. GRU vs. LSTM

| Feature | Standard RNN | Gated Recurrent Unit (GRU) | Long Short-Term Memory (LSTM) |
| :--- | :--- | :--- | :--- |
| **Number of Gates** | 0 | 2 (Reset $\Gamma_r$, Update $\Gamma_u$) | 3 (Forget $\Gamma_f$, Input $\Gamma_i$, Output $\Gamma_o$) |
| **Separate Cell State?** | No ($a^{\langle t \rangle}$ only) | No ($c^{\langle t \rangle} = a^{\langle t \rangle}$) | Yes ($c^{\langle t \rangle}$ and $a^{\langle t \rangle}$) |
| **Parameter Count** | Smallest ($O(n_a^2)$) | Moderate ($3 \times$ params of RNN) | Highest ($4 \times$ params of RNN) |
| **Computational Speed** | Fastest | Faster than LSTM | Slower |
| **Long-range Retention** | Poor ($< 10$ steps) | Excellent ($> 100$ steps) | State-of-the-art ($> 100$ steps) |
| **Best Used For** | Tiny sequences, baseline benchmarks | Large datasets, limited compute | Complex sequences, hierarchical language |

---

## 7. Week 1 Implementation Labs

### 🔬 Lab 1: Building an RNN Step-by-Step
- **File:** [`Building-an-RNN-Step-by-Step/rnn_step_by_step.py`](./Building-an-RNN-Step-by-Step/rnn_step_by_step.py)
- **Status:** **100% Tested & Verified** across all 8 forward and backward functions.
- **Includes:** Standalone autograder unit tests verifying exact numerical values matching Coursera autograder specifications.

### 🦕 Lab 2: Dinosaurus Island (Character-Level Language Model)
- Character-level vocabulary dictionary mappings (`char_to_ix`, `ix_to_char`).
- Gradient clipping implementation to prevent explosive gradient collapse:
  $$\text{np.clip}(\text{gradient}, -N, N, \text{out}=\text{gradient})$$
- Generative sampling with multinomial distributions and temperature.

### 🎷 Lab 3: Jazz Improvisation with LSTM
- Generating polyphonic musical sequences using trained LSTM cells over 78 pitch classes.
- Shared weight layers across musical time steps in Keras.
