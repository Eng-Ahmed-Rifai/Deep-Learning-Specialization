# Week 4: Transformers & Large Language Model Pre-training

> **Course 5: Sequence Models** — *DeepLearning.AI / Andrew Ng*  
> **Learner & Implementer:** Eng. Ahmed Rifai (100.00% Verified)

---

## 1. The Paradigm Shift: Why Transformers Replaced RNNs

For decades, sequential deep learning was dominated by recurrent networks (RNN, GRU, LSTM). However, RNNs suffered from two fundamental architectural bottlenecks:

1. **Sequential Computation Bottleneck:**
   RNNs require computing step $t-1$ before step $t$. This inherently sequential dependency prevents hardware parallelization during training across the sequence dimension ($T$), making large-scale GPU scaling impossible.
2. **Information Path Length:**
   In an RNN, information from token $1$ must traverse $N-1$ recurrent steps to reach token $N$. Even with gating (LSTM), gradients and signals degrade across $O(N)$ steps.
   - **Transformer:** In Self-Attention, every token directly connects to every other token in a single operation $\to \mathbf{O(1)}$ **constant path length**!

| Architecture | Path Length for Dependencies | Parallel Training? | Complexity per Layer |
| :--- | :---: | :---: | :---: |
| **Recurrent (LSTM/GRU)** | $O(N)$ | ❌ No ($O(N)$ sequential steps) | $O(N \cdot d^2)$ |
| **Convolutional (Conv1D)** | $O(\log_k N)$ | ✔️ Yes | $O(k \cdot N \cdot d^2)$ |
| **Transformer Self-Attention**| $\mathbf{O(1)}$ | ✔️ **Full GPU Parallelism** | $O(N^2 \cdot d)$ |

---

## 2. Scaled Dot-Product Attention

In the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017), self-attention computes relationships among tokens using three projected matrices:
- **Queries ($Q \in \mathbb{R}^{N \times d_k}$):** What each token is asking for.
- **Keys ($K \in \mathbb{R}^{M \times d_k}$):** What each token offers as content/index.
- **Values ($V \in \mathbb{R}^{M \times d_v}$):** The actual semantic representation to be extracted.

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

```
   Q        K
   │        │
   └───┬────┘
       ↓
    [ Q K^T ]  (Similarity Matrix: N x M)
       ↓
    [ / √dk ]  (Scale to prevent saturation)
       ↓
    [  + M  ]  (Optional Causal / Padding Mask)
       ↓
   [ Softmax ] (Attention Weights: Rows sum to 1)
       ↓
    [  * V  ]  (Weighted Value Aggregation)
       ↓
     Output
```

### 2.1 Mathematical Derivation of the Scaling Factor $\frac{1}{\sqrt{d_k}}$
Why scale by $\sqrt{d_k}$?
Assume elements of $q$ and $k$ are independent random variables with mean $0$ and variance $1$:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}[q_i] = 1, \quad \mathbb{E}[k_i] = 0, \quad \text{Var}[k_i] = 1$$

Their dot product is $z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. The mean and variance of $z$ are:

$$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

$$\text{Var}[z] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - (\mathbb{E}[q_i]\mathbb{E}[k_i])^2 \right) = \sum_{i=1}^{d_k} (1 \cdot 1 - 0) = d_k$$

As the dimension $d_k$ grows large (e.g., $d_k = 64$ or $128$), the magnitude of dot products $|z| \approx \sqrt{d_k}$ becomes very large.
Large inputs into the softmax function push it into regions of vanishingly small gradients (saturation), freezing training.

Scaling by $\frac{1}{\sqrt{d_k}}$ restores the variance of the logits to $\mathbf{1}$:

$$\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}[q \cdot k] = \frac{d_k}{d_k} = 1$$

---

## 3. Multi-Head Attention

Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, **Multi-Head Attention** linearly projects $Q, K, V$ into $h$ distinct lower-dimensional subspaces of dimension $d_k = d_v = d_{\text{model}} / h$:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}\left( Q W_i^Q, K W_i^K, V W_i^V \right)$$

Where parameter projections are:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

**Intuition:** Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. One head may focus on syntactic agreement (subject-verb), another on semantic co-reference (pronoun-noun), and another on adjacent neighboring words.

---

## 4. Sinusoidal Positional Encoding

Because self-attention operations are completely permutation invariant (shuffling word positions produces identical outputs), the architecture requires an explicit signal encoding the sequential position of each token.

Vaswani et al. introduced sinusoidal positional encodings using geometric frequencies:

$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$

$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{\text{model}}}} \right)$$

Where $pos \in [0, \text{max\_len}-1]$ is the token position and $i \in [0, d_{\text{model}}/2 - 1]$ is the dimension index.

### Key Properties:
1. **Boundedness:** Values are bounded in $[-1, 1]$.
2. **Linear Offset Translation:** For any fixed offset $k$, $PE_{(pos + k)}$ can be represented as a linear function of $PE_{(pos)}$ using trigonometric identities:
   $$\sin(\omega (pos + k)) = \sin(\omega \cdot pos)\cos(\omega \cdot k) + \cos(\omega \cdot pos)\sin(\omega \cdot k)$$
   This allows the model to easily learn relative positional relationships.
3. **Extrapolation:** Can generalize to sequences longer than those encountered during training.

---

## 5. Full Architecture: Encoder vs. Decoder

```
      OUTPUT PROBABILITIES
               ↑
            Linear
               ↑
            Softmax
               ↑
       ┌───────────────┐
       │ DECODER BLOCK │ x N
       │  ┌──────────┐ │
       │  │ FFN Layer│ │
       │  │ (Add&Norm│ │
       │  └────▲─────┘ │
       │       │       │
       │  ┌────┴─────┐ │
       │  │Cross-Attn│─┼───────┐
       │  │ (Add&Norm│ │       │
       │  └────▲─────┘ │       │
       │       │       │       │
       │  ┌────┴─────┐ │       │
       │  │Masked-Att│ │       │
       │  │ (Add&Norm│ │       │
       │  └────▲─────┘ │       │
       └───────┼───────┘       │
               │               │
       Target Embeddings       │
               +               │
       Positional Encoding     │
                               │
               ┌───────────────┘
               │
       ┌───────┴───────┐
       │ ENCODER BLOCK │ x N
       │  ┌──────────┐ │
       │  │ FFN Layer│ │
       │  │ (Add&Norm│ │
       │  └────▲─────┘ │
       │       │       │
       │  ┌────┴─────┐ │
       │  │Self-Attn │ │
       │  │ (Add&Norm│ │
       │  └────▲─────┘ │
       └───────┼───────┘
               │
       Source Embeddings
               +
       Positional Encoding
```

### 5.1 Layer Normalization (Add & Norm)
Residual connections bypass each sublayer followed by Layer Normalization:
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$
Unlike Batch Normalization, Layer Normalization computes mean $\mu$ and variance $\sigma^2$ across the hidden dimension of each token independently, making it invariant to batch size and ideal for sequence modeling.

### 5.2 Masking
- **Padding Mask ($M_{\text{pad}}$):** Masks out `<pad>` tokens in input sequences so they contribute $0$ attention weight.
- **Look-Ahead / Causal Mask ($M_{\text{causal}}$):** In the Decoder, a lower-triangular matrix prevents position $i$ from attending to subsequent tokens $j > i$ during auto-regressive decoding.

---

## 6. Pre-training Paradigms: BERT vs. GPT vs. T5

Modern LLM foundation models are built by configuring the Transformer building blocks:

| Model Paradigm | Architectural Core | Objective / Task | Representative Models |
| :--- | :--- | :--- | :--- |
| **Auto-Regressive** (Decoder-Only) | Masked Self-Attention (Causal) | Next-Token Prediction ($P(w_t \mid w_{<t})$) | **GPT-4, LLaMA-3, Claude** |
| **Auto-Encoding** (Encoder-Only) | Bidirectional Self-Attention | Masked Language Modeling (Fill-in-the-blank) | **BERT, RoBERTa, DeBERTa** |
| **Sequence-to-Sequence** (Encoder-Decoder) | Full Encoder + Causal Decoder | Span Corruption / Denoising / Translation | **T5, BART, Flan-T5** |

---

## 7. Week 4 Implementation Lab

### ⚡ Pure NumPy Transformer Architecture
- **File:** [`Transformer-Pre-training/transformer.py`](./Transformer-Pre-training/transformer.py)
- **Status:** **100% Tested & Verified**
- **Components Implemented from Scratch:**
  1. `get_angles` & `positional_encoding`
  2. `create_padding_mask` & `create_look_ahead_mask`
  3. `scaled_dot_product_attention` (with causal masking)
  4. `MultiHeadAttention` (split heads, parallel scaled dot-product, output projection)
  5. `LayerNormalization` & `PositionwiseFeedForward`
  6. `EncoderLayer` (Self-Attention + FFN + Residual Add & Norm)
  7. `DecoderLayer` (Masked Self-Attention + Cross-Attention + FFN + Residual Add & Norm)
  8. Full unit test suite asserting shape integrity, softmax normalization, and causal masking guarantees.
