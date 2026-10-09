# Week 3: Sequence Models & The Attention Mechanism

> **Course 5: Sequence Models** — *DeepLearning.AI / Andrew Ng*  
> **Learner & Implementer:** Eng. Ahmed Rifai (100.00% Verified)

---

## 1. Sequence-to-Sequence (Seq2Seq) & The Information Bottleneck

In tasks like Neural Machine Translation (NMT), Speech-to-Text, and Text Summarization, the input sequence length $T_x$ does not equal the output sequence length $T_y$.

```
ENCODER                                     DECODER
x⟨1⟩ → [RNN] ──┐
x⟨2⟩ → [RNN] ──┼─> [ c ] (Context Vector) ─> [RNN] → y⟨1⟩
x⟨3⟩ → [RNN] ──┘                             [RNN] → y⟨2⟩
                                             [RNN] → y⟨3⟩
```

### The Information Bottleneck Problem
In classic encoder-decoder architectures (Sutskever et al., 2014), the entire input sequence of arbitrary length is compressed into a single fixed-dimensional context vector $c = h_{T_x}$.
- For sentences longer than 20-30 words, translation quality drops catastrophically.
- A single vector cannot preserve granular token alignments, clauses, or syntax without loss of critical information.

---

## 2. Decoding & Search Strategies: Beam Search

### 2.1 Greedy vs. Exhaustive Search
- **Greedy Search:** Chooses the single most probable token $\hat{y}^{\langle t \rangle} = \arg\max_y P(y \mid y^{\langle <t \rangle}, x)$ at every step. Fails because early errors cannot be undone (e.g., choosing *"Jane"* instead of *"Jane is"* can derail the entire sentence).
- **Exhaustive Search:** Explores all possible output sentences $\to O(|V|^{T_y})$ possibilities, which is computationally intractable ($10,000^{20} = 10^{80}$).

### 2.2 Beam Search Formulation
Beam Search maintains a beam width $B$ (typically $B \in [3, 10]$) of the most promising partial translations:

At step $t$:
1. For each of the $B$ active candidates, compute probabilities across all $|V|$ possible next tokens.
2. Select the top $B$ candidates according to cumulative log probability:
   $$\arg\max_{\{y^{\langle 1 \rangle}, \dots, y^{\langle t \rangle}\}} \sum_{\tau=1}^t \log P(y^{\langle \tau \rangle} \mid y^{\langle <\tau \rangle}, x)$$

### 2.3 Length Normalization
Because summing log probabilities ($\log p < 0$) penalizes longer translations unfairly, Beam Search incorporates length normalization with exponent $\alpha \approx 0.7$:

$$\text{Score}(y, x) = \frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \mid y^{\langle <t \rangle}, x)$$

---

## 3. The Attention Mechanism: Intuition & Architecture

Rather than compressing the source sentence into a single vector, the **Attention Mechanism** allows the decoder to dynamically look back at all encoder hidden states at each decoding step $t$.

```
Encoder States:   h⟨1⟩       h⟨2⟩       h⟨3⟩       h⟨Tx⟩
                    │          │          │          │
Attention Weights: α⟨t,1⟩     α⟨t,2⟩     α⟨t,3⟩     α⟨t,Tx⟩
                    └──────────┴────┬─────┴──────────┘
                                    ↓
                       Context Vector c⟨t⟩ = Σ α⟨t,i⟩ * h⟨i⟩
                                    ↓
                       Decoder Hidden State s⟨t⟩
                                    ↓
                            Prediction y⟨t⟩
```

---

## 4. Deep Comparative Analysis: Bahdanau Attention vs. Luong Attention

Two landmark attention mechanisms shaped modern NLP: **Bahdanau Attention** (Additive Attention, 2014) and **Luong Attention** (Multiplicative Attention, 2015).

### 4.1 Bahdanau Attention (Additive Attention)
- **Proposed by:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (ICLR 2015 / 2014).
- **Decoder state timing:** Uses the *previous* decoder state $s_{t-1}$ to decide which encoder annotations $h_i$ to attend to.

**Mathematical Formulation:**
1. **Alignment Model (Score Function):**
   $$e_{t, i} = v_a^T \tanh\left( W_a s_{t-1} + U_a h_i \right)$$
   *(Parameterized as a feedforward network with weights $W_a, U_a, v_a$)*.

2. **Attention Weights (Softmax):**
   $$\alpha_{t, i} = \frac{\exp(e_{t, i})}{\sum_{k=1}^{T_x} \exp(e_{t, k})}$$

3. **Context Vector:**
   $$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i$$

4. **Decoder State Update:**
   $$s_t = f(s_{t-1}, [y_{t-1}; c_t])$$

5. **Output Prediction:**
   $$\hat{y}_t = g(s_t, y_{t-1}, c_t)$$

---

### 4.2 Luong Attention (Multiplicative Attention)
- **Proposed by:** Minh-Thang Luong, Hieu Pham, Christopher D. Manning (EMNLP 2015).
- **Decoder state timing:** The decoder *first* updates its current hidden state $s_t$ from $s_{t-1}$ and $y_{t-1}$, and then uses the *current* state $s_t$ to compute attention over encoder states.

**Mathematical Formulation:**
1. **Decoder State First:**
   $$s_t = \text{RNN}(s_{t-1}, y_{t-1})$$

2. **Alignment Score Options:**
   $$\text{score}(s_t, h_i) = \begin{cases} s_t^T h_i & \text{Dot (simplest, no extra parameters)} \\ s_t^T W_a h_i & \text{General (multiplicative matrix)} \\ v_a^T \tanh(W_a [s_t; h_i]) & \text{Concat (additive variant)} \end{cases}$$

3. **Attention Weights:**
   $$\alpha_{t, i} = \frac{\exp(\text{score}(s_t, h_i))}{\sum_{k=1}^{T_x} \exp(\text{score}(s_t, h_k))}$$

4. **Context Vector:**
   $$c_t = \sum_{i=1}^{T_x} \alpha_{t, i} h_i$$

5. **Attentional Hidden State ($\tilde{s}_t$):**
   Combines current state and context vector before final projection:
   $$\tilde{s}_t = \tanh\left( W_c [c_t; s_t] \right)$$

6. **Output Prediction:**
   $$\hat{y}_t = \text{softmax}(W_s \tilde{s}_t)$$

---

### 4.3 Direct Comparison Table

| Attribute | Bahdanau Attention (2014) | Luong Attention (2015) |
| :--- | :--- | :--- |
| **Primary Nature** | Additive ($W_a s_{t-1} + U_a h_i$) | Multiplicative ($s_t^T W_a h_i$) or Dot product |
| **Decoder State Used** | Previous hidden state $s_{t-1}$ | Current hidden state $s_t$ |
| **Information Flow** | $s_{t-1} \to c_t \to s_t \to \hat{y}_t$ | $s_{t-1} \to s_t \to c_t \to \tilde{s}_t \to \hat{y}_t$ |
| **Computational Cost** | Slower (2 matrix multiplies per step) | Faster (optimized BLAS dot product matrix multiplies) |
| **Scope** | Global attention across all $T_x$ | Introduces both **Global** and **Local** attention (Gaussian window) |
| **Legacy & Evolution** | Foundation of modern Seq2Seq | Direct precursor to Transformer Dot-Product Attention ($Q K^T$) |

---

## 5. Week 3 Implementation Labs

### 🌍 Lab 1: Neural Machine Translation with Attention
- Human date string translation (e.g., `"9th of May 1998"`, `"May 9, 1998"`, `"1998-05-09"`) into standard ISO format (`YYYY-MM-DD`).
- Implemented with Keras custom attention layers, computing $\alpha^{\langle t, t' \rangle}$ heatmaps visualizing model focus during translation.

### 🎙️ Lab 2: Trigger Word Detection
- Training an audio wake-word detection model (recognizing `"activate"`).
- Synthetic audio data synthesis (mixing speech with background noises).
- Spectrogram feature extraction via Short-Time Fourier Transform (STFT).
- 1D Convolutional temporal filtering followed by GRU sequence modeling, emitting 1s during a 50-step window immediately following the trigger keyword.
