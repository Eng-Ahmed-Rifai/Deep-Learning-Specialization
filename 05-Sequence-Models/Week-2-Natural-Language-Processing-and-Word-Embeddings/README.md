# Week 2: Natural Language Processing & Word Embeddings

> **Course 5: Sequence Models** — *DeepLearning.AI / Andrew Ng*  
> **Learner & Implementer:** Eng. Ahmed Rifai (100.00% Verified)

---

## 1. Word Representation: One-Hot vs. Distributed Dense Embeddings

### 1.1 Limitation of One-Hot Vectors
Traditionally, words in a vocabulary $V$ (e.g., $|V| = 50,000$) were represented as sparse one-hot vectors:

$$\mathbf{o}_{\text{apple}} = [0, 0, \dots, 1, \dots, 0]^T, \quad \mathbf{o}_{\text{orange}} = [0, \dots, 1, \dots, 0]^T$$

Because all one-hot vectors are mutually orthogonal:

$$\mathbf{o}_{\text{apple}}^T \mathbf{o}_{\text{orange}} = 0$$

An algorithm trained on `"I want a glass of orange juice"` has zero transfer ability to understand `"I want a glass of apple ______"`, as it views `apple` and `orange` as completely unrelated points in a 50,000-dimensional space.

### 1.2 Dense Distributed Word Embeddings
In word embeddings, each word is mapped to a low-dimensional dense vector $e_w \in \mathbb{R}^d$ ($d \in [50, 300]$). Each dimension represents an implicit semantic property (e.g., gender, animacy, royal status, food category, age):

$$e_{\text{king}} \approx [0.91, 0.95, 0.05, \dots]^T, \quad e_{\text{queen}} \approx [0.90, -0.93, 0.04, \dots]^T$$

Given an **Embedding Matrix** $E \in \mathbb{R}^{d \times |V|}$, retrieving the embedding of word $w$ with index $j$ is a column lookup:

$$e_w = E \cdot \mathbf{o}_w$$

---

## 2. Cosine Similarity & Vector Analogies

### 2.1 Cosine Similarity
To measure semantic similarity between two word vectors $u$ and $v$:

$$\text{Sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- $\text{Sim}(u, v) = 1$: identical direction.
- $\text{Sim}(u, v) = 0$: orthogonal concepts.
- $\text{Sim}(u, v) = -1$: exact opposite direction.

### 2.2 Vector Analogies
Remarkably, linear arithmetic on learned word embeddings captures rich semantic and syntactic relationships:

$$e_{\text{man}} - e_{\text{woman}} \approx e_{\text{king}} - e_{\text{queen}}$$

To solve the analogy problem *"man is to woman as king is to ?"*:

$$\arg\max_w \text{Sim}\left(e_w, e_{\text{king}} - e_{\text{man}} + e_{\text{woman}}\right) \quad \text{for } w \notin \{\text{king}, \text{man}, \text{woman}\}$$

---

## 3. Word2Vec Learning Algorithms

### 3.1 Skip-Gram Model
Given a target word $t$, predict words appearing within a surrounding context window $c \in [-C, +C]$.

Using naive softmax:

$$P(c \mid t) = \frac{\exp(\theta_c^T e_t)}{\sum_{w=1}^{|V|} \exp(\theta_w^T e_t)}$$

**Computational Bottleneck:** The denominator sums over all $|V| = 50,000$ to $1,000,000$ words, making gradient descent prohibitively slow $O(|V|)$ per word.

### 3.2 Negative Sampling (SGNS - Mikolov et al.)
Negative Sampling reformulates the multiclass problem into $|K| + 1$ independent binary logistic classifications:

1. For a true pair $(t, c)$, assign label $y = 1$.
2. Randomly sample $K$ context words $w_1, \dots, w_K$ from the vocabulary according to unigram frequency, and assign label $y = 0$.
3. Model probability of a pair $(t, w)$ being valid:
   $$P(y = 1 \mid (t, w)) = \sigma(\theta_w^T e_t)$$

The objective to maximize is:

$$\mathcal{L} = \log \sigma(\theta_c^T e_t) + \sum_{k=1}^K \log \sigma(-\theta_{w_k}^T e_t)$$

Complexity per training step drops from $O(|V|)$ to $O(K)$, where $K \approx 5-20$ for small datasets and $2-5$ for massive corpora!

**Sampling Distribution:**
Mikolov et al. empirically established the modified unigram distribution with exponent $3/4$:

$$P(w) = \frac{f(w)^{3/4}}{\sum_{i=1}^{|V|} f(w_i)^{3/4}}$$

This boosts the sampling probability of rare words while dampening extreme frequent stop words like *"the", "is", "of"*.

---

## 4. GloVe: Global Vectors for Word Representation

Pennington et al. (Stanford, 2014) combined the advantages of global matrix factorization (like SVD) and local context window methods (Skip-Gram).

Let $X_{i,j}$ be the co-occurrence count of word $i$ appearing in the context of word $j$. GloVe minimizes the weighted least-squares loss:

$$J = \sum_{i,j=1}^{|V|} f(X_{i,j}) \left( \theta_i^T e_j + b_i + b_j' - \log X_{i,j} \right)^2$$

Where the weighting function $f(X_{i,j})$ prevents rare pairs from dominating and frequent pairs from skewing the space:

$$f(x) = \begin{cases} \left( \frac{x}{x_{\max}} \right)^\alpha & \text{if } x < x_{\max} \quad (\text{with } \alpha = 0.75, x_{\max} = 100) \\ 1 & \text{otherwise} \end{cases}$$

Because $X_{i,j}$ is symmetric, $\theta_i$ and $e_j$ play symmetric roles. The final word representation is taken as:

$$e_w = \frac{\theta_w + e_w}{2}$$

---

## 5. Debiasing Word Embeddings

Word embeddings trained on uncurated internet corpora absorb and reinforce harmful demographic and societal stereotypes (e.g., associating *"doctor"* predominantly with *"man"* and *"homemaker"* with *"woman"*).

Bolukbasi et al. (2016) developed a 3-step geometric debiasing algorithm:

### Step 1: Identify the Bias Subspace
Compute the difference vectors between definitional gender pairs:
$$d_1 = e_{\text{he}} - e_{\text{she}}, \quad d_2 = e_{\text{male}} - e_{\text{female}}, \quad d_3 = e_{\text{father}} - e_{\text{mother}}$$

Apply Singular Value Decomposition (SVD) on $[d_1, d_2, d_3, \dots]$ to find the principal direction vector $g \in \mathbb{R}^d$ corresponding to the bias subspace.

### Step 2: Neutralize Non-Definitional Words
For non-inherently gendered words (e.g., *"doctor", "nurse", "babysitter", "scientist"*), eliminate their projection onto direction $g$:

$$e_{\text{neutral}} = e - \frac{e \cdot g}{\|g\|_2^2} g$$

### Step 3: Equalize Definitional Pairs
For pairs that should retain gender distinctions (e.g., *"grandfather"* and *"grandmother"*), equalize their distance to all neutralized concepts:

$$\mu = \frac{e_1 + e_2}{2}$$
$$e_1' = \mu_{\perp} + \sqrt{1 - \|\mu_{\perp}\|^2} \cdot \frac{(e_1)_g}{\|(e_1)_g\|}$$
$$e_2' = \mu_{\perp} - \sqrt{1 - \|\mu_{\perp}\|^2} \cdot \frac{(e_1)_g}{\|(e_1)_g\|}$$

---

## 6. Week 2 Implementation Labs

### 📐 Lab 1: Operations on Word Vectors
- Implementation of Cosine Similarity between word vectors.
- Word analogy completion algorithm ($\arg\max_w \text{Sim}$).
- Identification of bias vectors and mathematical neutralization and equalization.

### 😀 Lab 2: Emojify
- **Emojifier-V1 (Baseline):** Converts sentences into average GloVe 50d vectors $\frac{1}{T} \sum_{t=1}^T e_{w_t}$, fed into a Softmax classifier.
  - *Limitation:* Fails on sentences where word order changes meaning, such as `"not feeling happy today"` (average still dominated by `happy`).
- **Emojifier-V2 (Deep LSTM):** 2-layer stacked LSTM taking GloVe sequence inputs directly, using Dropout ($0.5$), and maintaining grammatical word order to achieve $> 90\%$ test accuracy.
