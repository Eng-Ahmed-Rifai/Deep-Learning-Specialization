# Lab 1: Operations on Word Vectors

> **Course 5: Sequence Models (Week 2, Lab 1)** — *DeepLearning.AI*

## Overview
Explores pre-trained 50-dimensional GloVe word embeddings to perform:
1. **Cosine Similarity:** Compute directional semantic alignment between words.
2. **Word Analogy Completion:** Solve linguistic word analogies using linear vector algebra:
   $$\arg\max_w \text{cosine\_similarity}(e_w, e_b - e_a + e_c)$$
3. **Gender Bias Mitigation:** Identify gender bias direction $g$ via difference vectors and implement:
   - **Neutralize:** Project out gender bias component from non-gender words.
   - **Equalize:** Ensure gender-specific pairs are equidistant from neutral words.
