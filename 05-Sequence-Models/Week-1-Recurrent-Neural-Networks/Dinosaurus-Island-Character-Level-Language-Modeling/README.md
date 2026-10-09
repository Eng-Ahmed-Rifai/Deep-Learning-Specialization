# Lab 2: Dinosaurus Island — Character-Level Language Modeling

> **Course 5: Sequence Models (Week 1, Lab 2)** — *DeepLearning.AI*

## Overview
Character-level language model generating novel dinosaur names from a training list of 1,536 real dinosaur taxonomy records.

## Mathematical Concepts
1. **Character Vocabulary Mapping:**
   $$V = \{ \text{'\n'}, \text{'a'}, \text{'b'}, \dots, \text{'z'} \}, \quad |V| = 27$$
2. **Gradient Clipping:**
   Prevents gradient explosion across unrolled recurrent computation graphs:
   $$\mathbf{g} \leftarrow \text{clip}(\mathbf{g}, -N, N)$$
3. **Generative Sampling with Temperature:**
   At each character generation step, choose the next character using multinomial sampling from the Softmax distribution rather than greedy $\arg\max$, enabling creative character diversity:
   $$\hat{y}^{\langle t \rangle} \sim \text{Multinomial}(P(\text{char} \mid a^{\langle t \rangle}))$$
