# Lab 2: Emojify

> **Course 5: Sequence Models (Week 2, Lab 2)** — *DeepLearning.AI*

## Overview
Classifies text sentences into 5 distinct emoji sentiment categories:
❤️ (Heart/Love), ⚾ (Baseball/Sports), 😄 (Smile/Joy), 😞 (Disappointed), 🍴 (Food).

## Architectures Implemented
1. **Emojifier-V1 (Baseline):**
   - Maps each word in a sentence to its GloVe 50d vector and calculates the arithmetic mean vector:
     $$\mathbf{e}_{\text{avg}} = \frac{1}{T} \sum_{t=1}^T e_{w_t}$$
   - Feeds $\mathbf{e}_{\text{avg}}$ into a Softmax classifier.
   - *Limitation:* Does not account for word ordering (e.g., misclassifies *"not feeling happy today"* as positive).
2. **Emojifier-V2 (Deep LSTM):**
   - Incorporates sequence order using a 2-layer stacked LSTM network with pre-trained non-trainable GloVe embedding layer.
   - Dropout layers ($0.5$) applied between recurrent layers to prevent overfitting on tiny datasets.
   - Accurately captures nuanced syntax and negation.
