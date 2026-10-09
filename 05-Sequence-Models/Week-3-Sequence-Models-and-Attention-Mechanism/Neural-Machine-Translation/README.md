# Lab 1: Neural Machine Translation with Attention

> **Course 5: Sequence Models (Week 3, Lab 1)** — *DeepLearning.AI*

## Overview
Translates diverse human-readable dates (e.g., `"9th of May 1998"`, `"May 9, 1998"`, `"1998-05-09"`, `"09/05/98"`) into standard ISO format (`YYYY-MM-DD`).

## Key Components
- **Bidirectional LSTM Pre-Attention Encoder:** Encodes input character sequence of length $T_x = 30$ into forward and backward representations $h^{\langle t \rangle} = [\vec{a}^{\langle t \rangle}, \overleftarrow{a}^{\langle t \rangle}]$.
- **Additive Attention Mechanism:** Calculates attention weights $\alpha^{\langle t, t' \rangle}$ over encoder states for each target decoding step $t \in [1, T_y]$ ($T_y = 10$).
- **Post-Attention LSTM Decoder:** Receives context vector $c^{\langle t \rangle} = \sum_{t'} \alpha^{\langle t, t' \rangle} h^{\langle t' \rangle}$ to predict the standardized date token.
- **Attention Visualization:** Generates alignment heatmaps displaying which input characters the network attended to when generating year, month, and day digits.
