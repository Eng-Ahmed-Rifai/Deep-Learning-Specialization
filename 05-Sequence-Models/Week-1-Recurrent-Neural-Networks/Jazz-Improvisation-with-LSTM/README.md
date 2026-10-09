# Lab 3: Jazz Improvisation with LSTM

> **Course 5: Sequence Models (Week 1, Lab 3)** — *DeepLearning.AI*

## Overview
Generates jazz solo music using a trained recurrent neural network (LSTM).

## Architecture
- Musical notes encoded into one-hot vectors across 78 pitch and chord classes ($n_x = 78$).
- LSTM sequence model with 64 hidden units sharing weights across all generation time steps.
- Auto-regressive generation: the output note sampled at step $t$ is fed as the input token $x^{\langle t+1 \rangle}$ to step $t+1$.
