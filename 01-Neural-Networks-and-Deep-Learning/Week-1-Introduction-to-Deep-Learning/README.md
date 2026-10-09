# Week 1: Introduction to Deep Learning

> **Course 1: Neural Networks and Deep Learning** — *DeepLearning.AI / Andrew Ng*

---

## 1. What is a Neural Network?

A neural network is a mathematical function that maps input features $x$ to predictions $y$ by passing them through layers of non-linear transformations.

### Housing Price Prediction Example:
- **Input ($x$):** Size of the house ($x$).
- **Linear Neuron:** Price cannot be negative, so we use a Rectified Linear Unit (**ReLU**):  
  $$f(x) = \max(0, w \cdot x + b)$$
- **Deep Architecture:** By combining features (size, number of bedrooms, zip code, wealth) into intermediate hidden units (family size, walkability, school quality), the network automatically discovers latent representations predicting the final price.

---

## 2. Supervised Learning with Neural Networks

Neural network architectures specialize depending on the data type:

| Input Domain | Data Modality | Architecture Family |
| :--- | :--- | :--- |
| Real Estate, Advertising | Structured Data (Databases, Tables) | Standard Fully Connected Deep NNs |
| Image Recognition, Vision | Unstructured Data (Pixel Grids) | Convolutional Neural Networks (CNNs) |
| Speech, Audio, Translation | Sequential / Temporal Data | Recurrent Neural Networks (RNNs, LSTMs) |
| Autonomous Driving | Multimodal (Video, Radar, Lidar) | Custom Hybrid CNN / Attention Networks |

---

## 3. Why Deep Learning is Taking Off: Scale Drivers

For decades, basic machine learning algorithms (Linear Regression, Support Vector Machines, Decision Trees) dominated industry. Deep learning took off due to the convergence of three factors:

```
        Performance ^
                    |               /  Large Deep Neural Network
                    |              /
                    |             /    Medium Neural Network
                    |            /
                    |    -------/      Small Neural Network
                    |   /
                    |  /               Traditional ML (SVM, Logistic Reg)
                    | /--------------- (Plateaus with more data)
                    +--------------------------------------------->
                                               Amount of Data
```

### The Three Drivers:
1. **Data:** Digitization, smartphones, IoT, and the internet created massive labeled datasets.
2. **Compute:** GPUs (Nvidia CUDA), TPUs, and modern multi-core CPUs enabled training networks with billions of parameters.
3. **Algorithmic Innovation:**
   - Shifting from **Sigmoid** to **ReLU** eliminated the vanishing gradient problem in deep networks, speeding up gradient descent by orders of magnitude.
   - Batch Normalization, Adam Optimizer, ResNet residual connections.
