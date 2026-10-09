"""
Transformer Architecture & Pre-training Foundations
Deep Learning Specialization - Course 5: Sequence Models (Week 4)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

This module implements the foundational components of the Transformer model from scratch
in pure NumPy:
1. get_angles & PositionalEncoding: Sinusoidal positional embeddings
2. create_padding_mask & create_look_ahead_mask: Attention masking
3. ScaledDotProductAttention: Scaled dot-product self-attention mechanism
4. MultiHeadAttention: Multi-head attention projection and aggregation
5. LayerNormalization: Stable layer normalization across feature channels
6. PositionwiseFeedForward: Two-layer MLP with ReLU activation
7. EncoderLayer: Multi-head self-attention + FFN with residual connections & LayerNorm
8. DecoderLayer: Masked self-attention + cross-attention + FFN with residual connections & LayerNorm
"""

import numpy as np


# ==============================================================================
# 1. Positional Encoding
# ==============================================================================

def get_angles(pos, k, d):
    """
    Get the angles for the sinusoidal positional encoding.

    Arguments:
        pos -- Column vector containing positions [[0], [1], ..., [N-1]], shape (pos, 1)
        k   -- Row vector containing dimension span [[0, 1, 2, ..., d-1]], shape (1, d)
        d   -- Encoding size (int)

    Returns:
        angles -- (pos, d) numpy array
    """
    i = k // 2
    angles = pos / np.power(10000.0, (2.0 * i) / float(d))
    return angles


def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Arguments:
        positions -- Maximum sequence length (int)
        d         -- Embedding dimension / model dimension (int)

    Returns:
        pos_encoding -- (1, positions, d) numpy array
    """
    pos_vec = np.arange(positions)[:, np.newaxis]
    dim_vec = np.arange(d)[np.newaxis, :]

    angle_rads = get_angles(pos_vec, dim_vec, d)

    # Apply sin to even indices; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding.astype(np.float32)


# ==============================================================================
# 2. Masking Utilities
# ==============================================================================

def create_padding_mask(seq, pad_token=0):
    """
    Creates a binary mask for padding tokens.

    Arguments:
        seq -- Input token sequences, shape (batch_size, seq_len)
        pad_token -- Value used for padding (default: 0)

    Returns:
        mask -- (batch_size, 1, 1, seq_len) array where 1 indicates valid token, 0 indicates pad
    """
    mask = (seq != pad_token).astype(np.float32)
    return mask[:, np.newaxis, np.newaxis, :]


def create_look_ahead_mask(seq_len):
    """
    Returns a lower-triangular matrix filled with ones to mask future tokens (causal mask).

    Arguments:
        seq_len -- Sequence length (int)

    Returns:
        mask -- (1, 1, seq_len, seq_len) lower-triangular array
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return mask[np.newaxis, np.newaxis, :, :]


# ==============================================================================
# 3. Scaled Dot-Product Attention
# ==============================================================================

def softmax(x, axis=-1):
    """
    Numerically stable softmax along specified axis.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + M) * V

    Arguments:
        q -- Queries, shape (..., seq_len_q, depth)
        k -- Keys,    shape (..., seq_len_k, depth)
        v -- Values,  shape (..., seq_len_v, depth_v) (seq_len_k == seq_len_v)
        mask -- Optional tensor broadcastable to (..., seq_len_q, seq_len_k)
                where 1 indicates keep, 0 indicates mask out.

    Returns:
        output -- Weighted values, shape (..., seq_len_q, depth_v)
        attention_weights -- Attention weights, shape (..., seq_len_q, seq_len_k)
    """
    # Matrix multiply Q and K^T
    matmul_qk = np.matmul(q, np.swapaxes(k, -1, -2))

    # Scale by sqrt(d_k)
    dk = float(q.shape[-1])
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    # Apply mask by setting masked entries to -1e9
    if mask is not None:
        scaled_attention_logits += (1.0 - mask) * -1e9

    # Softmax over key sequence dimension
    attention_weights = softmax(scaled_attention_logits, axis=-1)

    # Multiply by values
    output = np.matmul(attention_weights, v)

    return output, attention_weights


# ==============================================================================
# 4. Multi-Head Attention
# ==============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention module from "Attention Is All You Need".
    Projects Q, K, V into num_heads subspaces, computes attention in parallel,
    and linearly projects concatenated outputs.
    """
    def __init__(self, d_model, num_heads, seed=42):
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        rng = np.random.RandomState(seed)
        # Xavier / Glorot initialization
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wq = rng.randn(d_model, d_model) * scale
        self.Wk = rng.randn(d_model, d_model) * scale
        self.Wv = rng.randn(d_model, d_model) * scale
        self.Wo = rng.randn(d_model, d_model) * scale

        self.bq = np.zeros((1, 1, d_model))
        self.bk = np.zeros((1, 1, d_model))
        self.bv = np.zeros((1, 1, d_model))
        self.bo = np.zeros((1, 1, d_model))

    def split_heads(self, x, batch_size):
        """
        Split last dimension into (num_heads, depth) and transpose to:
        (batch_size, num_heads, seq_len, depth)
        """
        seq_len = x.shape[1]
        x = np.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, v, k, q, mask=None):
        """
        Forward pass for Multi-Head Attention.

        Arguments:
            v -- Values, shape (batch_size, seq_len_v, d_model)
            k -- Keys,   shape (batch_size, seq_len_k, d_model)
            q -- Query,  shape (batch_size, seq_len_q, d_model)
            mask -- Optional mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            output -- Projected output, shape (batch_size, seq_len_q, d_model)
            attention_weights -- shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.shape[0]

        # 1. Linear projections
        q_proj = np.matmul(q, self.Wq) + self.bq
        k_proj = np.matmul(k, self.Wk) + self.bk
        v_proj = np.matmul(v, self.Wv) + self.bv

        # 2. Split heads -> (batch_size, num_heads, seq_len, depth)
        q_split = self.split_heads(q_proj, batch_size)
        k_split = self.split_heads(k_proj, batch_size)
        v_split = self.split_heads(v_proj, batch_size)

        # 3. Scaled dot-product attention
        scaled_attn, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask
        )

        # 4. Transpose and concatenate heads back -> (batch_size, seq_len_q, d_model)
        scaled_attn = np.transpose(scaled_attn, (0, 2, 1, 3))
        concat_attn = np.reshape(scaled_attn, (batch_size, -1, self.d_model))

        # 5. Final linear output projection
        output = np.matmul(concat_attn, self.Wo) + self.bo

        return output, attention_weights


# ==============================================================================
# 5. Layer Normalization & Position-wise Feed Forward
# ==============================================================================

class LayerNormalization:
    """
    Layer Normalization over the last feature axis.
    Formula: LN(x) = gamma * ((x - mean) / sqrt(var + eps)) + beta
    """
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones((1, 1, d_model), dtype=np.float32)
        self.beta = np.zeros((1, 1, d_model), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta


class PositionwiseFeedForward:
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x * W1 + b1) * W2 + b2
    """
    def __init__(self, d_model, d_ff, seed=42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = rng.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros((1, 1, d_model))

    def forward(self, x):
        # First layer with ReLU
        hidden = np.maximum(0.0, np.matmul(x, self.W1) + self.b1)
        # Second linear layer
        output = np.matmul(hidden, self.W2) + self.b2
        return output


# ==============================================================================
# 6. Encoder Layer & Decoder Layer
# ==============================================================================

class EncoderLayer:
    """
    Single Transformer Encoder Layer consisting of:
    1. Multi-Head Self-Attention + Residual Connection + LayerNorm
    2. Position-wise Feed-Forward + Residual Connection + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, seed=42):
        self.mha = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, seed=seed + 1)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for Encoder Layer.

        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, d_model)
            mask -- Optional padding mask

        Returns:
            out -- Tensor of shape (batch_size, input_seq_len, d_model)
            weights -- Attention weights
        """
        # Self-attention sublayer with residual connection
        attn_out, weights = self.mha.forward(v=x, k=x, q=x, mask=mask)
        out1 = self.layernorm1.forward(x + attn_out)

        # Feed-forward sublayer with residual connection
        ffn_out = self.ffn.forward(out1)
        out2 = self.layernorm2.forward(out1 + ffn_out)

        return out2, weights


class DecoderLayer:
    """
    Single Transformer Decoder Layer consisting of:
    1. Masked Multi-Head Self-Attention + Residual + LayerNorm
    2. Multi-Head Cross-Attention (queries from decoder, keys/values from encoder) + Residual + LayerNorm
    3. Position-wise Feed-Forward + Residual + LayerNorm
    """
    def __init__(self, d_model, num_heads, d_ff, seed=42):
        self.mha1 = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.mha2 = MultiHeadAttention(d_model, num_heads, seed=seed + 2)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, seed=seed + 3)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.layernorm3 = LayerNormalization(d_model)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Forward pass for Decoder Layer.

        Arguments:
            x -- Target sequence representations, shape (batch_size, target_seq_len, d_model)
            enc_output -- Representations from Encoder, shape (batch_size, input_seq_len, d_model)
            look_ahead_mask -- Causal mask preventing attending to future tokens
            padding_mask -- Mask for encoder output key padding

        Returns:
            out -- Tensor of shape (batch_size, target_seq_len, d_model)
            self_attn_weights -- Decoder self-attention weights
            cross_attn_weights -- Decoder cross-attention weights
        """
        # 1. Masked Self-Attention
        attn1, self_attn_weights = self.mha1.forward(v=x, k=x, q=x, mask=look_ahead_mask)
        out1 = self.layernorm1.forward(x + attn1)

        # 2. Cross-Attention with Encoder Output (Q from decoder, K & V from encoder)
        attn2, cross_attn_weights = self.mha2.forward(v=enc_output, k=enc_output, q=out1, mask=padding_mask)
        out2 = self.layernorm2.forward(out1 + attn2)

        # 3. Position-wise Feed-Forward
        ffn_out = self.ffn.forward(out2)
        out3 = self.layernorm3.forward(out2 + ffn_out)

        return out3, self_attn_weights, cross_attn_weights


# ==============================================================================
# Standalone Unit Test Suite
# ==============================================================================

def test_positional_encoding():
    print("Testing Positional Encoding...", end=" ")
    pe = positional_encoding(positions=50, d=16)
    assert pe.shape == (1, 50, 16), f"Wrong PE shape: {pe.shape}"
    # Verify values are bounded between -1 and 1
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0), "PE values out of bounds [-1, 1]"
    # Position 0 sin should be 0, cos should be 1
    assert np.isclose(pe[0, 0, 0], 0.0, atol=1e-5), "PE sin(0) != 0"
    assert np.isclose(pe[0, 0, 1], 1.0, atol=1e-5), "PE cos(0) != 1"
    print("[PASSED]")


def test_scaled_dot_product_attention():
    print("Testing Scaled Dot-Product Attention...", end=" ")
    np.random.seed(42)
    q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
    k = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
    v = np.array([[[[10.0, 20.0], [30.0, 40.0]]]])  # (1, 1, 2, 2)

    # Without mask
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (1, 1, 2, 2), f"Wrong output shape: {out.shape}"
    assert weights.shape == (1, 1, 2, 2), f"Wrong weights shape: {weights.shape}"
    # Softmax rows must sum to 1.0
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Attention weights do not sum to 1"

    # With causal look-ahead mask
    mask = create_look_ahead_mask(2)
    out_masked, weights_masked = scaled_dot_product_attention(q, k, v, mask=mask)
    # Future token (row 0, col 1) should receive zero attention
    assert np.isclose(weights_masked[0, 0, 0, 1], 0.0, atol=1e-5), "Masked attention is not 0"
    assert np.isclose(weights_masked[0, 0, 0, 0], 1.0, atol=1e-5), "First token should receive 100% attention"
    print("[PASSED]")


def test_multi_head_attention():
    print("Testing Multi-Head Attention...", end=" ")
    batch_size = 2
    seq_len = 4
    d_model = 16
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
    x = np.random.randn(batch_size, seq_len, d_model)
    out, weights = mha.forward(v=x, k=x, q=x)

    assert out.shape == (batch_size, seq_len, d_model), f"Wrong MHA output shape: {out.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Wrong weights shape: {weights.shape}"
    print("[PASSED]")


def test_encoder_and_decoder_layers():
    print("Testing Encoder & Decoder Layers...", end=" ")
    batch_size = 2
    src_len = 5
    tgt_len = 6
    d_model = 16
    num_heads = 4
    d_ff = 32

    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, seed=1)
    decoder_layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, seed=2)

    # Input source and target representations
    src_input = np.random.randn(batch_size, src_len, d_model)
    tgt_input = np.random.randn(batch_size, tgt_len, d_model)

    # 1. Forward through Encoder
    enc_out, enc_weights = encoder_layer.forward(src_input)
    assert enc_out.shape == (batch_size, src_len, d_model), f"Wrong enc_out shape: {enc_out.shape}"

    # 2. Forward through Decoder with look-ahead mask
    look_ahead_mask = create_look_ahead_mask(tgt_len)
    dec_out, self_w, cross_w = decoder_layer.forward(
        tgt_input, enc_output=enc_out, look_ahead_mask=look_ahead_mask
    )
    assert dec_out.shape == (batch_size, tgt_len, d_model), f"Wrong dec_out shape: {dec_out.shape}"
    assert self_w.shape == (batch_size, num_heads, tgt_len, tgt_len), f"Wrong self_w shape: {self_w.shape}"
    assert cross_w.shape == (batch_size, num_heads, tgt_len, src_len), f"Wrong cross_w shape: {cross_w.shape}"

    # Check causal masking in decoder self-attention
    assert np.allclose(self_w[:, :, 0, 1:], 0.0, atol=1e-5), "Decoder self-attention leaked future tokens"
    print("[PASSED]")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING ALL TRANSFORMER UNIT TESTS")
    print("=" * 70)
    test_positional_encoding()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_encoder_and_decoder_layers()
    print("=" * 70)
    print("ALL TRANSFORMER ARCHITECTURE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)
