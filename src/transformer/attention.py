import numpy as np


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize Multi-Head Attention.

        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Initialize weights
        self.W_q = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.02, (d_model, d_model))
        self.W_o = np.random.normal(0, 0.02, (d_model, d_model))

        # Initialize biases
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, d_k)."""
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def combine_heads(self, x, batch_size):
        """Combine heads back together."""
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, -1, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Calculate scaled dot-product attention.

        Args:
            q: Query shape == (..., seq_len_q, d_k)
            k: Key shape == (..., seq_len_k, d_k)
            v: Value shape == (..., seq_len_v, d_v)
            mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)
        """
        matmul_qk = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))

        # Scale matmul_qk
        dk = np.sqrt(self.d_k)
        scaled_attention_logits = matmul_qk / dk

        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout
        if self.dropout > 0:
            attention_weights = self.dropout_layer(attention_weights)

        output = np.matmul(attention_weights, v)
        return output, attention_weights

    def softmax(self, x, axis=-1):
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def dropout_layer(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape) / (
                1 - self.dropout
            )
            return x * mask
        return x

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            q: Query input
            k: Key input
            v: Value input
            mask: Optional mask for attention
        """
        batch_size = q.shape[0]

        # Linear projections and split into heads
        q = np.matmul(q, self.W_q) + self.b_q
        k = np.matmul(k, self.W_k) + self.b_k
        v = np.matmul(v, self.W_v) + self.b_v

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # Combine heads
        concat_attention = self.combine_heads(scaled_attention, batch_size)

        # Final linear projection
        output = np.matmul(concat_attention, self.W_o) + self.b_o

        return output, attention_weights
