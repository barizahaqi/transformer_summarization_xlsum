import numpy as np
from .attention import MultiHeadAttention


class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        """
        Initialize layer normalization.

        Args:
            d_model (int): Model dimension
            eps (float): Small constant for numerical stability
        """
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.cache = {}  # Cache for storing intermediate values

    def forward(self, x):
        """
        Forward pass of layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        # Store input for backward pass
        self.cache["x"] = x

        # Compute mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Store intermediate values for backward pass
        self.cache["mean"] = mean
        self.cache["var"] = var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        self.cache["x_norm"] = x_norm

        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        """
        Backward pass of layer normalization.

        Args:
            dout: Gradient of loss w.r.t. output of shape (batch_size, seq_length, d_model)
                 or (num_layers, batch_size, seq_length, d_model)
        """
        # Get cached values
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_norm = self.cache["x_norm"]

        batch_size, seq_len, d_model = x.shape

        # Handle extra dimension if present
        if len(dout.shape) == 4:
            # If dout has shape (num_layers, batch_size, seq_length, d_model)
            # We need to sum over the first dimension to get the correct gradient
            dout = np.sum(dout, axis=0)  # Shape: (batch_size, seq_length, d_model)

        # Verify dout has the correct shape
        assert (
            len(dout.shape) == 3
        ), f"Expected dout to have 3 dimensions, got {len(dout.shape)}"
        assert (
            dout.shape[2] == d_model
        ), f"Expected d_model dimension to be {d_model}, got {dout.shape[2]}"
        assert (
            dout.shape[:2] == x_norm.shape[:2]
        ), f"Batch and sequence dimensions must match: dout {dout.shape[:2]} != x_norm {x_norm.shape[:2]}"

        # Gradient of loss w.r.t. beta
        # Sum over batch and sequence dimensions to get gradient for each feature
        dbeta = np.sum(dout, axis=(0, 1))  # Shape: (d_model,)
        assert (
            dbeta.shape == self.beta.shape
        ), f"dbeta shape {dbeta.shape} != beta shape {self.beta.shape}"

        # Gradient of loss w.r.t. gamma
        # Element-wise multiplication and sum over batch and sequence dimensions
        # Ensure x_norm has the same shape as dout
        x_norm_reshaped = x_norm.reshape(batch_size, seq_len, d_model)
        dgamma = np.sum(dout * x_norm_reshaped, axis=(0, 1))  # Shape: (d_model,)
        assert (
            dgamma.shape == self.gamma.shape
        ), f"dgamma shape {dgamma.shape} != gamma shape {self.gamma.shape}"

        # Gradient of loss w.r.t. normalized input
        dx_norm = dout * self.gamma  # Shape: (batch_size, seq_len, d_model)

        # Gradient of loss w.r.t. variance
        dvar = np.sum(
            dx_norm * (x - mean) * -0.5 * (var + self.eps) ** (-1.5),
            axis=-1,
            keepdims=True,
        )  # Shape: (batch_size, seq_len, 1)

        # Gradient of loss w.r.t. mean
        dmean = (
            np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True)
            + dvar * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / d_model
        )  # Shape: (batch_size, seq_len, 1)

        # Gradient of loss w.r.t. input
        dx = (
            dx_norm / np.sqrt(var + self.eps)
            + dvar * 2 * (x - mean) / d_model
            + dmean / d_model
        )  # Shape: (batch_size, seq_len, d_model)

        # Store gradients for parameter updates
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class FeedForward:
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model (int): Model dimension
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Initialize weights
        self.W1 = np.random.normal(0, 0.02, (d_model, d_ff))
        self.W2 = np.random.normal(0, 0.02, (d_ff, d_model))
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

        # Cache for backward pass
        self.cache = {}

    def dropout_layer(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape) / (
                1 - self.dropout
            )
            return x * mask
        return x

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Forward pass of feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        # Store input for backward pass
        self.cache["x"] = x

        # First linear layer with ReLU
        h = np.matmul(x, self.W1) + self.b1
        self.cache["h_pre_relu"] = h
        h = self.relu(h)

        # Apply dropout
        h = self.dropout_layer(h)
        self.cache["h_dropout"] = h

        # Second linear layer
        output = np.matmul(h, self.W2) + self.b2
        return output

    def backward(self, dout):
        """
        Backward pass of feed-forward network.

        Args:
            dout: Gradient of loss w.r.t. output of shape (batch_size, seq_length, d_model)
                 or (num_layers, batch_size, seq_length, d_model)
        """
        # Get cached values
        x = self.cache["x"]
        h_pre_relu = self.cache["h_pre_relu"]
        h_dropout = self.cache["h_dropout"]

        # Handle extra dimension if present
        if len(dout.shape) == 4:
            # If dout has shape (num_layers, batch_size, seq_length, d_model)
            # We need to sum over the first dimension to get the correct gradient
            dout = np.sum(dout, axis=0)  # Shape: (batch_size, seq_length, d_model)

        # Verify dout has the correct shape
        assert (
            len(dout.shape) == 3
        ), f"Expected dout to have 3 dimensions, got {len(dout.shape)}"
        assert (
            dout.shape[2] == self.d_model
        ), f"Expected d_model dimension to be {self.d_model}, got {dout.shape[2]}"

        batch_size, seq_len, _ = dout.shape

        # Gradient of loss w.r.t. second linear layer
        # Reshape tensors for matrix multiplication
        h_dropout_reshaped = h_dropout.reshape(
            -1, self.d_ff
        )  # (batch_size * seq_len, d_ff)
        dout_reshaped = dout.reshape(
            -1, self.d_model
        )  # (batch_size * seq_len, d_model)

        # Compute gradients for W2 and b2
        dW2 = np.matmul(h_dropout_reshaped.T, dout_reshaped)  # (d_ff, d_model)
        db2 = np.sum(dout_reshaped, axis=0)  # (d_model,)

        # Gradient through second linear layer
        dh = np.matmul(dout_reshaped, self.W2.T)  # (batch_size * seq_len, d_ff)
        dh = dh.reshape(batch_size, seq_len, self.d_ff)  # (batch_size, seq_len, d_ff)

        # Gradient through dropout
        if self.dropout > 0:
            dh = dh * (h_dropout != 0) / (1 - self.dropout)

        # Gradient through ReLU
        dh_pre_relu = dh * (h_pre_relu > 0)  # (batch_size, seq_len, d_ff)

        # Gradient of loss w.r.t. first linear layer
        # Reshape tensors for matrix multiplication
        x_reshaped = x.reshape(-1, self.d_model)  # (batch_size * seq_len, d_model)
        dh_pre_relu_reshaped = dh_pre_relu.reshape(
            -1, self.d_ff
        )  # (batch_size * seq_len, d_ff)

        # Compute gradients for W1 and b1
        dW1 = np.matmul(x_reshaped.T, dh_pre_relu_reshaped)  # (d_model, d_ff)
        db1 = np.sum(dh_pre_relu_reshaped, axis=0)  # (d_ff,)

        # Gradient through first linear layer
        dx = np.matmul(
            dh_pre_relu_reshaped, self.W1.T
        )  # (batch_size * seq_len, d_model)
        dx = dx.reshape(
            batch_size, seq_len, self.d_model
        )  # (batch_size, seq_len, d_model)

        # Verify gradient shapes
        assert (
            dW1.shape == self.W1.shape
        ), f"dW1 shape {dW1.shape} != W1 shape {self.W1.shape}"
        assert (
            dW2.shape == self.W2.shape
        ), f"dW2 shape {dW2.shape} != W2 shape {self.W2.shape}"
        assert (
            db1.shape == self.b1.shape
        ), f"db1 shape {db1.shape} != b1 shape {self.b1.shape}"
        assert (
            db2.shape == self.b2.shape
        ), f"db2 shape {db2.shape} != b2 shape {self.b2.shape}"

        # Store gradients for parameter updates
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

        return dx


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.

        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            d_ff (int): Feed-forward dimension
            dropout (float): Dropout rate
        """
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        self.dropout = dropout

    def dropout_layer(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape) / (
                1 - self.dropout
            )
            return x * mask
        return x

    def forward(self, x, mask=None):
        """
        Forward pass of decoder layer with pre-normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional mask for attention
        """
        # Pre-norm: normalize before self-attention
        x_norm = self.norm1.forward(x)

        # Self-attention block
        attn_output, _ = self.self_attention.forward(x_norm, x_norm, x_norm, mask)
        attn_output = self.dropout_layer(attn_output)
        x = x + attn_output  # Residual connection after attention

        # Pre-norm: normalize before feed-forward
        x_norm = self.norm2.forward(x)

        # Feed-forward block
        ff_output = self.feed_forward.forward(x_norm)
        ff_output = self.dropout_layer(ff_output)
        x = x + ff_output  # Residual connection after feed-forward

        return x
