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

        # Cache for backward pass
        self.cache = {}

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

        # Store for backward pass
        self.cache["matmul_qk"] = matmul_qk

        # Scale matmul_qk
        dk = np.sqrt(self.d_k)
        scaled_attention_logits = matmul_qk / dk

        # Store for backward pass
        self.cache["scaled_attention_logits"] = scaled_attention_logits

        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += mask * -1e9
            self.cache["mask"] = mask

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)

        # Store for backward pass
        self.cache["attention_weights"] = attention_weights

        # Apply dropout
        if self.dropout > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout, size=attention_weights.shape
            ) / (1 - self.dropout)
            attention_weights = attention_weights * dropout_mask
            self.cache["dropout_mask"] = dropout_mask

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

        # Store inputs for backward pass
        self.cache["q"] = q
        self.cache["k"] = k
        self.cache["v"] = v

        # Linear projections and split into heads
        q = np.matmul(q, self.W_q) + self.b_q
        k = np.matmul(k, self.W_k) + self.b_k
        v = np.matmul(v, self.W_v) + self.b_v

        # Store projections for backward pass
        self.cache["q_proj"] = q
        self.cache["k_proj"] = k
        self.cache["v_proj"] = v

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Store split heads for backward pass
        self.cache["q_heads"] = q
        self.cache["k_heads"] = k
        self.cache["v_heads"] = v

        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        # Combine heads
        concat_attention = self.combine_heads(scaled_attention, batch_size)

        # Store for backward pass
        self.cache["concat_attention"] = concat_attention

        # Final linear projection
        output = np.matmul(concat_attention, self.W_o) + self.b_o

        return output, attention_weights

    def backward(self, dout):
        """
        Backward pass of multi-head attention.

        Args:
            dout: Gradient of loss w.r.t. output of shape (batch_size, seq_length, d_model)
        """
        # Get the expected shapes from cached values
        expected_batch_size = self.cache["q"].shape[0]
        expected_seq_len = self.cache["q"].shape[1]
        expected_d_model = self.cache["q"].shape[2]

        # Ensure dout has the correct shape and dimensions
        if len(dout.shape) == 4:
            # If dout has an extra dimension, reshape it
            dout = dout.reshape(-1, dout.shape[-2], dout.shape[-1])

        # Handle batch size mismatch
        if dout.shape[0] != expected_batch_size:
            # If batch size doesn't match, reshape to expected batch size
            total_samples = dout.shape[0]
            if total_samples % expected_batch_size == 0:
                # Reshape by combining multiple samples into the expected batch size
                dout = dout.reshape(expected_batch_size, -1, dout.shape[-1])
                # Average the gradients across the combined samples
                dout = dout.mean(axis=1, keepdims=True).repeat(
                    dout.shape[1] // expected_batch_size, axis=1
                )
            else:
                raise ValueError(
                    f"Cannot reshape dout from shape {dout.shape} to batch size {expected_batch_size}"
                )

        # Handle sequence length mismatch
        if dout.shape[1] != expected_seq_len:
            # If sequence length doesn't match, we need to handle it
            if dout.shape[1] < expected_seq_len:
                # If sequence is shorter, pad with zeros
                padding = np.zeros(
                    (dout.shape[0], expected_seq_len - dout.shape[1], dout.shape[2])
                )
                dout = np.concatenate([dout, padding], axis=1)
            else:
                # If sequence is longer, truncate
                dout = dout[:, :expected_seq_len, :]

        # Handle model dimension mismatch
        if dout.shape[2] != expected_d_model:
            raise ValueError(
                f"Model dimension mismatch: got {dout.shape[2]}, expected {expected_d_model}"
            )

        batch_size = dout.shape[0]
        seq_len_q = dout.shape[1]  # Get sequence length from input

        # Get cached values
        q = self.cache["q"]
        k = self.cache["k"]
        v = self.cache["v"]
        q_proj = self.cache["q_proj"]
        k_proj = self.cache["k_proj"]
        v_proj = self.cache["v_proj"]
        q_heads = self.cache["q_heads"]  # (batch, heads, seq_len_q, d_k)
        k_heads = self.cache["k_heads"]  # (batch, heads, seq_len_k, d_k)
        v_heads = self.cache["v_heads"]  # (batch, heads, seq_len_v, d_k)
        concat_attention = self.cache["concat_attention"]
        attention_weights = self.cache[
            "attention_weights"
        ]  # (batch, heads, seq_len_q, seq_len_k)

        # Get sequence lengths from cached tensors
        seq_len_k = attention_weights.shape[
            -1
        ]  # Get key sequence length from attention weights
        seq_len_v = v_heads.shape[-2]  # Get value sequence length from v_heads

        # Verify all dimensions match
        assert dout.shape == (
            expected_batch_size,
            expected_seq_len,
            expected_d_model,
        ), f"Shape mismatch: got {dout.shape}, expected {(expected_batch_size, expected_seq_len, expected_d_model)}"
        assert concat_attention.shape == (
            expected_batch_size,
            expected_seq_len,
            expected_d_model,
        ), f"concat_attention shape mismatch: got {concat_attention.shape}, expected {(expected_batch_size, expected_seq_len, expected_d_model)}"

        # Gradient of loss w.r.t. output projection
        dW_o = np.matmul(
            concat_attention.transpose(0, 2, 1), dout
        )  # (d_model, d_model)
        db_o = np.sum(dout, axis=(0, 1))  # (d_model,)
        dconcat = np.matmul(dout, self.W_o.T)  # (batch, seq_len_q, d_model)

        # Verify dconcat shape
        assert dconcat.shape == (
            batch_size,
            seq_len_q,
            self.d_model,
        ), f"dconcat shape mismatch: got {dconcat.shape}, expected {(batch_size, seq_len_q, self.d_model)}"

        # Gradient through head combination
        # Ensure dconcat is properly reshaped
        dscaled_attention = dconcat.reshape(
            batch_size, -1, self.num_heads, self.d_k
        )  # (batch, seq_len_q, heads, d_k)
        dscaled_attention = np.transpose(
            dscaled_attention, (0, 2, 1, 3)
        )  # (batch, heads, seq_len_q, d_k)

        # Verify dscaled_attention shape
        assert dscaled_attention.shape == (
            batch_size,
            self.num_heads,
            seq_len_q,
            self.d_k,
        ), f"dscaled_attention shape mismatch: got {dscaled_attention.shape}, expected {(batch_size, self.num_heads, seq_len_q, self.d_k)}"

        # Reshape tensors to combine batch and head dimensions
        attention_weights_reshaped = attention_weights.reshape(
            -1, seq_len_q, seq_len_k
        )  # (batch*heads, seq_len_q, seq_len_k)
        dscaled_attention_reshaped = dscaled_attention.reshape(
            -1, seq_len_q, self.d_k
        )  # (batch*heads, seq_len_q, d_k)
        v_heads_reshaped = v_heads.reshape(
            -1, seq_len_v, self.d_k
        )  # (batch*heads, seq_len_v, d_k)

        # Verify the number of heads matches
        num_heads = batch_size * self.num_heads
        assert (
            attention_weights_reshaped.shape[0]
            == dscaled_attention_reshaped.shape[0]
            == v_heads_reshaped.shape[0]
            == num_heads
        ), (
            f"Number of heads mismatch: attention_weights={attention_weights_reshaped.shape[0]}, "
            f"dscaled_attention={dscaled_attention_reshaped.shape[0]}, v_heads={v_heads_reshaped.shape[0]}, "
            f"expected={num_heads}"
        )

        # Compute gradients through attention weights
        dv_heads_reshaped = np.zeros_like(
            v_heads_reshaped
        )  # (batch*heads, seq_len_v, d_k)
        dattention_weights_reshaped = np.zeros_like(
            attention_weights_reshaped
        )  # (batch*heads, seq_len_q, seq_len_k)

        # Process each head separately to avoid broadcasting issues
        for i in range(num_heads):
            # For v_heads gradient:
            # attention_weights[i].T: (seq_len_k, seq_len_q)
            # dscaled_attention[i]: (seq_len_q, d_k)
            # Result should be: (seq_len_k, d_k)
            # But we need (seq_len_v, d_k) for v_heads

            # First compute the gradient through attention weights
            dattention_weights_reshaped[i] = np.matmul(
                dscaled_attention_reshaped[i],  # (seq_len_q, d_k)
                v_heads_reshaped[i].T,  # (d_k, seq_len_v)
            )  # Result: (seq_len_q, seq_len_v)

            # Then compute the gradient for v_heads
            # We need to ensure the sequence lengths match
            if seq_len_k == seq_len_v:
                # If sequence lengths match, we can directly compute
                dv_heads_reshaped[i] = np.matmul(
                    attention_weights_reshaped[i].T,  # (seq_len_k, seq_len_q)
                    dscaled_attention_reshaped[i],  # (seq_len_q, d_k)
                )  # Result: (seq_len_k, d_k)
            else:
                # If sequence lengths don't match, we need to handle it differently
                # For now, we'll use a simple approach: take the first min(seq_len_k, seq_len_v) positions
                min_len = min(seq_len_k, seq_len_v)
                dv_heads_reshaped[i, :min_len] = np.matmul(
                    attention_weights_reshaped[
                        i, :, :min_len
                    ].T,  # (min_len, seq_len_q)
                    dscaled_attention_reshaped[i],  # (seq_len_q, d_k)
                )  # Result: (min_len, d_k)

        # Reshape back to original dimensions
        dv_heads = dv_heads_reshaped.reshape(
            batch_size, self.num_heads, seq_len_v, self.d_k
        )
        dattention_weights = dattention_weights_reshaped.reshape(
            batch_size, self.num_heads, seq_len_q, seq_len_v
        )

        # Gradient through dropout
        if self.dropout > 0 and "dropout_mask" in self.cache:
            dattention_weights *= self.cache["dropout_mask"]

        # Gradient through softmax
        dscaled_logits = (
            dattention_weights * attention_weights * (1 - attention_weights)
        )  # (batch, heads, seq_len_q, seq_len_k)

        # Gradient through scaling
        dmatmul_qk = dscaled_logits / np.sqrt(
            self.d_k
        )  # (batch, heads, seq_len_q, seq_len_k)

        # Gradient through QK multiplication
        # Reshape for batch matrix multiplication
        dmatmul_qk_reshaped = dmatmul_qk.reshape(
            -1, seq_len_q, seq_len_k
        )  # (batch*heads, seq_len_q, seq_len_k)
        k_heads_reshaped = k_heads.reshape(
            -1, seq_len_k, self.d_k
        )  # (batch*heads, seq_len_k, d_k)
        q_heads_reshaped = q_heads.reshape(
            -1, seq_len_q, self.d_k
        )  # (batch*heads, seq_len_q, d_k)

        # Compute gradients for each head separately
        dq_heads_reshaped = np.zeros_like(
            q_heads_reshaped
        )  # (batch*heads, seq_len_q, d_k)
        dk_heads_reshaped = np.zeros_like(
            k_heads_reshaped
        )  # (batch*heads, seq_len_k, d_k)

        for i in range(num_heads):
            # Compute gradients for q_heads
            dq_heads_reshaped[i] = np.matmul(
                dmatmul_qk_reshaped[i],  # (seq_len_q, seq_len_k)
                k_heads_reshaped[i],  # (seq_len_k, d_k)
            )  # Result: (seq_len_q, d_k)

            # Compute gradients for k_heads
            dk_heads_reshaped[i] = np.matmul(
                dmatmul_qk_reshaped[i].T,  # (seq_len_k, seq_len_q)
                q_heads_reshaped[i],  # (seq_len_q, d_k)
            )  # Result: (seq_len_k, d_k)

        # Reshape back to original dimensions
        dq_heads = dq_heads_reshaped.reshape(
            batch_size, self.num_heads, seq_len_q, self.d_k
        )
        dk_heads = dk_heads_reshaped.reshape(
            batch_size, self.num_heads, seq_len_k, self.d_k
        )

        # Gradient through head splitting
        dq_proj = self.combine_heads(dq_heads, batch_size)
        dk_proj = self.combine_heads(dk_heads, batch_size)
        dv_proj = self.combine_heads(dv_heads, batch_size)

        # Gradient through linear projections
        dW_q = np.matmul(q.transpose(0, 2, 1), dq_proj)
        db_q = np.sum(dq_proj, axis=(0, 1))
        dq = np.matmul(dq_proj, self.W_q.T)

        dW_k = np.matmul(k.transpose(0, 2, 1), dk_proj)
        db_k = np.sum(dk_proj, axis=(0, 1))
        dk = np.matmul(dk_proj, self.W_k.T)

        dW_v = np.matmul(v.transpose(0, 2, 1), dv_proj)
        db_v = np.sum(dv_proj, axis=(0, 1))
        dv = np.matmul(dv_proj, self.W_v.T)

        # Store gradients for parameter updates
        self.dW_q = dW_q
        self.db_q = db_q
        self.dW_k = dW_k
        self.db_k = db_k
        self.dW_v = dW_v
        self.db_v = db_v
        self.dW_o = dW_o
        self.db_o = db_o

        return dq, dk, dv
