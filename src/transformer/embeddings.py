import numpy as np


class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Model dimension
            max_seq_length (int): Maximum sequence length
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe[np.newaxis, :, :]  # Shape: (1, max_seq_length, d_model)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        return x + self.pe[:, : x.shape[1], :]


class Embeddings:
    def __init__(self, vocab_size, d_model, max_seq_length=5000, dropout=0.1):
        """
        Initialize token embeddings and positional encoding.

        Args:
            vocab_size (int): Size of vocabulary
            d_model (int): Model dimension
            max_seq_length (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Initialize token embeddings
        self.token_embeddings = np.random.normal(0, 0.02, (vocab_size, d_model))

        # Initialize positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

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

    def forward(self, x):
        """
        Forward pass of embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing token indices
        """
        # Store input for backward pass
        self.cache["x"] = x

        # Get token embeddings
        embeddings = self.token_embeddings[
            x
        ]  # Shape: (batch_size, seq_length, d_model)

        # Store embeddings before scaling for backward pass
        self.cache["embeddings"] = embeddings

        # Scale embeddings
        embeddings = embeddings * np.sqrt(self.d_model)

        # Store scaled embeddings for backward pass
        self.cache["scaled_embeddings"] = embeddings

        # Add positional encoding
        embeddings = self.positional_encoding.forward(embeddings)

        # Store embeddings before dropout for backward pass
        self.cache["embeddings_before_dropout"] = embeddings

        # Apply dropout
        if self.dropout > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.dropout, size=embeddings.shape
            ) / (1 - self.dropout)
            embeddings = embeddings * dropout_mask
            self.cache["dropout_mask"] = dropout_mask

        return embeddings

    def backward(self, dout):
        """
        Backward pass of embeddings.

        Args:
            dout: Gradient of loss w.r.t. output of shape (batch_size, seq_length, d_model)
                 or (num_layers, batch_size, seq_length, d_model)
        """
        # Get cached values
        x = self.cache["x"]  # Input token indices
        dropout_mask = self.cache.get("dropout_mask")

        # Handle extra dimension if present
        if len(dout.shape) == 4:
            # If dout has shape (num_layers, batch_size, seq_length, d_model)
            # We need to sum the gradients across layers
            dout = np.sum(dout, axis=0)  # Shape: (batch_size, seq_length, d_model)

        # Verify dout has the correct shape
        assert (
            len(dout.shape) == 3
        ), f"Expected dout to have 3 dimensions, got {len(dout.shape)}"
        assert (
            dout.shape[2] == self.d_model
        ), f"Expected d_model dimension to be {self.d_model}, got {dout.shape[2]}"

        # Gradient through dropout
        if dropout_mask is not None:
            dout = dout * dropout_mask

        # Gradient through positional encoding (no parameters to update)
        # The positional encoding is deterministic, so we just pass the gradient through

        # Gradient through scaling
        dout = dout * np.sqrt(self.d_model)

        # Gradient through token embeddings
        # We need to accumulate gradients for each token in the vocabulary
        d_embeddings = np.zeros_like(
            self.token_embeddings
        )  # Shape: (vocab_size, d_model)

        # Get batch dimensions
        batch_size, seq_length = x.shape

        # For each unique token in the batch, accumulate its gradient
        unique_tokens = np.unique(x)
        for token in unique_tokens:
            # Create a mask for each position where this token appears
            # Shape: (batch_size, seq_length)
            token_mask = x == token

            # For each position where the token appears, add its gradient
            # We need to handle each position separately to avoid broadcasting issues
            for b in range(batch_size):
                for s in range(seq_length):
                    if token_mask[b, s]:
                        # Add the gradient for this position
                        d_embeddings[token] += dout[b, s]

        # Store gradient for parameter update
        self.d_token_embeddings = d_embeddings

        # Return gradient for input (not used since input is discrete tokens)
        return None
