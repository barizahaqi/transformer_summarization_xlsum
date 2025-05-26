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
        # Get token embeddings
        embeddings = self.token_embeddings[
            x
        ]  # Shape: (batch_size, seq_length, d_model)

        # Scale embeddings
        embeddings = embeddings * np.sqrt(self.d_model)

        # Add positional encoding
        embeddings = self.positional_encoding.forward(embeddings)

        # Apply dropout
        embeddings = self.dropout_layer(embeddings)

        return embeddings
