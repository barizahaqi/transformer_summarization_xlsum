import numpy as np
from .embeddings import Embeddings
from .layers import DecoderLayer


class TransformerDecoder:
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=5000,
        dropout=0.1,
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size (int): Size of vocabulary
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of decoder layers
            d_ff (int): Feed-forward dimension
            max_seq_length (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        self.d_model = d_model
        self.num_layers = num_layers

        # Initialize embeddings
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_length, dropout)

        # Initialize decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]

        # Initialize output layer
        self.output_layer = np.random.normal(0, 0.02, (d_model, vocab_size))
        self.output_bias = np.zeros(vocab_size)

        self.dropout = dropout

    def dropout_layer(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape) / (
                1 - self.dropout
            )
            return x * mask
        return x

    def create_mask(self, seq):
        """
        Create causal mask for decoder.

        Args:
            seq: Input sequence of shape (batch_size, seq_length)
        """
        seq_len = seq.shape[1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32)
        mask = (mask == 0).astype(np.float32)
        return mask[np.newaxis, np.newaxis, :, :]

    def forward(self, x, training=True):
        """
        Forward pass of transformer decoder.

        Args:
            x: Input tensor of shape (batch_size, seq_length)
            training (bool): Whether in training mode
        """
        # Create causal mask
        mask = self.create_mask(x)

        # Get embeddings
        x = self.embeddings.forward(x)

        # Apply dropout if training
        if training:
            x = self.dropout_layer(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, mask)

        # Output layer
        logits = np.matmul(x, self.output_layer) + self.output_bias

        return logits

    def generate(self, start_token, max_length, temperature=1.0):
        """
        Generate sequence using the decoder.

        Args:
            start_token (int): Starting token index
            max_length (int): Maximum sequence length to generate
            temperature (float): Sampling temperature
        """
        # Initialize sequence with start token
        seq = np.array([[start_token]])

        for _ in range(max_length - 1):
            # Get model predictions
            logits = self.forward(seq, training=False)

            # Get next token probabilities
            next_token_logits = logits[:, -1, :] / temperature
            probs = self.softmax(next_token_logits)

            # Sample next token
            next_token = np.random.choice(len(probs[0]), p=probs[0])

            # Append to sequence
            seq = np.append(seq, [[next_token]], axis=1)

            # Stop if we predict the end token
            if next_token == 1:  # Assuming 1 is the end token
                break

        return seq

    def softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def compute_loss(self, logits, targets):
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions of shape (batch_size, seq_length, vocab_size)
            targets: Target sequences of shape (batch_size, seq_length)
        """
        # Reshape for loss computation
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

        # Compute cross-entropy loss
        log_probs = self.log_softmax(logits)
        nll_loss = -np.sum(log_probs[np.arange(len(targets)), targets]) / len(targets)

        return nll_loss

    def log_softmax(self, x):
        """Compute log softmax values for each set of scores in x."""
        x_max = np.max(x, axis=-1, keepdims=True)
        return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
