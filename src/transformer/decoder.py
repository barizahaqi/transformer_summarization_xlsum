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
            x: Input tensor of shape (batch_size, seq_length) containing <BOS> article <SEP> summary <EOS>
            training (bool): Whether in training mode
        """
        # Ensure input is integer type for embedding lookup
        x = x.astype(np.int64)

        # Create causal mask to prevent looking at future tokens
        # This ensures each position can only attend to previous positions
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

    def generate(
        self, input_article_tokens, word2idx, max_length=None, temperature=1.0
    ):
        """
        Generate summary tokens given input article tokens.

        Args:
            input_article_tokens: Array of input article token indices
            word2idx (dict): Dictionary mapping tokens to indices
            max_length (int, optional): Maximum length of generated sequence. If None, uses model's max_seq_length.
            temperature (float): Sampling temperature (higher = more random)

        Returns:
            numpy.ndarray: Generated token indices
        """
        if max_length is None:
            max_length = self.max_seq_length

        # Get special token indices from the dataset's tokenizer
        bos_idx = word2idx["<BOS>"]  # Get actual <BOS> token index
        sep_idx = word2idx["<SEP>"]  # Get actual <SEP> token index
        eos_idx = word2idx["<EOS>"]  # Get actual <EOS> token index
        pad_idx = word2idx["<PAD>"]  # Get actual <PAD> token index

        # Find where the article ends (before <SEP> token)
        try:
            sep_pos = np.where(input_article_tokens == sep_idx)[0][0]
            # Only use tokens up to <SEP> (exclusive)
            article_tokens = input_article_tokens[:sep_pos]
        except IndexError:
            # If no <SEP> found, use the whole sequence
            article_tokens = input_article_tokens

        # Create initial sequence: <BOS> article <SEP>
        sequence = np.concatenate(
            [
                np.array([bos_idx]),  # <BOS>
                article_tokens,  # Article tokens
                np.array([sep_idx]),  # <SEP>
            ]
        )

        # Pad sequence to max_length if needed
        if len(sequence) < max_length:
            padding = np.full(max_length - len(sequence), pad_idx)
            sequence = np.concatenate([sequence, padding])
        else:
            sequence = sequence[:max_length]

        # Generate summary tokens
        start_pos = len(article_tokens) + 2  # Start after <BOS>, article, and <SEP>

        for i in range(start_pos, max_length):
            # Get model predictions
            logits = self.forward(sequence.reshape(1, -1), training=False)
            next_token_logits = logits[0, i - 1]  # Get logits for next token

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Sample from the distribution
            probs = self.softmax(next_token_logits)
            next_token = np.random.choice(len(probs), p=probs)

            # Update sequence
            sequence[i] = next_token

            # Stop if we generate <EOS>
            if next_token == eos_idx:
                break

        return sequence

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

        # Ensure targets are integers
        targets = targets.astype(np.int64)

        # Reshape for loss computation
        logits = logits.reshape(
            -1, logits.shape[-1]
        )  # (batch_size * seq_length, vocab_size)
        targets = targets.reshape(-1)  # (batch_size * seq_length,)

        # Compute cross-entropy loss
        log_probs = self.log_softmax(logits)

        # Create index array for gathering target log probabilities
        batch_indices = np.arange(len(targets), dtype=np.int64)

        # Gather target log probabilities and compute loss
        target_log_probs = log_probs[batch_indices, targets]
        nll_loss = -np.sum(target_log_probs) / len(targets)

        return nll_loss

    def log_softmax(self, x):
        """Compute log softmax values for each set of scores in x."""
        x_max = np.max(x, axis=-1, keepdims=True)
        return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
