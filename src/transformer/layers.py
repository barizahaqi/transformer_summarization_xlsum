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
    
    def forward(self, x):
        """
        Forward pass of layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

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
    
    def dropout_layer(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1-self.dropout, size=x.shape) / (1-self.dropout)
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
        # First linear layer with ReLU
        h = np.matmul(x, self.W1) + self.b1
        h = self.relu(h)
        
        # Apply dropout
        h = self.dropout_layer(h)
        
        # Second linear layer
        output = np.matmul(h, self.W2) + self.b2
        
        return output

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
            mask = np.random.binomial(1, 1-self.dropout, size=x.shape) / (1-self.dropout)
            return x * mask
        return x
    
    def forward(self, x, mask=None):
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional mask for attention
        """
        # Self-attention block
        attn_output, _ = self.self_attention.forward(x, x, x, mask)
        attn_output = self.dropout_layer(attn_output)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward block
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout_layer(ff_output)
        x = self.norm2.forward(x + ff_output)
        
        return x 