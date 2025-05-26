import numpy as np
import time
from tqdm import tqdm
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset


def train_epoch(model, dataset, learning_rate=1e-4):
    """
    Train for one epoch.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        learning_rate (float): Learning rate
    """
    total_loss = 0
    num_batches = len(dataset.train_data) // dataset.batch_size

    for _ in tqdm(range(num_batches), desc="Training"):
        # Get batch
        text_batch, summary_batch = dataset.get_batch("train")

        # Forward pass
        logits = model.forward(text_batch)

        # Compute loss
        loss = model.compute_loss(logits, summary_batch)
        total_loss += loss

        # Backward pass (gradient computation)
        gradients = compute_gradients(model, logits, summary_batch)

        # Update parameters
        update_parameters(model, gradients, learning_rate)

    return total_loss / num_batches


def compute_gradients(model, logits, targets):
    """
    Compute gradients using backpropagation.

    Args:
        model: TransformerDecoder model
        logits: Model predictions
        targets: Target sequences
    """
    # Initialize gradients dictionary
    gradients = {}

    # Compute gradients for output layer
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Gradient of loss w.r.t. logits
    probs = model.softmax(logits_flat)
    probs[np.arange(len(targets_flat)), targets_flat] -= 1
    d_logits = probs / batch_size

    # Gradient of loss w.r.t. output layer weights
    gradients["output_layer"] = np.matmul(
        model.embeddings.token_embeddings[targets].reshape(-1, model.d_model).T,
        d_logits,
    )
    gradients["output_bias"] = np.sum(d_logits, axis=0)

    # Backpropagate through decoder layers
    d_h = d_logits.reshape(batch_size, seq_len, vocab_size)
    d_h = np.matmul(d_h, model.output_layer.T)

    for i in range(len(model.decoder_layers) - 1, -1, -1):
        layer = model.decoder_layers[i]

        # Backpropagate through feed-forward network
        d_ff = d_h
        d_ff = layer.norm2.backward(d_ff)
        d_ff = layer.feed_forward.backward(d_ff)
        d_h = d_h + d_ff

        # Backpropagate through self-attention
        d_attn = d_h
        d_attn = layer.norm1.backward(d_attn)
        d_attn = layer.self_attention.backward(d_attn)
        d_h = d_h + d_attn

    # Backpropagate through embeddings
    d_emb = d_h
    d_emb = model.embeddings.backward(d_emb)

    return gradients


def update_parameters(model, gradients, learning_rate):
    """
    Update model parameters using gradients.

    Args:
        model: TransformerDecoder model
        gradients: Dictionary of gradients
        learning_rate (float): Learning rate
    """
    # Update output layer
    model.output_layer -= learning_rate * gradients["output_layer"]
    model.output_bias -= learning_rate * gradients["output_bias"]

    # Update embeddings
    model.embeddings.token_embeddings -= learning_rate * gradients.get("embeddings", 0)

    # Update decoder layers
    for layer in model.decoder_layers:
        # Update self-attention
        layer.self_attention.W_q -= learning_rate * gradients.get(
            f"attn_{layer}_W_q", 0
        )
        layer.self_attention.W_k -= learning_rate * gradients.get(
            f"attn_{layer}_W_k", 0
        )
        layer.self_attention.W_v -= learning_rate * gradients.get(
            f"attn_{layer}_W_v", 0
        )
        layer.self_attention.W_o -= learning_rate * gradients.get(
            f"attn_{layer}_W_o", 0
        )

        # Update feed-forward network
        layer.feed_forward.W1 -= learning_rate * gradients.get(f"ff_{layer}_W1", 0)
        layer.feed_forward.W2 -= learning_rate * gradients.get(f"ff_{layer}_W2", 0)


def evaluate(model, dataset, split="validation"):
    """
    Evaluate model on validation/test set.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        split (str): Dataset split to evaluate on
    """
    total_loss = 0
    num_batches = len(getattr(dataset, f"{split}_data")) // dataset.batch_size

    for _ in tqdm(range(num_batches), desc=f"Evaluating on {split}"):
        # Get batch
        text_batch, summary_batch = dataset.get_batch(split)

        # Forward pass
        logits = model.forward(text_batch, training=False)

        # Compute loss
        loss = model.compute_loss(logits, summary_batch)
        total_loss += loss

    return total_loss / num_batches


def main():
    # Hyperparameters
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 256
    max_seq_length = 64
    batch_size = 32
    vocab_size = 8000
    num_epochs = 10
    learning_rate = 1e-4

    # Initialize dataset
    print("Loading dataset...")
    dataset = XLSumDataset(
        max_seq_length=max_seq_length, batch_size=batch_size, vocab_size=vocab_size
    )

    # Initialize model
    print("Initializing model...")
    model = TransformerDecoder(
        vocab_size=len(dataset.tokenizer.word2idx),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
    )

    # Print model size
    total_params = (
        vocab_size * d_model  # Embeddings
        + d_model * vocab_size  # Output layer
        + num_layers
        * (
            3 * (d_model * d_model)  # Q, K, V matrices
            + d_model * d_model  # Output matrix
            + d_model * d_ff  # FF W1
            + d_ff * d_model  # FF W2
        )
    )
    print(f"\nModel Configuration:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Number of layers: {num_layers}")
    print(f"Feed-forward dimension: {d_ff}")
    print(f"Maximum sequence length: {max_seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"Total parameters: {total_params:,}")

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, dataset, learning_rate)

        # Evaluate
        val_loss = evaluate(model, dataset, "validation")

        # Print epoch statistics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weights
            np.save(
                "best_model_weights.npy",
                {
                    "output_layer": model.output_layer,
                    "output_bias": model.output_bias,
                    "embeddings": model.embeddings.token_embeddings,
                    "decoder_layers": [
                        {
                            "self_attention": {
                                "W_q": layer.self_attention.W_q,
                                "W_k": layer.self_attention.W_k,
                                "W_v": layer.self_attention.W_v,
                                "W_o": layer.self_attention.W_o,
                            },
                            "feed_forward": {
                                "W1": layer.feed_forward.W1,
                                "W2": layer.feed_forward.W2,
                            },
                        }
                        for layer in model.decoder_layers
                    ],
                },
            )
            print("Saved best model weights")


if __name__ == "__main__":
    main()
