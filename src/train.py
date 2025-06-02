import numpy as np
import time
from tqdm import tqdm
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset


def train_epoch(model, dataset, learning_rate=1e-4):
    """
    Train model for one epoch.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        learning_rate (float): Learning rate for gradient descent
    """
    num_batches = len(dataset.train_data) // dataset.batch_size
    total_loss = 0

    for batch_idx in tqdm(range(num_batches), desc="Training"):
        # Get batch of sequences in format <BOS> article <BOS> summary <EOS>
        sequences, targets = dataset.get_batch("train")

        # Forward pass
        logits = model.forward(sequences, training=True)

        # Compute loss
        loss = model.compute_loss(logits, targets)
        total_loss += loss

        # Compute gradients
        gradients = compute_gradients(model, logits, targets)

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
    targets_flat = targets.reshape(-1).astype(np.int32)  # Convert to int32

    # Gradient of loss w.r.t. logits
    probs = model.softmax(logits_flat)
    probs[
        np.arange(len(targets_flat), dtype=np.int32), targets_flat
    ] -= 1  # Use int32 for indices
    d_logits = probs / batch_size

    # Gradient of loss w.r.t. output layer weights
    gradients["output_layer"] = np.matmul(
        model.embeddings.token_embeddings[targets.astype(np.int32)]
        .reshape(-1, model.d_model)
        .T,  # Convert to int32
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
    for i, layer in enumerate(model.decoder_layers):
        # Update layer normalization parameters
        layer.norm1.gamma -= learning_rate * layer.norm1.dgamma
        layer.norm1.beta -= learning_rate * layer.norm1.dbeta
        layer.norm2.gamma -= learning_rate * layer.norm2.dgamma
        layer.norm2.beta -= learning_rate * layer.norm2.dbeta

        # Update feed-forward network
        layer.feed_forward.W1 -= learning_rate * layer.feed_forward.dW1
        layer.feed_forward.b1 -= learning_rate * layer.feed_forward.db1
        layer.feed_forward.W2 -= learning_rate * layer.feed_forward.dW2
        layer.feed_forward.b2 -= learning_rate * layer.feed_forward.db2

        # Update self-attention
        layer.self_attention.W_q -= learning_rate * gradients.get(f"attn_{i}_W_q", 0)
        layer.self_attention.W_k -= learning_rate * gradients.get(f"attn_{i}_W_k", 0)
        layer.self_attention.W_v -= learning_rate * gradients.get(f"attn_{i}_W_v", 0)
        layer.self_attention.W_o -= learning_rate * gradients.get(f"attn_{i}_W_o", 0)


def evaluate(model, dataset, split="val"):
    """
    Evaluate model on validation/test set.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        split (str): Which split to evaluate on ('val' or 'test')

    Returns:
        float: Average loss on the evaluation set
    """
    # Map split name to dataset attribute
    split_map = {"val": "val_data", "test": "test_data"}
    data_attr = split_map[split]

    num_batches = len(getattr(dataset, data_attr)) // dataset.batch_size
    total_loss = 0

    for _ in tqdm(range(num_batches), desc=f"Evaluating on {split}"):
        # Get batch of sequences
        sequences, targets = dataset.get_batch(split)

        # Forward pass (no training)
        logits = model.forward(sequences, training=False)

        # Compute loss
        loss = model.compute_loss(logits, targets)
        total_loss += loss

    return total_loss / num_batches


def main():
    # Model configurations
    model_configs = [
        {
            "name": "model_128d",
            "d_model": 128,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 75,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 3200,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "max_samples": 100,  # Use 1000 samples for each split
        },
        {
            "name": "model_8k_vocab",
            "d_model": 56,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 96,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 8000,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "max_samples": 100,  # Use 1000 samples for each split
        },
        {
            "name": "model_1layer",
            "d_model": 88,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 352,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 5000,
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "max_samples": 100,
        },
    ]

    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Training {config['name']}")
        print(f"{'='*50}")

        # Initialize dataset
        print(f"Initializing dataset for {config['name']}...")
        dataset = XLSumDataset(
            max_seq_length=config["max_seq_length"],
            batch_size=config["batch_size"],
            vocab_size=config["vocab_size"],
            model_name=config["name"],
            max_samples=config["max_samples"],  # Pass max_samples to dataset
        )

        # Initialize model
        print("Initializing model...")
        model = TransformerDecoder(
            vocab_size=len(dataset.tokenizer.word2idx),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_seq_length=config["max_seq_length"],
        )

        # Print model size
        total_params = (
            config["vocab_size"] * config["d_model"]  # Embeddings
            + config["d_model"] * config["vocab_size"]  # Output layer
            + config["num_layers"]
            * (
                3 * (config["d_model"] * config["d_model"])  # Q, K, V matrices
                + config["d_model"] * config["d_model"]  # Output matrix
                + config["d_model"] * config["d_ff"]  # FF W1
                + config["d_ff"] * config["d_model"]  # FF W2
            )
        )

        # Create configuration text
        config_text = f"""
Model Configuration:
Model name: {config['name']}
Vocabulary size: {config['vocab_size']}
Embedding dimension: {config['d_model']}
Number of attention heads: {config['num_heads']}
Number of layers: {config['num_layers']}
Feed-forward dimension: {config['d_ff']}
Maximum sequence length: {config['max_seq_length']}
Batch size: {config['batch_size']}
Total parameters: {total_params:,}
"""

        # Print to console
        print(config_text)

        # Save to file
        with open(f"model_config_{config['name']}.txt", "w") as f:
            f.write(config_text)

        # Training loop
        print("Starting training...")
        best_val_loss = float("inf")

        for epoch in range(config["num_epochs"]):
            start_time = time.time()

            # Train
            train_loss = train_epoch(model, dataset, config["learning_rate"])

            # Evaluate on validation set
            val_loss = evaluate(model, dataset, "val")

            # Print epoch statistics
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model weights
                np.save(
                    f"best_model_weights_{config['name']}.npy",
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
                print(f"Saved best model weights for {config['name']}")


if __name__ == "__main__":
    main()
