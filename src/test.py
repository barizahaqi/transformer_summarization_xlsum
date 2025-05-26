import numpy as np
from tqdm import tqdm
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset


def load_model(model_path, dataset):
    """
    Load a trained model from saved weights.

    Args:
        model_path (str): Path to saved model weights
        dataset: XLSumDataset instance

    Returns:
        TransformerDecoder: Loaded model
    """
    # Initialize model with same parameters as training
    model = TransformerDecoder(
        vocab_size=len(dataset.tokenizer.word2idx),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=64,
    )

    # Load weights
    try:
        weights = np.load(model_path, allow_pickle=True).item()
        print("Loading model weights...")

        # Load weights
        model.output_layer = weights["output_layer"]
        model.output_bias = weights["output_bias"]
        model.embeddings.token_embeddings = weights["embeddings"]

        # Load decoder layers
        for i, layer_weights in enumerate(weights["decoder_layers"]):
            layer = model.decoder_layers[i]
            # Load attention weights
            layer.self_attention.W_q = layer_weights["self_attention"]["W_q"]
            layer.self_attention.W_k = layer_weights["self_attention"]["W_k"]
            layer.self_attention.W_v = layer_weights["self_attention"]["W_v"]
            layer.self_attention.W_o = layer_weights["self_attention"]["W_o"]
            # Load feed-forward weights
            layer.feed_forward.W1 = layer_weights["feed_forward"]["W1"]
            layer.feed_forward.W2 = layer_weights["feed_forward"]["W2"]

        print("Successfully loaded model weights")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No model weights found at {model_path}")


def evaluate_model(model, dataset, split="test"):
    """
    Evaluate model on a dataset split.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        split (str): Dataset split to evaluate on

    Returns:
        float: Average loss on the split
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


def generate_summaries(model, dataset, num_examples=5, temperature=0.7):
    """
    Generate summaries for example texts.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        num_examples (int): Number of examples to generate summaries for
        temperature (float): Sampling temperature for generation

    Returns:
        list: List of dictionaries containing original text, original summary, and generated summary
    """
    # Get examples from test set
    text_batch, summary_batch = dataset.get_batch("test")

    results = []
    for i in range(min(num_examples, len(text_batch))):
        # Get original text and summary
        original_text = dataset.decode_batch([text_batch[i]])[0]
        original_summary = dataset.decode_batch([summary_batch[i]])[0]

        # Generate summary
        generated_tokens = model.generate(
            start_token=dataset.tokenizer.word2idx[dataset.tokenizer.bos_token],
            max_length=dataset.max_seq_length,
            temperature=temperature,
        )
        generated_summary = dataset.decode_batch([generated_tokens[0]])[0]

        results.append(
            {
                "original_text": original_text,
                "original_summary": original_summary,
                "generated_summary": generated_summary,
            }
        )

    return results


def print_results(results):
    """
    Print generation results in a readable format.

    Args:
        results: List of dictionaries containing generation results
    """
    print("\nGeneration Results:")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nExample {i}:")
        print(f"Original Text: {result['original_text'][:200]}...")
        print(f"Original Summary: {result['original_summary']}")
        print(f"Generated Summary: {result['generated_summary']}")
        print("-" * 80)


def main():
    # Initialize dataset
    print("Loading dataset...")
    dataset = XLSumDataset(max_seq_length=64, batch_size=32, vocab_size=8000)

    # Load model
    model = load_model("best_model_weights.npy", dataset)

    # Evaluate model
    print("\nEvaluating model...")
    test_loss = evaluate_model(model, dataset, "test")
    print(f"Test Loss: {test_loss:.4f}")

    # Generate and print summaries
    print("\nGenerating summaries...")
    results = generate_summaries(model, dataset, num_examples=5, temperature=0.7)
    print_results(results)


if __name__ == "__main__":
    main()
