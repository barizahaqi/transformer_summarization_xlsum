import numpy as np
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset
from rouge_score import rouge_scorer
import json
from datetime import datetime
from tqdm import tqdm


def load_model(model_path, dataset, config):
    """Load a trained model with its configuration."""
    try:
        # Initialize model with configuration
        model = TransformerDecoder(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            d_ff=config["d_ff"],
            max_seq_length=config["max_seq_length"],
        )

        # Load weights
        weights = np.load(model_path, allow_pickle=True).item()

        # Load weights into model
        model.output_layer = weights["output_layer"]
        model.output_bias = weights["output_bias"]
        model.embeddings.token_embeddings = weights["embeddings"]

        # Load decoder layers
        for i, layer_weights in enumerate(weights["decoder_layers"]):
            layer = model.decoder_layers[i]
            layer.self_attention.W_q = layer_weights["self_attention"]["W_q"]
            layer.self_attention.W_k = layer_weights["self_attention"]["W_k"]
            layer.self_attention.W_v = layer_weights["self_attention"]["W_v"]
            layer.self_attention.W_o = layer_weights["self_attention"]["W_o"]
            layer.feed_forward.W1 = layer_weights["feed_forward"]["W1"]
            layer.feed_forward.W2 = layer_weights["feed_forward"]["W2"]

        print(f"Successfully loaded model from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def evaluate_model(model, dataset, split="test", max_samples=None):
    """Evaluate model performance on a dataset split."""
    # Map split name to dataset attribute
    split_data = getattr(dataset, f"{split}_data")
    if max_samples:
        split_data = split_data[:max_samples]

    # Get number of batches
    num_batches = len(split_data) // dataset.batch_size
    if len(split_data) % dataset.batch_size != 0:
        num_batches += 1

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    total_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    num_samples = 0

    # Evaluate each batch
    for i in tqdm(range(num_batches), desc=f"Evaluating {split} set"):
        # Get input and target sequences using the split name
        input_sequences, target_sequences = dataset.get_batch(split=split)

        # Generate summaries
        generated_tokens = model.generate(
            input_article_tokens=input_sequences[0],  # Take first sequence in batch
            word2idx=dataset.tokenizer.word2idx,
            max_length=dataset.max_seq_length,
            temperature=0.7,
        )

        # Decode reference and generated summaries
        reference_decoded = dataset.decode_batch(target_sequences)
        generated_decoded = dataset.decode_batch(generated_tokens)

        # Calculate ROUGE scores for each pair
        for ref, gen in zip(reference_decoded, generated_decoded):
            # Extract summary text from dictionaries and convert to lowercase
            ref_summary = ref["summary"].lower()
            gen_summary = gen["summary"].lower()

            # Calculate ROUGE scores
            scores = scorer.score(ref_summary, gen_summary)
            for metric in total_scores:
                total_scores[metric] += getattr(scores[metric], "fmeasure")
            num_samples += 1

    # Calculate average scores
    avg_scores = {metric: score / num_samples for metric, score in total_scores.items()}
    return avg_scores


def generate_summaries(model, dataset, num_examples=5, temperature=0.7):
    """Generate summaries for a few examples from the test set."""
    results = []

    # Get examples from test set
    test_examples = dataset.test_data[:num_examples]

    for example in test_examples:
        # Get input sequence using the test split
        input_sequence, _ = dataset.get_batch(split="test")

        # Generate summary
        generated_tokens = model.generate(
            input_article_tokens=input_sequence[0],
            word2idx=dataset.tokenizer.word2idx,
            max_length=dataset.max_seq_length,
            temperature=temperature,
        )

        # Decode sequences
        decoded_input = dataset.decode_batch(input_sequence)[0]
        decoded_output = dataset.decode_batch(generated_tokens)[0]

        results.append(
            {
                "original_text": decoded_input["article"],
                "original_summary": example["summary"],
                "generated_summary": decoded_output["summary"],
            }
        )

    return results


def save_results(eval_scores, generated_summaries, model_name):
    """Save evaluation results and generated summaries to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "evaluation_scores": eval_scores,
        "generated_summaries": generated_summaries,
    }

    filename = f"model_evaluation_{model_name}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filename


def print_results(eval_scores, generated_summaries, model_name):
    """Print evaluation results and generated summaries in a readable format."""
    print("\n" + "=" * 80)
    print(f"MODEL EVALUATION RESULTS - {model_name.upper()}")
    print("=" * 80)

    print("\nROUGE SCORES:")
    print("-" * 40)
    for metric, score in eval_scores.items():
        print(f"{metric}: {score:.4f}")

    print("\nGENERATED SUMMARIES:")
    print("-" * 40)
    for i, result in enumerate(generated_summaries, 1):
        print(f"\nExample {i}:")
        print(f"Original Text: {result['original_text']}")
        print(f"Original Summary: {result['original_summary']}")
        print(f"Generated Summary: {result['generated_summary']}")


def compare_models(models, dataset, num_examples=1, temperature=0.7):
    """
    Compare multiple models on the same example data.

    Args:
        models (list): List of tuples containing (model, model_name)
        dataset: XLSumDataset instance
        num_examples (int): Number of examples to test
        temperature (float): Sampling temperature for generation

    Returns:
        list: List of results for each example, containing original text and summaries from each model
    """
    results = []

    # Get examples from test set
    test_examples = dataset.test_data[:num_examples]

    for example in test_examples:
        # Get input sequence using the test split
        input_sequence, _ = dataset.get_batch(split="test")

        # Store results for this example
        example_result = {
            "original_text": example["text"],
            "original_summary": example["summary"],
            "model_summaries": {},
        }

        # Generate summaries with each model
        for model, model_name in models:
            generated_tokens = model.generate(
                input_article_tokens=input_sequence[0],
                word2idx=dataset.tokenizer.word2idx,
                max_length=dataset.max_seq_length,
                temperature=temperature,
            )

            # Decode generated summary
            decoded_output = dataset.decode_batch(generated_tokens)[0]
            example_result["model_summaries"][model_name] = decoded_output["summary"]

        results.append(example_result)

    return results


def print_comparison_results(results):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nExample {i}:")
        print("-" * 40)
        print(f"Original Text:\n{result['original_text']}\n")
        print(
            f"Original Summary:\n{result['original_summary'].lower()}\n"
        )  # Convert to lowercase for display
        print("Generated Summaries:")
        for model_name, summary in result["model_summaries"].items():
            print(f"\n{model_name}:")
            print(f"Summary: {summary.lower()}")  # Convert to lowercase for display
            # Get the model's dataset to access its tokenizer
            model_dataset = next(d for m, n, d in models if n == model_name)
            # Show tokens with UNK markers
            tokens = model_dataset.tokenizer.encode(
                summary.lower(), add_special_tokens=False
            )  # Convert to lowercase for tokenization
            token_words = [
                model_dataset.tokenizer.idx2word.get(int(t), "<UNK>") for t in tokens
            ]
            print(f"Tokens: {' '.join(token_words)}")
            # Count UNK tokens
            unk_count = sum(1 for t in token_words if t == "<UNK>")
            print(f"Number of UNK tokens: {unk_count}")
            print(f"Vocabulary size: {len(model_dataset.tokenizer.word2idx)}")


def list_available_examples(dataset, num_examples=5):
    """Print a list of available examples from the test set."""
    print("\nAvailable examples from test set:")
    print("-" * 40)
    for i, example in enumerate(dataset.test_data[:num_examples]):
        print(f"\nExample {i}:")
        print(f"Text: {example['text'][:200]}...")  # Show first 200 chars
        print(f"Summary: {example['summary']}")


def find_best_and_worst_samples(model, dataset, num_samples=20):
    """
    Find the samples with the best and worst generation quality from the test set.
    Considers both ROUGE scores and number of UNK tokens.

    Args:
        model: The model to evaluate
        dataset: XLSumDataset instance
        num_samples (int): Number of samples to evaluate before selecting

    Returns:
        tuple: (best_sample_index, best_sample, best_score, worst_sample_index, worst_sample, worst_score)
    """
    print("\nEvaluating samples to find best and worst generations...")
    best_score = float("-inf")
    worst_score = float("inf")
    best_sample = None
    worst_sample = None
    best_index = None
    worst_index = None

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Store all scores for debugging
    all_scores = []

    # Evaluate first num_samples from test set
    for i in tqdm(
        range(min(num_samples, len(dataset.test_data))), desc="Evaluating samples"
    ):
        # Get the example
        example = dataset.test_data[i]

        # Tokenize input
        article_tokens = dataset.tokenizer.encode(
            example["text"], add_special_tokens=False
        )
        input_seq = np.concatenate(
            [
                np.array([dataset.tokenizer.word2idx[dataset.bos_token]]),  # <BOS>
                article_tokens,
                np.array([dataset.tokenizer.word2idx[dataset.sep_token]]),  # <SEP>
            ]
        )

        # Pad sequence
        if len(input_seq) < dataset.max_seq_length:
            input_padding = np.full(
                dataset.max_seq_length - len(input_seq),
                dataset.tokenizer.word2idx[dataset.pad_token],
            )
            input_seq = np.concatenate([input_seq, input_padding])
        else:
            input_seq = input_seq[: dataset.max_seq_length]

        # Generate summary
        generated_tokens = model.generate(
            input_article_tokens=input_seq,
            word2idx=dataset.tokenizer.word2idx,
            max_length=dataset.max_seq_length,
            temperature=0.7,
        )

        # Decode generated summary
        decoded_output = dataset.decode_batch(generated_tokens)[0]
        generated_summary = decoded_output["summary"].lower()

        # Count UNK tokens in generated summary
        summary_tokens = dataset.tokenizer.encode(
            generated_summary, add_special_tokens=False
        )
        token_words = [
            dataset.tokenizer.idx2word.get(int(t), "<UNK>") for t in summary_tokens
        ]
        unk_count = sum(1 for t in token_words if t == "<UNK>")
        unk_ratio = unk_count / len(token_words) if token_words else 1.0

        # Calculate ROUGE scores
        scores = scorer.score(example["summary"].lower(), generated_summary)
        rouge_score = (
            scores["rouge1"].fmeasure
            + scores["rouge2"].fmeasure
            + scores["rougeL"].fmeasure
        ) / 3

        # Combine ROUGE score and UNK ratio into a single quality score
        # Lower UNK ratio is better, so we subtract it from 1
        # We weight ROUGE score more heavily (0.7) than UNK ratio (0.3)
        quality_score = (0.7 * rouge_score) + (0.3 * (1 - unk_ratio))

        # Store score for debugging
        all_scores.append(
            (i, quality_score, rouge_score, unk_ratio, example, generated_summary)
        )

        # Update best and worst samples
        if quality_score > best_score:
            best_score = quality_score
            best_sample = example
            best_index = i
        if quality_score < worst_score:
            worst_score = quality_score
            worst_sample = example
            worst_index = i

    # Print score distribution
    all_scores.sort(key=lambda x: x[1])  # Sort by quality score
    print("\nScore distribution:")
    print(f"Best quality score: {best_score:.4f} (sample {best_index})")
    print(f"Worst quality score: {worst_score:.4f} (sample {worst_index})")
    print(f"Median quality score: {all_scores[len(all_scores)//2][1]:.4f}")
    print(f"Average quality score: {sum(s[1] for s in all_scores)/len(all_scores):.4f}")

    # Print details of best and worst samples
    print("\nBest sample details:")
    best_details = next(s for s in all_scores if s[0] == best_index)
    print(f"ROUGE score: {best_details[2]:.4f}")
    print(f"UNK ratio: {best_details[3]:.4f}")
    print(f"Generated summary: {best_details[5]}")

    print("\nWorst sample details:")
    worst_details = next(s for s in all_scores if s[0] == worst_index)
    print(f"ROUGE score: {worst_details[2]:.4f}")
    print(f"UNK ratio: {worst_details[3]:.4f}")
    print(f"Generated summary: {worst_details[5]}")

    return best_index, best_sample, best_score, worst_index, worst_sample, worst_score


def main(sample_indices=None):
    """
    Run model comparison.

    Args:
        sample_indices (list, optional): List of indices of examples to use from test set.
            If None, will find the best and worst performing samples.
    """
    # Model configurations with exact parameters from config files
    model_configs = [
        {
            "name": "model_128d",
            "d_model": 128,  # Embedding dimension
            "num_heads": 2,  # Number of attention heads
            "num_layers": 2,  # Number of layers
            "d_ff": 75,  # Feed-forward dimension
            "max_seq_length": 64,
            "vocab_size": 3200,
            "weights_file": "best_model_weights_model_128d.npy",
        },
        {
            "name": "model_8k_vocab",
            "d_model": 56,  # Embedding dimension
            "num_heads": 4,  # Number of attention heads
            "num_layers": 2,  # Number of layers
            "d_ff": 96,  # Feed-forward dimension
            "max_seq_length": 64,
            "vocab_size": 8000,
            "weights_file": "best_model_weights_model_8k_vocab.npy",
        },
        {
            "name": "model_1layer",
            "d_model": 88,  # Embedding dimension
            "num_heads": 4,  # Number of attention heads
            "num_layers": 1,  # Number of layers
            "d_ff": 352,  # Feed-forward dimension
            "max_seq_length": 64,
            "vocab_size": 5000,
            "weights_file": "best_model_weights_model_1layer.npy",
        },
    ]

    # Initialize dataset (using the first model's config)
    dataset = XLSumDataset(
        max_seq_length=model_configs[0]["max_seq_length"],
        batch_size=32,
        vocab_size=model_configs[0]["vocab_size"],
        model_name=model_configs[0]["name"],
    )

    # Load models
    print("Loading models...")
    global models  # Make models accessible to print_comparison_results
    models = []
    for config in model_configs:
        print(f"\nLoading {config['name']}...")
        print(f"Config:")
        print(f"  - Embedding dimension (d_model): {config['d_model']}")
        print(f"  - Number of attention heads: {config['num_heads']}")
        print(f"  - Number of layers: {config['num_layers']}")
        print(f"  - Feed-forward dimension: {config['d_ff']}")
        print(f"  - Vocabulary size: {config['vocab_size']}")
        print(f"  - Maximum sequence length: {config['max_seq_length']}")

        # Create a new dataset instance for each model since they have different vocab sizes
        model_dataset = XLSumDataset(
            max_seq_length=config["max_seq_length"],
            batch_size=32,
            vocab_size=config["vocab_size"],
            model_name=config["name"],
        )

        model = load_model(config["weights_file"], model_dataset, config)
        if model is None:
            print(f"Failed to load model {config['name']}. Skipping...")
            continue
        models.append((model, config["name"], model_dataset))
        print(f"Successfully loaded {config['name']}")

    if not models:
        print("No models were loaded successfully. Exiting...")
        return

    # Handle sample selection
    if sample_indices is None:
        # Use the first model to find the best and worst samples
        model, model_name, model_dataset = models[0]
        best_index, best_sample, best_score, worst_index, worst_sample, worst_score = (
            find_best_and_worst_samples(model, model_dataset, num_samples=20)
        )
        sample_indices = [best_index, worst_index]
        samples = [best_sample, worst_sample]
        scores = [best_score, worst_score]

        print("\nSelected samples:")
        for i, (idx, sample, score) in enumerate(zip(sample_indices, samples, scores)):
            print(
                f"\nSample {i+1} ({'Best' if i == 0 else 'Worst'} - index {idx}, score: {score:.4f}):"
            )
            print(f"Text: {sample['text'][:200]}...")
            print(f"Summary: {sample['summary'].lower()}")
    else:
        # Get the selected examples from test set
        samples = []
        for idx in sample_indices:
            try:
                example = dataset.test_data[idx]
                samples.append(example)
                print(f"\nUsing example {idx}:")
                print(f"Text: {example['text'][:200]}...")
                print(f"Summary: {example['summary'].lower()}")
            except IndexError:
                print(f"Index {idx} out of range. Skipping...")
                continue

    results = []
    for sample in samples:
        example_result = {
            "original_text": sample["text"],
            "original_summary": sample["summary"],
            "model_summaries": {},
            "sample_index": dataset.test_data.index(sample),
        }

        # Generate summaries with each model
        for model, model_name, model_dataset in models:
            print(f"\nGenerating summary with {model_name}...")
            # Tokenize input for this specific model
            article_tokens = model_dataset.tokenizer.encode(
                sample["text"], add_special_tokens=False
            )
            input_seq = np.concatenate(
                [
                    np.array(
                        [model_dataset.tokenizer.word2idx[model_dataset.bos_token]]
                    ),  # <BOS>
                    article_tokens,
                    np.array(
                        [model_dataset.tokenizer.word2idx[model_dataset.sep_token]]
                    ),  # <SEP>
                ]
            )

            # Pad sequence
            if len(input_seq) < model_dataset.max_seq_length:
                input_padding = np.full(
                    model_dataset.max_seq_length - len(input_seq),
                    model_dataset.tokenizer.word2idx[model_dataset.pad_token],
                )
                input_seq = np.concatenate([input_seq, input_padding])
            else:
                input_seq = input_seq[: model_dataset.max_seq_length]

            # Generate summary
            generated_tokens = model.generate(
                input_article_tokens=input_seq,
                word2idx=model_dataset.tokenizer.word2idx,
                max_length=model_dataset.max_seq_length,
                temperature=0.7,
            )

            # Decode generated summary
            decoded_output = model_dataset.decode_batch(generated_tokens)[0]
            example_result["model_summaries"][model_name] = decoded_output["summary"]

        results.append(example_result)

    # Print results
    print_comparison_results(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    import sys

    # Allow sample indices to be passed as command line arguments
    sample_indices = [int(idx) for idx in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(sample_indices)
