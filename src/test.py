import numpy as np
from tqdm import tqdm
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset
import json
from datetime import datetime
from rouge_score import rouge_scorer
import re


def load_model(model_path, dataset, config):
    """
    Load a trained model from saved weights.

    Args:
        model_path (str): Path to saved model weights
        dataset: XLSumDataset instance
        config (dict): Model configuration parameters

    Returns:
        TransformerDecoder: Loaded model
    """
    # Initialize model with configuration parameters
    model = TransformerDecoder(
        vocab_size=len(dataset.tokenizer.word2idx),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=config["max_seq_length"],
    )

    # Load weights
    try:
        weights = np.load(model_path, allow_pickle=True).item()
        print(f"Loading model weights for {config['name']}...")

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

        print(f"Successfully loaded model weights for {config['name']}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No model weights found at {model_path}")


def calculate_rouge_scores(references, predictions):
    """
    Calculate ROUGE scores for a batch of summaries.

    Args:
        references (list): List of reference summaries
        predictions (list): List of generated summaries

    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Clean and normalize texts
    def clean_text(text):
        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text).strip()
        return text

    references = [clean_text(ref) for ref in references]
    predictions = [clean_text(pred) for pred in predictions]

    # Calculate scores for each pair
    scores = {
        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
        "rouge2": {"precision": [], "recall": [], "fmeasure": []},
        "rougeL": {"precision": [], "recall": [], "fmeasure": []},
    }

    for ref, pred in zip(references, predictions):
        if not pred.strip():  # Skip empty predictions
            continue

        score = scorer.score(ref, pred)
        for metric in ["rouge1", "rouge2", "rougeL"]:
            scores[metric]["precision"].append(score[metric].precision)
            scores[metric]["recall"].append(score[metric].recall)
            scores[metric]["fmeasure"].append(score[metric].fmeasure)

    # Calculate averages
    avg_scores = {}
    for metric in scores:
        avg_scores[metric] = {
            "precision": np.mean(scores[metric]["precision"]),
            "recall": np.mean(scores[metric]["recall"]),
            "fmeasure": np.mean(scores[metric]["fmeasure"]),
        }

    return avg_scores


def evaluate_model(model, dataset, split="test"):
    """
    Evaluate model on test set using ROUGE scores.

    Args:
        model: TransformerDecoder model
        dataset: XLSumDataset instance
        split (str): Dataset split to evaluate on ('val' or 'test')

    Returns:
        dict: ROUGE scores for the test set
    """
    # Map split name to dataset attribute
    split_map = {"val": "val_data", "test": "test_data"}
    data_attr = split_map[split]
    split_data = getattr(dataset, data_attr)

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Lists to store all references and predictions for ROUGE calculation
    all_references = []
    all_predictions = []
    all_unk_counts = []  # Track UNK tokens
    all_summary_lengths = []  # Track summary lengths

    # Evaluate each example in the split
    for example in tqdm(split_data, desc=f"Evaluating on {split}"):
        # Tokenize input text properly
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

        # Generate prediction
        generated_tokens = model.generate(
            input_article_tokens=input_seq,
            word2idx=dataset.tokenizer.word2idx,
            max_length=dataset.max_seq_length,
            temperature=0.7,
        )

        # Decode reference and generated summaries
        reference = example["summary"].lower()  # Convert to lowercase
        decoded = dataset.decode_batch([generated_tokens])
        prediction = decoded[0]["summary"].lower()  # Convert to lowercase

        # Count UNK tokens in prediction
        pred_tokens = dataset.tokenizer.encode(prediction, add_special_tokens=False)
        token_words = [
            dataset.tokenizer.idx2word.get(int(t), "<UNK>") for t in pred_tokens
        ]
        unk_count = sum(1 for t in token_words if t == "<UNK>")

        # Store metrics
        all_references.append(reference)
        all_predictions.append(prediction)
        all_unk_counts.append(unk_count)
        all_summary_lengths.append(len(token_words))

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(all_references, all_predictions)

    # Print debugging information
    print("\nEvaluation Statistics:")
    print(f"Number of samples evaluated: {len(all_predictions)}")
    print(f"Average summary length: {np.mean(all_summary_lengths):.1f} tokens")
    print(f"Average UNK tokens per summary: {np.mean(all_unk_counts):.1f}")
    print(
        f"UNK token ratio: {np.mean(all_unk_counts) / np.mean(all_summary_lengths):.4f}"
    )

    # Print some example predictions
    print("\nExample Predictions (first 3):")
    for i in range(min(3, len(all_predictions))):
        print(f"\nExample {i+1}:")
        print(f"Reference: {all_references[i]}")
        print(f"Prediction: {all_predictions[i]}")
        print(f"UNK tokens: {all_unk_counts[i]}")
        print(f"Summary length: {all_summary_lengths[i]}")

    return rouge_scores


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
    results = []

    # Get examples from test set
    test_examples = dataset.test_data[:num_examples]

    for example in test_examples:
        # Tokenize input text properly
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
            temperature=temperature,
        )

        # Decode generated summary
        decoded = dataset.decode_batch([generated_tokens])
        generated_summary = decoded[0]["summary"]

        # Count UNK tokens in generated summary
        pred_tokens = dataset.tokenizer.encode(
            generated_summary, add_special_tokens=False
        )
        token_words = [
            dataset.tokenizer.idx2word.get(int(t), "<UNK>") for t in pred_tokens
        ]
        unk_count = sum(1 for t in token_words if t == "<UNK>")

        results.append(
            {
                "original_text": example["text"],
                "original_summary": example["summary"].lower(),
                "generated_summary": generated_summary,
                "unk_count": unk_count,
                "summary_length": len(token_words),
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
        print(f"UNK tokens: {result['unk_count']}")
        print(f"Summary length: {result['summary_length']}")
        print("-" * 80)


def save_results_to_file(results_dict, filename=None):
    """
    Save evaluation results to a text file.

    Args:
        results_dict (dict): Dictionary containing evaluation results for all models
        filename (str, optional): Name of the output file. If None, generates timestamp-based name.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_evaluation_results_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Transformer Model Evaluation Results\n")
        f.write("=" * 80 + "\n\n")

        for model_name, model_results in results_dict.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 40 + "\n")

            # Write model configuration
            f.write("Configuration:\n")
            for key, value in model_results["config"].items():
                if key not in [
                    "weights_file",
                    "num_epochs",
                    "learning_rate",
                ]:  # Skip training-specific configs
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Write ROUGE scores
            f.write("ROUGE Scores:\n")
            for metric in ["rouge1", "rouge2", "rougeL"]:
                scores = model_results["rouge_scores"][metric]
                f.write(f"  {metric.upper()}:\n")
                f.write(f"    Precision: {scores['precision']:.4f}\n")
                f.write(f"    Recall: {scores['recall']:.4f}\n")
                f.write(f"    F1-Score: {scores['fmeasure']:.4f}\n")
            f.write("\n")

            # Write example generations
            f.write("Example Generations:\n")
            for i, example in enumerate(model_results["examples"], 1):
                f.write(f"\nExample {i}:\n")
                f.write(f"Original Text: {example['original_text'][:200]}...\n")
                f.write(f"Original Summary: {example['original_summary']}\n")
                f.write(f"Generated Summary: {example['generated_summary']}\n")
                f.write("-" * 40 + "\n")

            f.write("\n" + "=" * 80 + "\n\n")

        print(f"Results saved to {filename}")


def main():
    model_configs = {
        "model_128d": {
            "name": "model_128d",
            "d_model": 128,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 75,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 3200,
            "weights_file": "best_model_weights_model_128d.npy",
        },
        "model_8k_vocab": {
            "name": "model_8k_vocab",
            "d_model": 56,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 96,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 8000,
            "weights_file": "best_model_weights_model_8k_vocab.npy",
        },
        "model_1layer": {
            "name": "model_1layer",
            "d_model": 88,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 352,
            "max_seq_length": 64,
            "batch_size": 32,
            "vocab_size": 5000,
            "weights_file": "best_model_weights_model_1layer.npy",
        },
    }

    # Dictionary to store results for all models
    all_results = {}

    # Evaluate each model
    for model_name, config in model_configs.items():
        print(f"\nEvaluating {model_name}...")
        try:
            # Initialize dataset with model-specific vocabulary size
            print(f"Loading dataset with vocab_size={config['vocab_size']}...")
            dataset = XLSumDataset(
                max_seq_length=config["max_seq_length"],
                batch_size=config["batch_size"],
                vocab_size=config["vocab_size"],
                model_name=config["name"],
                max_samples=None,  # Use full dataset
            )

            # Load model
            model = load_model(config["weights_file"], dataset, config)

            # Print dataset size
            print(f"Test set size: {len(dataset.test_data)} samples")

            # Evaluate on test set
            rouge_scores = evaluate_model(model, dataset, "test")

            # Generate example summaries
            examples = generate_summaries(
                model, dataset, num_examples=3, temperature=0.7
            )

            # Calculate total parameters
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

            # Store results
            all_results[model_name] = {
                "config": config,
                "rouge_scores": rouge_scores,
                "examples": examples,
                "total_params": total_params,
                "test_set_size": len(dataset.test_data),
            }

            # Print results for this model
            print(f"\nResults for {model_name}:")
            print(f"Configuration:")
            print(f"  Vocabulary size: {config['vocab_size']}")
            print(f"  Embedding dimension: {config['d_model']}")
            print(f"  Number of attention heads: {config['num_heads']}")
            print(f"  Number of layers: {config['num_layers']}")
            print(f"  Feed-forward dimension: {config['d_ff']}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Test set size: {len(dataset.test_data)} samples")
            print("\nROUGE Scores:")
            for metric in ["rouge1", "rouge2", "rougeL"]:
                scores = rouge_scores[metric]
                print(f"  {metric.upper()}:")
                print(f"    Precision: {scores['precision']:.4f}")
                print(f"    Recall: {scores['recall']:.4f}")
                print(f"    F1-Score: {scores['fmeasure']:.4f}")
            print("\nExample generations:")
            print_results(examples)

        except FileNotFoundError as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Save all results to file
    if all_results:
        save_results_to_file(all_results)
    else:
        print("No results to save - all models failed to load.")


if __name__ == "__main__":
    main()
