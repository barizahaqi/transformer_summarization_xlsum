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
        split (str): Dataset split to evaluate on (only test is used)

    Returns:
        dict: ROUGE scores for the test set
    """
    num_batches = len(getattr(dataset, f"{split}_data")) // dataset.batch_size

    # Lists to store all references and predictions for ROUGE calculation
    all_references = []
    all_predictions = []

    for _ in tqdm(range(num_batches), desc=f"Evaluating on {split}"):
        # Get batch
        text_batch, summary_batch = dataset.get_batch(split)

        # Generate summaries for ROUGE evaluation
        for i in range(len(text_batch)):
            # Get reference summary
            reference = dataset.decode_batch([summary_batch[i]])[0]

            # Generate prediction
            generated_tokens = model.generate(
                start_token=dataset.tokenizer.word2idx[dataset.tokenizer.bos_token],
                max_length=dataset.max_seq_length,
                temperature=0.7,
            )
            prediction = dataset.decode_batch([generated_tokens[0]])[0]

            all_references.append(reference)
            all_predictions.append(prediction)

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(all_references, all_predictions)

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
    # Model configurations - exactly matching train.py
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
            )

            # Load model
            model = load_model(config["weights_file"], dataset, config)

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
