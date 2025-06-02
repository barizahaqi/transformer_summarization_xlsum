import numpy as np
from transformer.decoder import TransformerDecoder
from data.dataset import XLSumDataset
from rouge_score import rouge_scorer
import json
from datetime import datetime
from tqdm import tqdm


def load_all_models(dataset):
    """Load all three trained models."""
    models = {}
    model_configs = {
        "model_128d": {
            "d_model": 128,
            "num_heads": 2,
            "num_layers": 2,
            "d_ff": 75,
            "max_seq_length": 64,
            "vocab_size": 3200,  # Correct vocab size from train.py
        },
        "model_8k_vocab": {
            "d_model": 56,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 96,
            "max_seq_length": 64,
            "vocab_size": 8000,  # Correct vocab size from train.py
        },
        "model_1layer": {
            "d_model": 88,
            "num_heads": 4,
            "num_layers": 1,
            "d_ff": 352,
            "max_seq_length": 64,
            "vocab_size": 5000,  # Correct vocab size from train.py
        },
    }

    # Create separate datasets for each model with their specific vocab sizes
    model_datasets = {}
    for model_name, config in model_configs.items():
        model_datasets[model_name] = XLSumDataset(
            max_seq_length=config["max_seq_length"],
            batch_size=32,
            vocab_size=config["vocab_size"],  # Using correct vocab size for each model
        )

    for model_name, config in model_configs.items():
        try:
            # Initialize model with correct configuration
            model = TransformerDecoder(
                vocab_size=config["vocab_size"],
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                d_ff=config["d_ff"],
                max_seq_length=config["max_seq_length"],
            )

            # Load weights
            weights = np.load(
                f"best_model_weights_{model_name}.npy", allow_pickle=True
            ).item()

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

            models[model_name] = {"model": model, "dataset": model_datasets[model_name]}
            print(f"Successfully loaded {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")

    return models


def get_test_example(models):
    """Get a single test example from the dataset for each model."""
    text = None
    original_summary = None
    tokenized_words = {}

    # Get example from first model's dataset
    first_model_name = list(models.keys())[0]
    first_dataset = models[first_model_name]["dataset"]

    # Get the raw example from dataset
    test_example = first_dataset.dataset["test"][30]

    # Get tokenized words for each model using their specific tokenizers
    for model_name, model_data in models.items():
        dataset = model_data["dataset"]
        # Encode text and take first 64 tokens (matching training max_seq_length)
        token_ids = dataset.tokenizer.encode(test_example["text"])[
            :64
        ]  # Take first 64 tokens
        # Pad if less than 64 tokens
        token_ids = dataset.tokenizer.pad_sequence(token_ids, 64)
        tokenized_words[model_name] = dataset.decode_batch([token_ids])[0]

    # Get original text and summary
    text = test_example["text"]
    original_summary = test_example["summary"]

    return text, original_summary, tokenized_words


def generate_summaries(models, input_tokens, temperature=0.7):
    """Generate summaries using all models."""
    summaries = {}
    tokenized_words = {}

    for model_name, model_data in models.items():
        try:
            model = model_data["model"]
            dataset = model_data["dataset"]

            # Generate summary with max length of 64 tokens (matching training max_seq_length)
            generated_tokens = model.generate(
                start_token=dataset.tokenizer.word2idx[dataset.tokenizer.bos_token],
                max_length=64,  # Set max length to 64 to match training
                temperature=temperature,
            )
            generated_summary = dataset.decode_batch([generated_tokens[0]])[0]
            summaries[model_name] = generated_summary

            # Get tokenized words
            token_ids = dataset.tokenizer.encode(generated_summary)
            # Pad if less than 64 tokens
            token_ids = dataset.tokenizer.pad_sequence(token_ids, 64)
            tokenized_words[model_name] = dataset.decode_batch([token_ids])[0]
        except Exception as e:
            print(f"Error generating summary with {model_name}: {str(e)}")
            summaries[model_name] = "Error generating summary"
            tokenized_words[model_name] = ""

    return summaries, tokenized_words


def calculate_rouge_scores(original_summary, generated_summaries):
    """Calculate ROUGE scores for all generated summaries."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {}

    for model_name, summary in generated_summaries.items():
        try:
            scores[model_name] = scorer.score(original_summary, summary)
        except Exception as e:
            print(f"Error calculating ROUGE scores for {model_name}: {str(e)}")
            scores[model_name] = None

    return scores


def save_results(
    text,
    original_summary,
    tokenized_input_words,
    generated_summaries,
    tokenized_output_words,
    rouge_scores,
):
    """Save test results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "input_text": text,
        "tokenized_input_words": tokenized_input_words,
        "original_summary": original_summary,
        "generated_summaries": generated_summaries,
        "tokenized_output_words": tokenized_output_words,
        "rouge_scores": {
            model: {
                metric: (
                    {
                        "precision": score.precision,
                        "recall": score.recall,
                        "fmeasure": score.fmeasure,
                    }
                    if score
                    else None
                )
                for metric, score in model_scores.items()
            }
            for model, model_scores in rouge_scores.items()
        },
    }

    filename = f"summarization_test_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filename


def print_results(
    text,
    original_summary,
    tokenized_input_words,
    generated_summaries,
    tokenized_output_words,
    rouge_scores,
):
    """Print test results in a readable format."""
    print("\n" + "=" * 80)
    print(
        "SUMMARIZATION TEST RESULTS (64 tokens input/output - matching training setup)"
    )
    print("=" * 80)

    print("\nINPUT TEXT (First 64 tokens):")
    print("-" * 40)
    for model_name, words in tokenized_input_words.items():
        print(f"\n{model_name.upper()}:")
        print(words)

    print("\nORIGINAL SUMMARY:")
    print("-" * 40)
    print(original_summary)

    print("\nGENERATED SUMMARIES (64 tokens max):")
    print("-" * 40)
    for model_name, summary in generated_summaries.items():
        print(f"\n{model_name.upper()}:")
        print(f"Generated Text: {summary}")
        print(f"Tokenized Words: {tokenized_output_words[model_name]}")

    print("\nROUGE SCORES:")
    print("-" * 40)
    for model_name, model_scores in rouge_scores.items():
        print(f"\n{model_name.upper()}:")
        if model_scores:
            for metric, score in model_scores.items():
                if score:
                    print(f"{metric}:")
                    print(f"  Precision: {score.precision:.4f}")
                    print(f"  Recall: {score.recall:.4f}")
                    print(f"  F1-Score: {score.fmeasure:.4f}")
        else:
            print("No scores available")


def main():
    # Load all models with their specific datasets
    print("Loading models...")
    models = load_all_models(None)  # We don't need the initial dataset anymore

    if not models:
        print("No models were successfully loaded. Exiting...")
        return

    # Get test example
    print("\nGetting test example...")
    text, original_summary, tokenized_input_words = get_test_example(models)

    # Generate summaries
    print("\nGenerating summaries...")
    generated_summaries, tokenized_output_words = generate_summaries(
        models, tokenized_input_words
    )

    # Calculate ROUGE scores
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge_scores(original_summary, generated_summaries)

    # Save results
    results_file = save_results(
        text,
        original_summary,
        tokenized_input_words,
        generated_summaries,
        tokenized_output_words,
        rouge_scores,
    )
    print(f"\nResults saved to {results_file}")

    # Print results
    print_results(
        text,
        original_summary,
        tokenized_input_words,
        generated_summaries,
        tokenized_output_words,
        rouge_scores,
    )


if __name__ == "__main__":
    main()
