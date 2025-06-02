import numpy as np
from datasets import load_dataset
from .tokenizer import SimpleTokenizer
import os


class XLSumDataset:
    def __init__(
        self,
        max_seq_length=64,
        batch_size=32,
        vocab_size=5000,
        data_dir="data/xlsum",
        model_name=None,  # Add model_name parameter
        max_samples=None,  # Add max_samples parameter
    ):
        """
        Initialize XLSum dataset.

        Args:
            max_seq_length (int): Maximum sequence length for both article and summary
            batch_size (int): Batch size for training
            vocab_size (int): Size of vocabulary to use
            data_dir (str): Directory containing dataset files
            model_name (str): Name of the model (e.g., 'model_128d', 'model_8k_vocab', 'model_1layer')
            max_samples (int, optional): Maximum number of samples to use from each split. If None, use all data.
        """
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.data_dir = data_dir
        self.model_name = model_name
        self.max_samples = max_samples

        # Set tokenizer path based on model name and vocab size
        if model_name:
            self.tokenizer_path = f"tokenizer_{model_name}_vocab{vocab_size}.json"
        else:
            self.tokenizer_path = f"tokenizer_vocab{vocab_size}.json"

        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        # Load dataset
        self.dataset = load_dataset("csebuetnlp/xlsum", "indonesian", cache_dir="cache")

        # Build or load vocabulary
        if os.path.exists(self.tokenizer_path):
            print(f"Loading existing tokenizer from {self.tokenizer_path}")
            self.tokenizer.load(self.tokenizer_path)
        else:
            print(
                f"Building vocabulary for {model_name or 'default'} (vocab_size={vocab_size})..."
            )
            self._build_vocab()
            self.tokenizer.save(self.tokenizer_path)
            print(f"Vocabulary building complete. Saved to {self.tokenizer_path}")

        # Load and prepare data
        print(
            f"Loading dataset splits (max_samples={max_samples if max_samples else 'all'})..."
        )
        self.train_data = self._load_data("train")
        self.val_data = self._load_data("validation")
        self.test_data = self._load_data("test")
        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.val_data)} validation samples")
        print(f"Loaded {len(self.test_data)} test samples")

        # Special tokens
        self.bos_token = "<BOS>"
        self.sep_token = "<SEP>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

    def _build_vocab(self):
        """Build vocabulary from training data."""
        texts = []
        for item in self.dataset["train"]:
            texts.append(item["text"])
            texts.append(item["summary"])

        self.tokenizer.build_vocab(texts)
        print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")

    def _load_data(self, split):
        """
        Load and prepare data for a specific split.

        Args:
            split (str): Dataset split ("train", "validation", or "test")

        Returns:
            list: List of dictionaries containing text and summary
        """
        data = []
        for item in self.dataset[split]:
            data.append({"text": item["text"], "summary": item["summary"]})
            if self.max_samples and len(data) >= self.max_samples:
                break
        return data

    def _prepare_sequence(self, text, summary):
        """
        Prepare a sequence in the format <BOS> article <SEP> summarization <EOS>.

        Args:
            text (str): Input article text
            summary (str): Target summary text

        Returns:
            tuple: (tokenized_sequence, target_sequence)
        """
        # Tokenize article and summary
        article_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        summary_tokens = self.tokenizer.encode(summary, add_special_tokens=False)

        # Calculate max lengths (leaving room for special tokens)
        max_article_len = (
            self.max_seq_length - 4
        ) // 2  # -4 for <BOS>, <SEP>, <EOS>, and one extra token
        max_summary_len = (self.max_seq_length - 4) // 2

        # Truncate if needed
        article_tokens = article_tokens[:max_article_len]
        summary_tokens = summary_tokens[:max_summary_len]

        # Create sequence: <BOS> article <SEP> summary <EOS>
        sequence = np.concatenate(
            [
                np.array([self.tokenizer.word2idx[self.tokenizer.bos_token]]),  # <BOS>
                article_tokens,
                np.array([self.tokenizer.word2idx[self.tokenizer.sep_token]]),  # <SEP>
                summary_tokens,
                np.array([self.tokenizer.word2idx[self.tokenizer.eos_token]]),  # <EOS>
            ]
        )

        # Create target sequence (same as input for training)
        target_sequence = sequence.copy()

        # Pad sequences
        if len(sequence) < self.max_seq_length:
            padding = np.full(
                self.max_seq_length - len(sequence),
                self.tokenizer.word2idx[self.tokenizer.pad_token],
            )
            sequence = np.concatenate([sequence, padding])
            target_sequence = np.concatenate([target_sequence, padding])
        else:
            sequence = sequence[: self.max_seq_length]
            target_sequence = target_sequence[: self.max_seq_length]

        return sequence, target_sequence

    def get_batch(self, split="train"):
        """
        Get a batch of data.

        Args:
            split (str): Which split to use ('train', 'val', or 'test')

        Returns:
            tuple: (input_sequences, target_sequences) where:
                - input_sequences: numpy array of shape (batch_size, max_seq_length) containing <BOS> article <SEP>
                - target_sequences: numpy array of shape (batch_size, max_seq_length) containing <BOS> article <SEP> summary <EOS>
        """
        data = getattr(self, f"{split}_data")
        indices = np.random.choice(len(data), self.batch_size, replace=False)

        input_sequences = []
        target_sequences = []

        for idx in indices:
            example = data[idx]
            # Tokenize article and summary
            article_tokens = self.tokenizer.encode(
                example["text"], add_special_tokens=False
            )
            summary_tokens = self.tokenizer.encode(
                example["summary"], add_special_tokens=False
            )

            # Calculate max lengths
            max_article_len = (
                self.max_seq_length - 3
            ) // 2  # -3 for <BOS>, <SEP>, and one extra token
            max_summary_len = (self.max_seq_length - 3) // 2

            # Truncate if needed
            article_tokens = article_tokens[:max_article_len]
            summary_tokens = summary_tokens[:max_summary_len]

            # Create input sequence: <BOS> article <SEP>
            input_seq = np.concatenate(
                [
                    np.array([self.tokenizer.word2idx[self.bos_token]]),  # <BOS>
                    article_tokens,
                    np.array([self.tokenizer.word2idx[self.sep_token]]),  # <SEP>
                ]
            )

            # Create target sequence: <BOS> article <SEP> summary <EOS>
            target_seq = np.concatenate(
                [
                    input_seq,  # <BOS> article <SEP>
                    summary_tokens,
                    np.array([self.tokenizer.word2idx[self.eos_token]]),  # <EOS>
                ]
            )

            # Pad sequences
            if len(input_seq) < self.max_seq_length:
                input_padding = np.full(
                    self.max_seq_length - len(input_seq),
                    self.tokenizer.word2idx[self.pad_token],
                )
                input_seq = np.concatenate([input_seq, input_padding])
            else:
                input_seq = input_seq[: self.max_seq_length]

            if len(target_seq) < self.max_seq_length:
                target_padding = np.full(
                    self.max_seq_length - len(target_seq),
                    self.tokenizer.word2idx[self.pad_token],
                )
                target_seq = np.concatenate([target_seq, target_padding])
            else:
                target_seq = target_seq[: self.max_seq_length]

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

        return np.array(input_sequences), np.array(target_sequences)

    def decode_batch(self, sequences):
        """
        Decode a batch of token sequences back to text.

        Args:
            sequences: List of token sequences or single sequence

        Returns:
            list: List of dictionaries containing decoded article and summary
        """
        # Ensure sequences is a list/array
        if not isinstance(sequences, (list, np.ndarray)):
            sequences = [sequences]
        elif isinstance(sequences, np.ndarray) and sequences.ndim == 1:
            sequences = [sequences]

        decoded = []
        for seq in sequences:
            # Ensure seq is a numpy array
            seq = np.asarray(seq)

            # Find the <SEP> token to separate article and summary
            try:
                sep_idx = np.where(seq == self.tokenizer.word2idx[self.sep_token])[0][0]
                # Get article part (between <BOS> and <SEP>)
                article_start = 1  # Skip <BOS>
                article_end = sep_idx
                article_tokens = seq[article_start:article_end]

                # Get summary part (between <SEP> and <EOS>)
                summary_start = sep_idx + 1
                try:
                    eos_idx = np.where(
                        seq[summary_start:] == self.tokenizer.word2idx[self.eos_token]
                    )[0][0]
                    summary_end = summary_start + eos_idx
                except IndexError:
                    summary_end = len(seq)  # No <EOS> found, use until end
                summary_tokens = seq[summary_start:summary_end]

                # Decode article
                article_words = [
                    self.tokenizer.idx2word.get(int(token), self.unk_token)
                    for token in article_tokens
                ]
                article_words = [
                    w
                    for w in article_words
                    if w
                    not in [
                        self.bos_token,
                        self.sep_token,
                        self.eos_token,
                        self.pad_token,
                    ]
                ]
                article = " ".join(article_words)

                # Decode summary
                summary_words = [
                    self.tokenizer.idx2word.get(int(token), self.unk_token)
                    for token in summary_tokens
                ]
                summary_words = [
                    w
                    for w in summary_words
                    if w
                    not in [
                        self.bos_token,
                        self.sep_token,
                        self.eos_token,
                        self.pad_token,
                    ]
                ]
                summary = " ".join(summary_words)

                decoded.append({"article": article, "summary": summary})

            except IndexError:
                # If no <SEP> found, treat the whole sequence as summary
                words = [
                    self.tokenizer.idx2word.get(int(token), self.unk_token)
                    for token in seq
                ]
                words = [
                    w
                    for w in words
                    if w
                    not in [
                        self.bos_token,
                        self.sep_token,
                        self.eos_token,
                        self.pad_token,
                    ]
                ]
                decoded.append({"article": "", "summary": " ".join(words)})

        return decoded
