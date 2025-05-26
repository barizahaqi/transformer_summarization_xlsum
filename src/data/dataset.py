import numpy as np
from datasets import load_dataset
from .tokenizer import SimpleTokenizer


class XLSumDataset:
    def __init__(self, max_seq_length=512, batch_size=32, vocab_size=32000):
        """
        Initialize XLSum dataset.

        Args:
            max_seq_length (int): Maximum sequence length
            batch_size (int): Batch size for training
            vocab_size (int): Vocabulary size for tokenizer
        """
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)

        # Load dataset
        self.dataset = load_dataset("csebuetnlp/xlsum", "indonesian")

        # Build vocabulary
        self._build_vocab()

        # Prepare data
        self.train_data = self._prepare_data("train")
        self.val_data = self._prepare_data("validation")
        self.test_data = self._prepare_data("test")

    def _build_vocab(self):
        """Build vocabulary from training data."""
        texts = []
        for item in self.dataset["train"]:
            texts.append(item["text"])
            texts.append(item["summary"])

        self.tokenizer.build_vocab(texts)
        print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")

    def _prepare_data(self, split):
        """
        Prepare data for a specific split.

        Args:
            split (str): Dataset split ("train", "validation", or "test")
        """
        data = []
        for item in self.dataset[split]:
            # Tokenize text and summary
            text_tokens = self.tokenizer.encode(item["text"])
            summary_tokens = self.tokenizer.encode(item["summary"])

            # Pad sequences
            text_tokens = self.tokenizer.pad_sequence(text_tokens, self.max_seq_length)
            summary_tokens = self.tokenizer.pad_sequence(
                summary_tokens, self.max_seq_length
            )

            data.append({"text": text_tokens, "summary": summary_tokens})

        return data

    def get_batch(self, split="train"):
        """
        Get a batch of data.

        Args:
            split (str): Dataset split ("train", "validation", or "test")
        """
        data = getattr(self, f"{split}_data")

        # Randomly sample batch_size examples
        indices = np.random.choice(len(data), self.batch_size, replace=False)
        batch = [data[i] for i in indices]

        # Stack text and summary tensors
        text_batch = np.stack([item["text"] for item in batch])
        summary_batch = np.stack([item["summary"] for item in batch])

        return text_batch, summary_batch

    def decode_batch(self, indices_batch):
        """
        Decode a batch of token indices to text.

        Args:
            indices_batch: Batch of token indices
        """
        texts = []
        for indices in indices_batch:
            text = self.tokenizer.decode(indices)
            texts.append(text)
        return texts
