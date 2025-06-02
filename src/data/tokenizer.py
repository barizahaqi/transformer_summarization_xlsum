import re
import numpy as np
from collections import Counter


class SimpleTokenizer:
    def __init__(self, vocab_size=5000, min_freq=2):
        """
        Initialize simple tokenizer.

        Args:
            vocab_size (int): Maximum vocabulary size
            min_freq (int): Minimum frequency for a token to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.sep_token = "<SEP>"

        # Add special tokens to vocabulary
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
            self.sep_token: 4,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def preprocess_text(self, text):
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing special characters
        3. Splitting into words
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and extra whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split into words
        words = text.split()

        return words

    def build_vocab(self, texts):
        """
        Build vocabulary from texts.

        Args:
            texts: List of text strings
        """
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            words = self.preprocess_text(text)
            word_freq.update(words)

        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))

        # Add most frequent words to vocabulary
        for word, freq in sorted_words:
            if len(self.word2idx) >= self.vocab_size:
                break
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token indices.

        Args:
            text (str): Input text
            add_special_tokens (bool): Whether to add BOS/EOS tokens

        Returns:
            numpy.ndarray: Array of token indices
        """
        words = self.preprocess_text(text)

        # Convert words to indices
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx[self.bos_token])

        for word in words:
            idx = self.word2idx.get(word, self.word2idx[self.unk_token])
            indices.append(idx)

        if add_special_tokens:
            indices.append(self.word2idx[self.eos_token])

        return np.array(indices)

    def decode(self, indices):
        """
        Decode token indices to text.

        Args:
            indices: Array of token indices

        Returns:
            str: Decoded text
        """
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in [
                    self.pad_token,
                    self.bos_token,
                    self.eos_token,
                    self.sep_token,
                ]:
                    words.append(word)

        return " ".join(words)

    def save(self, path):
        """Save tokenizer to file."""
        import json

        data = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Load tokenizer from file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.min_freq = data["min_freq"]
        self.word2idx = data["word2idx"]
        # Create idx2word by swapping keys and values
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def pad_sequence(self, sequence, max_length):
        """
        Pad sequence to max_length.

        Args:
            sequence: Array of token indices
            max_length: Maximum sequence length
        """
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            padding = [self.word2idx[self.pad_token]] * (max_length - len(sequence))
            sequence = np.concatenate([sequence, padding])

        return sequence
