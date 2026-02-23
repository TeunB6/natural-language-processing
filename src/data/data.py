from src.data.download import download_ag_news
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import polars as pl
from pathlib import Path
from src.const import DEBUG, LOGGER, RANDOM_SEED
from rich.panel import Panel

import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from src.utils.data import TorchDataset

from typing import Literal
from pathlib import Path


class AGNews:
    """Class to handle loading and vectorizing the AG News dataset."""

    def __init__(self, path: Path | str = "data/ag_news") -> None:
        """Initialize the class and load/vectorize the dataset.

        Args:
            path (Path | str, optional): The path to the AG News dataset
                                         directory. Defaults to "data/ag_news".
        """
        self.path = Path(path)

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")
            download_ag_news()
        self._load_data()
        self._vectorize()

    def _load_data(self) -> None:
        """Load the data from disk into memory."""
        self.train_df = pl.read_csv(self.path / "train.csv")
        self.dev_df = pl.read_csv(self.path / "dev.csv")
        self.test_df = pl.read_csv(self.path / "test.csv")

        # Combine columns into a single text column
        self.train_df = self.train_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )
        self.test_df = self.test_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )
        if DEBUG:
            print("Sample of loaded data:")
            print(self.train_df.head())

    def _vectorize(self, max_features=5000) -> None:
        """Vectorize the text data using TF-IDF."""
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features
        )

        self.X_train = self.vectorizer.fit_transform(self.train_df["text"].to_list())
        self.X_dev = self.vectorizer.transform(self.dev_df["text"].to_list())
        self.X_test = self.vectorizer.transform(self.test_df["text"].to_list())

        self.y_train = self.train_df["label"].to_numpy()
        self.y_dev = self.dev_df["label"].to_numpy()
        self.y_test = self.test_df["label"].to_numpy()

        if DEBUG:
            print("Sample of vectorized data:")
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            print("X_dev shape:", self.X_dev.shape)
            print("y_dev shape:", self.y_dev.shape)
            print("X_test shape:", self.X_test.shape)
            print("y_test shape:", self.y_test.shape)

    def _normalize(self) -> None:
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_dev = self.scaler.transform(self.X_dev)
        self.X_test = self.scaler.transform(self.X_test)

    @property
    def label_mapping(self):
        """Return the mapping of label indices to class names."""
        return {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

class AGNewsWord2Vec(AGNews):
    """PyTorch Dataset wrapper for the AG News dataset."""

    def __init__(
        self, path: Path | str = "data/ag_news", max_length: int = 256
    ) -> None:
        self.path = Path(path)

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")
            download_ag_news()
        self._load_data()
        self._embeddings()
        
    def _get_word2vec(self) -> KeyedVectors:
        # TODO: Add pretrained alternative and different hyperparameters
        if Path("models/ag_news_word2vec.kv").exists():
            LOGGER.log_and_print(Panel("Loading pre-trained Word2Vec model...", style="bold yellow"))
            kv = KeyedVectors.load("models/ag_news_word2vec.kv", mmap="r")
        else:
            # Train Word2Vec on the training data
            LOGGER.log_and_print(Panel("Training Word2Vec model...", style="bold yellow"))
            w2v = Word2Vec(
                sentences=self.train_df["tokens"].to_list(),
                vector_size=100,
                window=5,
                min_count=3,
                workers=4,
                sg=1,
                negative=10,
                epochs=10,
                seed=RANDOM_SEED,
            )
            kv = w2v.wv
            kv.save("models/ag_news_word2vec.kv")
        kv.add_vector("<PAD>", np.zeros(100))  # Add a padding token with zero vector
        return kv
        
        
    def _embeddings(self):
        # Simple Preprocessing for Word2Vec
        self.train_df = self.train_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )
        
        self.kv = self._get_word2vec()
        
        # Apply embeddings to the datasets
        self.train_df = self.train_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )

    def _pad_sequences(self, sequences, max_length):
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                padded.append(seq + [self.kv["<PAD>"]] * (max_length - len(seq)))
            else:
                padded.append(seq[:max_length])
        return padded

    def get_torch_dataset(self, split=Literal["train", "dev", "test"], max_length=256):
        if split == "train":
            X = self._pad_sequences(self.train_df["embeddings"].to_list(), max_length)
            y = self.train_df["label"].to_numpy()
        elif split == "dev":
            X = self._pad_sequences(self.dev_df["embeddings"].to_list(), max_length)
            y = self.dev_df["label"].to_numpy()
        elif split == "test":
            X = self._pad_sequences(self.test_df["embeddings"].to_list(), max_length)
            y = self.test_df["label"].to_numpy()
        else:
            raise ValueError("Invalid split name. Use 'train', 'dev', or 'test'.")

        return TorchDataset(X, y)

    def nearest_neighbors(self, word, topn=5):
        """Find the nearest neighbors of a word in the embedding space."""
        if word in self.kv:
            return self.kv.most_similar(word, topn=topn)
        else:
            return []
