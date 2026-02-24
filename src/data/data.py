from src.data.download import download_ag_news
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import polars as pl
from pathlib import Path
from src.const import DEBUG, LOGGER, RANDOM_SEED, MODEL_DIR, DATA_DIR
from rich.panel import Panel
from rich.progress import track

import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from src.utils.data import TorchDataset
from src.utils.singleton import SingletonMeta

from torch.utils.data import Dataset
from torch import as_tensor, Tensor
from torch.nn.functional import pad, one_hot

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

class AGNewsWord2Vec(AGNews, SingletonMeta):
    """PyTorch Dataset wrapper for the AG News dataset."""

    def __init__(
        self, path: Path | str | None = None, verbose: bool = True
    ) -> None:
        self.path = Path(path) if path is not None else DATA_DIR
        self.verbose = verbose

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")
            download_ag_news()
        self._load_data()
        self._embeddings()
        
    def _get_word2vec(self) -> KeyedVectors:
        # TODO: Add pretrained alternative and different hyperparameters
        model_path = MODEL_DIR / "ag_news_word2vec.kv"
        if model_path.exists():
            if self.verbose:
                LOGGER.log_and_print(Panel("Loading pre-trained Word2Vec model...", style="bold yellow"))
            kv = KeyedVectors.load(str(model_path), mmap="r") # for some reason this doesn't work with pathlib.Path objects, so we convert to string
        else:
            # Train Word2Vec on the training data
            if self.verbose:
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
            kv.save(str(model_path))
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
        seq_wrap = track(sequences, description="Padding sequences") if self.verbose else sequences
        for seq in seq_wrap:
            if len(seq) < max_length:
                padded.append(seq + [self.kv["<PAD>"]] * (max_length - len(seq)))
            else:
                padded.append(seq[:max_length])
        return padded

    def get_torch_dataset(self, split=Literal["train", "dev", "test"], max_length=256):
        if split == "train":
            X = self._pad_sequences(self.train_df["embeddings"].to_list(), max_length)
            y = one_hot(as_tensor(self.train_df["label"] - 1), num_classes=4).float()
        elif split == "dev":
            X = self._pad_sequences(self.dev_df["embeddings"].to_list(), max_length)
            y = one_hot(as_tensor(self.dev_df["label"] - 1), num_classes=4).float()
        elif split == "test":
            X = self._pad_sequences(self.test_df["embeddings"].to_list(), max_length)
            y = one_hot(as_tensor(self.test_df["label"] - 1), num_classes=4).float()
        else:
            raise ValueError("Invalid split name. Use 'train', 'dev', or 'test'.")

        return TorchDataset(X, y)

    def nearest_neighbors(self, word, topn=5):
        """Find the nearest neighbors of a word in the embedding space."""
        if word in self.kv:
            return self.kv.most_similar(word, topn=topn)
        else:
            return []


class AGNewsWord2VecDataset(AGNewsWord2Vec, Dataset):
    """PyTorch Dataset wrapper for the AG News dataset with Word2Vec embeddings."""
    
    def __init__(
        self, path: Path | str| None = None, split=Literal["train", "dev", "test"], verbose: bool = True
    ) -> None:
        self.ds = AGNewsWord2Vec(path=path, verbose=verbose)
        self.df = {
            "train": self.ds.train_df,
            "dev": self.ds.dev_df,
            "test": self.ds.test_df
        }[split]
    
    def __len__(self) -> int:
        return len(self.df)
    

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset."""
        embedding = as_tensor(self.df[idx, "embeddings"])
        padded_embedding = pad(embedding, (0, 0, 0, 256 - embedding.shape[0]), value=0)  # Pad to max_length
        label = self.df[idx, "label"]
        one_hot_label = one_hot(as_tensor(label - 1), num_classes=4)  # Convert to zero-based index for one-hot encoding
        one_hot_label = one_hot_label.float()  # Convert to float for compatibility with loss functions
        return padded_embedding, one_hot_label