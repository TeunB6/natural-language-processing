from src.data.download import download_ag_news
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
from pathlib import Path
from src.const import DEBUG


class AGNews:
    """Class to handle loading and vectorizing the AG News dataset."""

    def __init__(self, path: Path | str = "data/ag_news"):
        self.path = Path(path)

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                print("CSV files not found, downloading dataset...")
            download_ag_news()
        self.load_data()
        self.vectorize()

    def load_data(self):
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

    def vectorize(self, max_features=5000):
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
    
    @property
    def label_mapping(self):
        """Return a mapping of label indices to class names."""
        return {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech"
        }
