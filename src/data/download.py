import polars as pl
from src.const import DATA_DIR, RANDOM_SEED


def download_ag_news():
    splits = {"train": "train.jsonl", "test": "test.jsonl"}
    train_df = pl.read_ndjson(
        "hf://datasets/sh0416/ag_news/" + splits["train"]
    )
    test_df = pl.read_ndjson("hf://datasets/sh0416/ag_news/" + splits["test"])

    # Randomize and take dev split from train
    train_df = train_df.sample(fraction=1.0, seed=RANDOM_SEED)
    dev_size = int(0.1 * len(train_df))
    dev_df = train_df[:dev_size]
    train_df = train_df[dev_size:]

    # Save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.write_csv(DATA_DIR / "train.csv")
    dev_df.write_csv(DATA_DIR / "dev.csv")
    test_df.write_csv(DATA_DIR / "test.csv")
