from src.data.data import AGNews


def main():
    # Ensure the AG News dataset is downloaded and saved as CSV files
    ds = AGNews(path="data/ag_news")


if __name__ == "__main__":
    main()
