from src.data.download import download_ag_news


def main():
    # Ensure the AG News dataset is downloaded and saved as CSV files
    download_ag_news()


if __name__ == "__main__":
    main()
