from src.data.data import AGNews
from src.training.eval import evaluate_model
from src.training.train import train_model, get_model

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Visualization
from rich.panel import Panel
from rich.progress import track
from src.const import CONSOLE, RESULTS_DIR
from src.utils.ui import cli_menu


RETRAIN_MODEL = False  # Set to True to retrain models every time, False to load from disk if available

def assignment1_showcase(ds: AGNews):
    """Showcase the AG News dataset and vectorization."""

    # Train a simple logistic regression model
    
    CONSOLE.print(Panel(f"Logistic Regression Model", style="bold green"))
    if RETRAIN_MODEL:
        logreg_model = train_model(LogisticRegression(max_iter=1000), ds)
    else:
        logreg_model = get_model(LogisticRegression(max_iter=1000), ds)
    evaluate_model(logreg_model, ds)

    # Train SVM model
    CONSOLE.print(Panel("Linear SVM Model", style="bold green"))  
    if RETRAIN_MODEL:
        svm_model = train_model(SVC(kernel="linear"), ds)
    else:
        svm_model = get_model(SVC(kernel="linear"), ds)
    evaluate_model(svm_model, ds)
    


def main():
    # Ensure the AG News dataset is downloaded and saved as CSV files
    ds = AGNews(path="data/ag_news")
    
    CONSOLE.print(Panel("AG News NLP Pipeline", style="bold blue"))
    cli_menu(
        "Select an assignment to showcase different functionalities:",
        {
            "Assignment 1 - Dataset Showcase & Baseline Models": lambda: assignment1_showcase(ds),
            "Exit": lambda: CONSOLE.print("[bold yellow]Exiting...[/bold yellow]")
            # Future assignments can be added here
        },
    )        

if __name__ == "__main__":
    main()
