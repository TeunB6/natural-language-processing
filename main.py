from src.data.data import AGNews
from src.training.eval import evaluate_model
from src.training.train import train_model, get_model

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Visualization
from rich.panel import Panel
from rich.progress import track
from src.const import CONSOLE

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
    
    # Run CLI menu
    CONSOLE.print(Panel("AG News NLP Pipeline", style="bold blue"))
    CONSOLE.print("\n[bold]Available Assignments:[/bold]")
    CONSOLE.print("1. Assignment 1 - Dataset Showcase & Baseline Models")
    CONSOLE.print("2. Exit")

    choice = CONSOLE.input("\n[bold cyan]Enter your choice (1-2):[/bold cyan] ").strip()

    if choice == "1":
        assignment1_showcase(ds)
    elif choice == "2":
        CONSOLE.print("[bold yellow]Exiting...[/bold yellow]")
    else:
        CONSOLE.print("[bold red]Invalid choice. Please try again.[/bold red]")
        

if __name__ == "__main__":
    main()
