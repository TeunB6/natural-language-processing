from src.data.data import AGNews
from src.training.eval import evaluate_model
from src.training.train import train_model, get_model
from src.training.gridsearch import svm_gridsearch
import argparse

# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np


# Visualization
from rich.panel import Panel
from src.const import CONSOLE, RESULTS_DIR, DATA_DIR, DEBUG
from src.utils.ui import cli_menu


RETRAIN_MODEL = False  # Set to True to retrain models every time, False to load from disk if available

def assignment1_showcase(ds: AGNews, choice: int = None):
    """Showcase the AG News dataset and vectorization."""

    def train_and_evaluate():
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
            svm_model = train_model(SVC(kernel="linear", max_iter=1000), ds)
        else:
            svm_model = get_model(SVC(kernel="linear", max_iter=1000), ds)
        evaluate_model(svm_model, ds)
    
    def grid_search():
        param_grid = {
            "C": np.logspace(-3, 3, 7),
            "kernel": ["linear"]
        }
        svm_gridsearch(ds, param_grid, RESULTS_DIR)
        
    if choice is not None:
        if choice == 1:
            train_and_evaluate()
        elif choice == 2:
            grid_search()
        return
    else:
        cli_menu(
            "Select a functionality to showcase:",
            {
                "Train and Evaluate Baseline Models": train_and_evaluate,
                "Perform SVM Grid Search": grid_search,
                "Back to Main Menu": lambda: CONSOLE.print("[bold yellow]Returning to Main Menu...[/bold yellow]")
            },
        )
    
def main():
    # Ensure the AG News dataset is downloaded and saved as CSV files
    
    if DEBUG:
        import os, sys
        print("=== MAIN FUNCTION STARTED ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script arguments: {sys.argv}")
    
    if DEBUG:
        print("Loading AG News dataset...")
    ds = AGNews(path=DATA_DIR)
    
    # Parser for allow running specific functionalities directly from command line without going through menus
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=int, choices=[1], help='Assignment number')
    parser.add_argument('--functionality', type=int, choices=[1, 2], help='Functionality number')
    args = parser.parse_args()
    
    if args.assignment and args.functionality:
        # Batch mode - skip menus
        if args.assignment == 1:
            if args.functionality == 1:
                assignment1_showcase(ds, choice=1)
            elif args.functionality == 2:
                assignment1_showcase(ds, choice=2)
    else:
        # Interactive mode - show menus
        CONSOLE.print(Panel("AG News NLP Pipeline", style="bold blue"))
        while True:
            cli_menu("Select an assignment to showcase different functionalities:", {
                "Assignment 1 - Dataset Showcase & Baseline Models": lambda: assignment1_showcase(ds),
                "Exit": lambda: exit(0)
            })
    
if __name__ == "__main__":
    print("Starting NLP Pipeline...")
    main()
