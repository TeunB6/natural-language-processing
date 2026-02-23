from src.data.data import AGNews, AGNewsWord2Vec
from src.training.eval import evaluate_model, analyze_model_errors
from src.training.train import train_model, get_model
from src.training.gridsearch import svm_gridsearch
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from rich.panel import Panel
from src.const import CONSOLE, DATA_DIR, DEBUG, RETRAIN_MODEL, LOGGER
from src.utils.ui import cli_menu
from typing import Optional
import os
import sys


def assignment1_showcase(
    choice: Optional[int] = None,
) -> None:
    """Showcase Assignment 1.

    Args:
        ds (AGNews): The AG News dataset object.
        choice (Optional[int], optional): The choice of functionality to
                                          showcase. Defaults to None.
    """
    LOGGER.info("Loading AG News dataset...")
    ds = AGNews(path=DATA_DIR)
    LOGGER.info("Dataset loaded successfully")

    def train_and_evaluate() -> None:
        """Train baseline models and evaluate them on dev/test sets."""

        # Logistic Regression model
        if RETRAIN_MODEL:
            panel = Panel(
                "Training: Logistic Regression Model...",
                style="bold yellow",
            )
            LOGGER.log_and_print(panel)
            logreg_model = train_model(
                LogisticRegression(max_iter=1000),
                ds,
                assignment=1,
            )
        else:
            logreg_model = get_model(
                LogisticRegression(max_iter=1000),
                ds,
                assignment=1,
            )

        # SVM model
        if RETRAIN_MODEL:
            panel = Panel("Training: SVM...", style="bold yellow")
            LOGGER.log_and_print(panel)
            svm_model = train_model(
                SVC(kernel="linear", C=0.1),
                ds,
                assignment=1,
            )
        else:
            svm_model = get_model(
                SVC(kernel="linear", C=0.1),
                ds,
                assignment=1,
            )

        # Evaluate both models on the set
        cli_menu(
            "Evaluate on which set?",
            {
                "Dev Set": lambda: (
                    evaluate_model(logreg_model, ds),
                    evaluate_model(svm_model, ds),
                ),
                "Test Set": lambda: (
                    evaluate_model(logreg_model, ds, use_test=True),
                    evaluate_model(svm_model, ds, use_test=True),
                ),
                "Back to Menu": lambda: None,
            },
        )

    def grid_search() -> None:
        """Perform SVM grid search to find the best hyperparameters."""
        panel = Panel(
            "WARNING: Running SVM Grid Search can take a long time.",
            style="bold red",
        )
        LOGGER.log_and_print(panel)
        param_grid = {"C": np.logspace(-3, 3, 7), "kernel": ["linear"]}

        svm_gridsearch(
            ds=ds,
            param_grid=param_grid,
            eval=True,
            assignment=1,
        )

    def analyze_errors():
        """Analyze model errors."""
        logreg_model = get_model(
            LogisticRegression(max_iter=1000),
            ds,
            assignment=1,
        )
        svm_model = get_model(
            SVC(kernel="linear", C=0.1),
            ds,
            assignment=1,
        )

        cli_menu(
            "Analyze errors for which split?",
            {
                "Dev Set": lambda: (
                    analyze_model_errors(
                        logreg_model, ds, split="dev", min_examples=10
                    ),
                    analyze_model_errors(svm_model, ds, split="dev", min_examples=10),
                ),
                "Test Set": lambda: (
                    analyze_model_errors(
                        logreg_model, ds, split="test", min_examples=10
                    ),
                    analyze_model_errors(svm_model, ds, split="test", min_examples=10),
                ),
                "Back to Menu": lambda: None,
            },
        )

        analyze_model_errors(logreg_model, ds, split="dev", min_examples=10)

        analyze_model_errors(svm_model, ds, split="dev", min_examples=10)

    if choice is not None:
        if choice == 1:
            train_and_evaluate()
        elif choice == 2:
            grid_search()
        elif choice == 3:
            analyze_errors()
        return None
    else:
        cli_menu(
            "Select a functionality to showcase:",
            {
                "Train and Evaluate Baseline Models": train_and_evaluate,
                "Perform SVM Grid Search": grid_search,
                "Analyze Errors on Models": analyze_errors,
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel("[bold yellow]Returning to Main Menu...[/bold yellow]")
                ),
            },
        )

def assignment2_showcase(
    choice: Optional[int] = None,
):
    ds = AGNewsWord2Vec(path=DATA_DIR)

    def word_similarity(ds: AGNewsWord2Vec):
        """Showcase word similarity functionality."""
        while True:
            word = CONSOLE.input("Enter a word to find its nearest neighbors (x to exit): ").strip()
            
            if word.lower() == 'x':
                LOGGER.log_and_print(Panel("[bold yellow]Exiting Word Similarity Showcase...[/bold yellow]"))
                break
            
            neighbors = ds.nearest_neighbors(word, topn=10)
            if neighbors:
                panel = Panel(
                    f"Nearest neighbors for [bold green]{word}[/bold green]:\n\n"
                    + "\n".join(
                        [
                            f"{neighbor[0]} (similarity: {neighbor[1]:.4f})"
                            for neighbor in neighbors
                        ]
                    ),
                    style="bold blue",
                )
                LOGGER.log_and_print(panel)
            else:
                panel = Panel(
                    f"No neighbors found for [bold red]{word}[/bold red]. It may not be in the vocabulary.",
                    style="bold red",
                )
                LOGGER.log_and_print(panel)
    
    def train_and_evaluate_cnn_lstm(ds: AGNewsWord2Vec):
        """Train and evaluate CNN and LSTM models."""
        train = ds.get_torch_dataset("train")
        dev = ds.get_torch_dataset("dev")
        test = ds.get_torch_dataset("test")
                
        # Placeholder for CNN and LSTM training/evaluation
        panel = Panel(
            "CNN and LSTM training/evaluation functionality is not implemented yet.",
            style="bold yellow",
        )
        LOGGER.log_and_print(panel)

    if choice is not None:
        if choice == 1:
            word_similarity(ds)
        elif choice == 2:
            train_and_evaluate_cnn_lstm(ds)
        return None
    else:
        cli_menu(
            "Select a functionality to showcase:",
            {
                "Examine Word Similarity": lambda: word_similarity(ds),
                "Train and Evaluate CNN/LSTM Models": lambda: train_and_evaluate_cnn_lstm(ds),
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel("[bold yellow]Returning to Main Menu...[/bold yellow]")
                ),
            },
        )


def main():
    """Run main pipeline."""
    LOGGER.info("Starting NLP Pipeline...")

    if DEBUG:
        print("=== MAIN FUNCTION STARTED ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script arguments: {sys.argv}")
        print("Loading AG News dataset...")

    # Parser for allow running specific functionalities directly from command
    # line without going through menus.
    parser = argparse.ArgumentParser()

    parser.add_argument("--assignment", type=int, choices=[1, 2], help="Assignment number")
    parser.add_argument(
        "--functionality",
        type=int,
        choices=[1, 2, 3],
        help="Functionality number",
    )

    args = parser.parse_args()

    if args.assignment and args.functionality:
        if args.assignment == 1:
            if args.functionality == 1:
                assignment1_showcase(choice=1)
            elif args.functionality == 2:
                assignment1_showcase(choice=2)
            elif args.functionality == 3:
                assignment1_showcase(choice=3)
        elif args.assignment == 2:
            if args.functionality == 1:
                assignment2_showcase(choice=1)
            elif args.functionality == 2:
                assignment2_showcase(choice=2)
    else:
        panel = Panel("AG News NLP Pipeline", style="bold blue")
        LOGGER.log_and_print(panel)
        while True:
            cli_menu(
                "Select an assignment to showcase different functionalities:",
                {
                    "Assignment 1 - Dataset Showcase & Baseline Models": (
                        lambda: assignment1_showcase()
                    ),
                    "Assignment 2 - CNN & LSTM": (lambda: assignment2_showcase()),
                    "Exit": lambda: exit(0),
                },
            )


if __name__ == "__main__":
    panel = Panel("Starting NLP Pipeline...", style="bold green")
    LOGGER.log_and_print(panel)
    main()
