from src.data.data import AGNews
from src.training.eval import evaluate_model, analyze_model_errors
from src.training.train import train_model, get_model
from src.training.gridsearch import svm_gridsearch
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from rich.panel import Panel
from src.const import CONSOLE, DATA_DIR, DEBUG, RETRAIN_MODEL
from src.utils.ui import cli_menu
from typing import Optional
import os, sys


def assignment1_showcase(ds: AGNews, choice: Optional[int] = None) -> None:
    """Showcase Assignment 1.

    Args:
        ds (AGNews): The AG News dataset object.
        choice (Optional[int], optional): The choice of functionality to
                                          showcase. Defaults to None.
    """

    def train_and_evaluate() -> None:
        """Train baseline models and evaluate them on dev/test sets."""

        # Logistic Regression model.
        if RETRAIN_MODEL:
            CONSOLE.print(
                Panel(
                    f"Training: Logistic Regression Model...",
                    style="bold yellow",
                )
            )
            logreg_model = train_model(LogisticRegression(max_iter=1000), ds)
        else:
            logreg_model = get_model(LogisticRegression(max_iter=1000), ds)

        # SVM model.
        if RETRAIN_MODEL:
            CONSOLE.print(Panel(f"Training: SVM...", style="bold yellow"))
            svm_model = train_model(SVC(kernel="linear", C=0.1), ds)
        else:
            svm_model = get_model(SVC(kernel="linear", C=0.1), ds)

        # Evaluate both models on the set
        cli_menu(
            "Evaluate on which set?",
            {
                "Dev Set": lambda: (
                    evaluate_model(logreg_model, ds),
                    CONSOLE.input(
                        "[bold cyan]Press Enter to analyze SVM errors on Dev Set...[/bold cyan]"
                    ),
                    evaluate_model(svm_model, ds),
                ),
                "Test Set": lambda: (
                    evaluate_model(logreg_model, ds, use_test=True),
                    CONSOLE.input(
                        "[bold cyan]Press Enter to analyze SVM errors on Test Set...[/bold cyan]"
                    ),
                    evaluate_model(svm_model, ds, use_test=True),
                ),
                "Back to Menu": lambda: None,
            },
        )

    def grid_search() -> None:
        """Perform SVM grid search to find the best hyperparameters."""
        CONSOLE.print(
            Panel(
                "WARNING: Running SVM Grid Search can take a long time.",
                style="bold red",
            )
        )
        param_grid = {"C": np.logspace(-3, 3, 7), "kernel": ["linear"]}

        svm_gridsearch(ds=ds, param_grid=param_grid, eval=True)

    def analyze_errors():
        """Analyze model errors."""
        logreg_model = get_model(LogisticRegression(max_iter=1000), ds)
        svm_model = get_model(SVC(kernel="linear", C=0.1), ds)

        cli_menu(
            "Analyze errors for which split?",
            {
                "Dev Set": lambda: (
                    analyze_model_errors(
                        logreg_model, ds, split="dev", min_examples=10
                    ),
                    CONSOLE.input(
                        "[bold cyan]Press Enter to analyze SVM errors on Dev Set...[/bold cyan]"
                    ),
                    analyze_model_errors(
                        svm_model, ds, split="dev", min_examples=10
                    ),
                ),
                "Test Set": lambda: (
                    analyze_model_errors(
                        logreg_model, ds, split="test", min_examples=10
                    ),
                    CONSOLE.input(
                        "[bold cyan]Press Enter to analyze SVM errors on Test Set...[/bold cyan]"
                    ),
                    analyze_model_errors(
                        svm_model, ds, split="test", min_examples=10
                    ),
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
                "Back to Main Menu": lambda: CONSOLE.print(
                    "[bold yellow]Returning to Main Menu...[/bold yellow]"
                ),
            },
        )


def main():
    """Run main pipeline."""
    if DEBUG:
        print("=== MAIN FUNCTION STARTED ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script arguments: {sys.argv}")
        print("Loading AG News dataset...")

    ds = AGNews(path=DATA_DIR)

    # Parser for allow running specific functionalities directly from command
    # line without going through menus.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assignment", type=int, choices=[1], help="Assignment number"
    )
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
                assignment1_showcase(ds, choice=1)
            elif args.functionality == 2:
                assignment1_showcase(ds, choice=2)
            elif args.functionality == 3:
                assignment1_showcase(ds, choice=3)
    else:
        CONSOLE.print(Panel("AG News NLP Pipeline", style="bold blue"))
        while True:
            cli_menu(
                "Select an assignment to showcase different functionalities:",
                {
                    "Assignment 1 - Dataset Showcase & Baseline Models": lambda: assignment1_showcase(
                        ds
                    ),
                    "Exit": lambda: exit(0),
                },
            )


if __name__ == "__main__":
    CONSOLE.print("Starting NLP Pipeline...", style="bold green")
    main()
