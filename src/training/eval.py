from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.base import BaseEstimator
from src.data.data import AGNews
from typing import Optional

# Visualization
from rich.table import Table
from rich.panel import Panel

from src.const import CONSOLE, DEBUG
from src.utils.error_analysis_pipeline import ErrorAnalysisPipeline


def evaluate_model(model: BaseEstimator, ds: AGNews, use_test: bool = False):
    """Evaluate a trained model on the dev set and display results.

    Args:
        model: Trained sklearn model
        ds: AGNews dataset
        use_test: If True, evaluate on test set; otherwise use dev set
        analyze_errors: If True, run detailed error analysis
        min_error_examples: Minimum examples per error type to show
    """
    # Predict on the dev set
    X, y = (ds.X_dev, ds.y_dev) if not use_test else (ds.X_test, ds.y_test)

    y_pred = model.predict(X)

    # Display initial metrics in a table
    CONSOLE.print(
        Panel(
            f"{model.__class__.__name__} Results on {"Test" if use_test else "Dev"} Set",
            style="bold green",
        )
    )
    
    if DEBUG:
        print(y_pred.shape, y.shape, X.shape)    
    
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    table.add_row(
        "F1 Score (weighted)", f"{f1_score(y, y_pred, average='weighted'):.4f}"
    )
    CONSOLE.print(table)
    cm = confusion_matrix(y, y_pred)

    # Make Confusion Matrix more readable by mapping label indices to class names
    label_mapping = ds.label_mapping
    cm_table = Table(title="Confusion Matrix (with Class Names)")
    cm_table.add_column("Predicted \\ Actual", style="bold white")
    for i in range(cm.shape[0]):
        cm_table.add_column(label_mapping[i + 1], style="magenta")

    for i in range(cm.shape[0]):
        row = [label_mapping[i + 1]] + [str(cm[i, j]) for j in range(cm.shape[1])]
        cm_table.add_row(*row)
    CONSOLE.print(cm_table)


def analyze_model_errors(
    model: BaseEstimator,
    ds: AGNews,
    split: str = "dev",
    min_examples: int = 10,
    show_hardest: bool = True,
):
    """
    Run error analysis on model predictions.

    Args:
        model: Trained sklearn model
        ds: AGNews dataset
        split: Dataset split to analyze ('dev' or 'test')
        min_examples: Minimum number of examples to show per error type
        show_hardest: Whether to show the hardest cases
    """
    pipeline = ErrorAnalysisPipeline()
    pipeline.run(
        model=model,
        ds=ds,
        split=split,
        min_examples=min_examples,
        wrap_width=80,
        show_full_text=True,
    )
