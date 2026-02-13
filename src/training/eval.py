from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.base import BaseEstimator
from src.data.data import AGNews

# Visualization
from rich.table import Table

from src.const import CONSOLE

def evaluate_model(model: BaseEstimator, ds: AGNews):
    """Evaluate a trained model on the dev set and display results."""
    # Predict on the dev set
    y_pred = model.predict(ds.X_dev)

    # Display initial metrics in a table
    table = Table(title=f"{model.__class__.__name__} Results on Dev Set")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Accuracy", f"{accuracy_score(ds.y_dev, y_pred):.4f}")
    table.add_row("F1 Score (weighted)", f"{f1_score(ds.y_dev, y_pred, average='weighted'):.4f}")
    CONSOLE.print(table)
    cm = confusion_matrix(ds.y_dev, y_pred)
    
    # Make Confusion Matrix more readable by mapping label indices to class names
    label_mapping = ds.label_mapping
    cm_table = Table(title="Confusion Matrix (with Class Names)")
    cm_table.add_column("Predicted \\ Actual", style="bold white")
    for i in range(cm.shape[0]):
        cm_table.add_column(label_mapping[i+1], style="magenta")
        
    for i in range(cm.shape[0]):
        row = [label_mapping[i+1]] + [str(cm[i, j]) for j in range(cm.shape[1])]
        cm_table.add_row(*row)
    CONSOLE.print(cm_table)