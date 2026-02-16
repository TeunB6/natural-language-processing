from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from rich.table import Table
from rich.panel import Panel
import textwrap
from src.const import CONSOLE
from src.data.data import AGNews


class ErrorAnalyzer:
    def __init__(
        self,
        model,
        ds: AGNews,
        min_examples: int = 10,
        show_full_text: bool = False,
        wrap_width: int = 80,
    ):
        self.model = model
        self.ds = ds
        self.min_examples = min_examples
        self.show_full_text = show_full_text
        self.wrap_width = wrap_width
        self.console = CONSOLE
        self.X = None
        self.y = None
        self.df = None
        self.predictions = None
        self.confidence = None
        self.misclassifications: Dict[Tuple[int, int], List[Dict]] = {}
        self.error_stats = {}

    def analyze(self, split: str = "dev", get_confidence: bool = False) -> Dict:
        self._load_split(split)
        self._generate_predictions(get_confidence)
        self._extract_misclassifications()
        self._compute_statistics()
        return self.misclassifications

    def _load_split(self, split: str):
        if split == "dev":
            self.X, self.y, self.df = self.ds.X_dev, self.ds.y_dev, self.ds.dev_df
        elif split == "test":
            self.X, self.y, self.df = self.ds.X_test, self.ds.y_test, self.ds.test_df
        else:
            raise ValueError(f"Unknown split: {split}")

    def _generate_predictions(self, get_confidence: bool):
        self.predictions = self.model.predict(self.X)
        if get_confidence:
            if hasattr(self.model, "predict_proba"):
                self.confidence = np.max(self.model.predict_proba(self.X), axis=1)
            elif hasattr(self.model, "decision_function"):
                decisions = self.model.decision_function(self.X)
                decisions = (
                    np.max(np.abs(decisions), axis=1)
                    if len(decisions.shape) > 1
                    else np.abs(decisions)
                )
                self.confidence = 1 / (1 + np.exp(-decisions))

    def _extract_misclassifications(self):
        misclass_mask = self.predictions != self.y
        indices = np.where(misclass_mask)[0]
        label_map = self.ds.label_mapping
        texts = self.df["text"].to_list()

        errors = defaultdict(list)
        for idx in indices:
            actual, pred = self.y[idx], self.predictions[idx]
            errors[(pred, actual)].append(
                {
                    "text": texts[idx],
                    "actual_label": actual,
                    "predicted_label": pred,
                    "actual_class": label_map[actual],
                    "predicted_class": label_map[pred],
                    "confidence": (
                        self.confidence[idx] if self.confidence is not None else None
                    ),
                    "index": idx,
                }
            )
        self.misclassifications = dict(errors)

    def _compute_statistics(self):
        total = len(self.y)
        errors = sum(len(v) for v in self.misclassifications.values())
        self.error_stats = {
            "total_samples": total,
            "total_errors": errors,
            "error_rate": errors / total if total > 0 else 0,
            "accuracy": (total - errors) / total if total > 0 else 0,
            "error_types": len(self.misclassifications),
        }

    def display_summary(self):
        s = self.error_stats
        panel = Panel(
            f"[bold cyan]Total Samples:[/][white] {s['total_samples']}[/]\n"
            f"[bold red]Total Errors:[/][white] {s['total_errors']}[/]\n"
            f"[bold yellow]Error Rate:[/][white] {s['error_rate']:.2%}[/]\n"
            f"[bold green]Accuracy:[/][white] {s['accuracy']:.2%}[/]",
            title="Model Error Summary",
        )
        self.console.print(panel)

    def display_error_matrix(self):
        labels = self.ds.label_mapping
        n = len(labels)
        table = Table(title="Confusion Matrix (Errors Only)")
        table.add_column("Pred \\ Actual", style="bold white")
        for i in range(1, n + 1):
            table.add_column(labels[i], style="magenta")

        for p in range(1, n + 1):
            row = [labels[p]]
            for a in range(1, n + 1):
                count = len(self.misclassifications.get((p, a), []))
                row.append(str(count) if count > 0 else "-")
            table.add_row(*row)
        self.console.print(table)

    def _display_error_group(self, pred, actual, examples, limit, full_text):
        labels = self.ds.label_mapping
        title = f"Predicted: [red]{labels[pred]}[/] | Actual: [green]{labels[actual]}[/] ({len(examples)} cases)"

        table = Table(title=title, show_lines=True)
        table.add_column("Idx", width=6)
        table.add_column("Text (Wrapped Description)")
        if self.confidence is not None:
            table.add_column("Conf", width=10)

        for ex in examples[:limit]:
            # Applying full description wrapping
            wrapped_text = textwrap.fill(ex["text"], width=self.wrap_width)
            row = [str(ex["index"]), wrapped_text]
            if self.confidence is not None:
                row.append(f"{ex['confidence']:.4f}")
            table.add_row(*row)

        self.console.print(table)

    def get_hardest_cases(self, top_n: int = 10) -> List[Dict]:
        """
        Get the hardest cases (lowest confidence predictions).
        """
        if self.confidence is None:
            self.console.print(
                "[yellow]Warning: Confidence scores not available.[/yellow]"
            )
            return []

        # Flatten all misclassifications into a single list
        all_errors = [
            ex for examples in self.misclassifications.values() for ex in examples
        ]

        # Sort by confidence
        return sorted(
            all_errors,
            key=lambda x: x["confidence"] if x["confidence"] is not None else 1.0,
        )[:top_n]
