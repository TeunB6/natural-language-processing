from typing import Optional, List, Dict
from rich.table import Table
import textwrap
from src.const import CONSOLE
from src.training.analyze_errors import ErrorAnalyzer


class ErrorAnalysisPipeline:
    """Utility class to run the full error analysis pipeline for a model and dataset."""

    def __init__(self):
        self.console = CONSOLE

    def run(
        self,
        model,
        ds,
        split: str = "dev",
        min_examples: int = 10,
        show_hardest: bool = True,
        wrap_width: int = 80,
    ):
        analyzer = ErrorAnalyzer(
            model, ds, min_examples=min_examples, wrap_width=wrap_width
        )

        analyzer.analyze(split=split, get_confidence=True)
        analyzer.display_summary()
        analyzer.display_error_matrix()

        for (pred, actual), examples in sorted(
            analyzer.misclassifications.items(), key=lambda x: -len(x[1])
        ):
            analyzer._display_error_group(pred, actual, examples, min_examples, True)

        if show_hardest:
            hardest = analyzer.get_hardest_cases(top_n=10)
            if hardest:
                self._display_hardest_cases(hardest, wrap_width)

    def _display_hardest_cases(self, examples: List[Dict], wrap_width: int):
        table = Table(
            title="[bold red]Top 10 Lowest Confidence Errors[/bold red]",
            show_lines=True,
        )
        table.add_column("Conf", style="red", width=8)
        table.add_column("Description (Full Wrap)", style="white")
        table.add_column("Path", style="yellow", width=20)

        for ex in examples:
            wrapped = textwrap.fill(ex["text"], width=wrap_width)
            path = f"{ex['actual_class']} -> {ex['predicted_class']}"
            table.add_row(f"{ex['confidence']:.4f}", wrapped, path)

        self.console.print(table)
