from typing import Callable
from src.const import CONSOLE, DEBUG


def cli_menu(question: str, options: dict[str, Callable]) -> int:
    """Display a CLI menu for the user to select different assignments and functionalities."""

    CONSOLE.print(f"\n[bold cyan]{question}[/bold cyan]")
    for i, option in enumerate(options.keys(), 1):
        CONSOLE.print(f"{i}. {option}")

    choice = CONSOLE.input("\n[bold cyan]Enter your choice:[/bold cyan] ").strip()
    if DEBUG:
        print(f"User selected option: {choice}")  # Debug print to check user input
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        selected_option = list(options.values())[int(choice) - 1]
        selected_option()
    else:
        CONSOLE.print("[bold red]Invalid choice. Please try again.[/bold red]")
