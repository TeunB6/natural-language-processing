from torch.utils.data import DataLoader
from src.utils.data import TorchDataset
import torch.nn as nn
import torch.optim as optim
import torch

from rich.progress import track
from src.const import LOGGER, DEVICE

import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-dark-palette")  # seaborn style
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),  # set default figure size
        "axes.labelsize": 16,  # set default label size
        "axes.grid": True,  # enable grid by default
        
        "xtick.labelsize": 14,  # set default tick label size
        "ytick.labelsize": 14,  # set default tick label size
        "axes.titlesize": 18,  # set default title size
        "legend.fontsize": 16,  # set default legend font size
        "lines.linewidth": 4,  # set default line width
        "text.usetex": False,  # use LaTeX
        "font.family": "serif",  # use a serif font
        "image.cmap": "magma",  # set default colormap
    }
)

class Trainer:
    def __init__(self, model: nn.Module, train_data: TorchDataset, eval_data: TorchDataset, batch_size: int=32):
        self.model = model.to(DEVICE)
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            shuffle=False,
        )
        
        self.history = {"train_loss": [], "eval_loss": []}
    
    def train(self, num_epochs: int=10, learning_rate: float=1e-3, config: dict=None):
        """Train the model."""
        
        # Set up optimizer and loss function
        optimizer = config.get("optimizer", optim.Adam)(self.model.parameters(), lr=learning_rate)
        criterion = config.get("criterion", nn.CrossEntropyLoss)()

        for epoch in track(range(num_epochs), description=f"Training Epoch {epoch+1}/{num_epochs}"):
            self.model.train()
            total_loss = 0
            for batch in track(self.train_loader, description="Training Batches"):
                # Update model parameters based on the batch
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Log training loss
                self.history["train_loss"].append(loss.item())
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            eval_loss = self.evaluate(criterion)
            self.history["eval_loss"].append(eval_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def evaluate(self, criterion):
        """Evaluate the model on the evaluation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in track(self.eval_loader, description="Evaluating"):
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.eval_loader)
        return avg_loss
    
    def plot_history(self, show: bool=True, save_path: str=None):
        """Plot the training and evaluation loss history."""
        x_train = np.asarray(range(1, len(self.history["train_loss"]) + 1))
        x_eval = np.asarray(range(1, len(self.history["eval_loss"]) + 1))
        x_eval = x_eval * (len(self.history["train_loss"]) / len(self.history["eval_loss"]))  # Scale eval x-axis to match train
        
        plt.figure()
        plt.plot(x_train, self.history["train_loss"], label="Train Loss")
        plt.plot(x_eval, self.history["eval_loss"], label="Eval Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss History")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            LOGGER.info(f"Training history plot saved to {save_path}")
        if show:
            plt.show()
