import torch
import torch.nn as nn
from typing import Optional, Dict

class LSTMClassifier(nn.Module):
    """Class for a v Classifier on NLP."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initialise the LSTM class.

        Args:
            config (Optional[Dict], optional): The configuration of the LSTM.
                                               Defaults to None.
        """

        super(LSTMClassifier, self).__init__()

        if config is None:
            config = {}

        

        self.convs = 
        


