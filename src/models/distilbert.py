from torch import nn
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification


class DistilBERTClassifer(nn.Module):
    """DistilBERT Classifer class."""

    def __init__(
        self, model_name: str = "distilbert-base-uncased", num_classes: int = 4
    ) -> None:
        super(DistilBERTClassifer, self).__init__()

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        return outputs.logits

    def predict(
        self, x: torch.Tensor, return_prob: bool = True
    ) -> torch.Tensor:
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)

        if return_prob:
            return F.softmax(logits, dim=1)

        return torch.argmax(logits, dim=1) + 1
