import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=7, dropout=0.3):
        super(BertClassifier, self).__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)

        # Hidden size of BERT (768 for base)
        hidden_size = self.bert.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        self.dropout = nn.Dropout(dropout)

        # Softmax for inference only (NOT used in training loss)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  
        # shape: (batch_size, hidden_size)

        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        # shape: (batch_size, num_labels)

        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probs = self.softmax(logits)
        preds = torch.argmax(probs, dim=1)
        return preds