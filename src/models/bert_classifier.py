import torch
import torch.nn as nn
from transformers import BertModel

def freeze_bert_except_last_k(bert_model, k=4):
    # Freeze embeddings
    for param in bert_model.embeddings.parameters():
        param.requires_grad = False

    # Total layers (BERT-base = 12)
    total_layers = len(bert_model.encoder.layer)

    if (k > total_layers):
        raise RuntimeError("Number of unfreezed layers is larger than total number of layers in BERT-base (12)")

    # Freeze all except last k layers
    for layer_idx in range(total_layers - k):
        for param in bert_model.encoder.layer[layer_idx].parameters():
            param.requires_grad = False

class BertGRUClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=7, dropout_bert=0.1, dropout_head=0.3, gru_hidden = 256, freeze_except_last_k=4):
        super(BertGRUClassifier, self).__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        freeze_bert_except_last_k(self.bert, k=freeze_except_last_k)
        self.dropout_bert = nn.Dropout(dropout_bert)

        # Hidden size of BERT (768 for base)
        hidden_size = self.bert.config.hidden_size

        # GRU over utterances
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False # Only consider past utterances
        )
        # self.dropout_head = nn.Dropout(dropout_head)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, num_labels)
        )


        # Softmax for inference only (NOT used in training loss)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        B, T, L = input_ids.shape # [B, T, L]

        # Flatten for BERT
        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L) # [B * T, L]

        # BERT output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        h = outputs.last_hidden_state[:, 0, :] # [B * T, hidden_size]
        h = self.dropout_bert(h)

        # Reshape back to dialogue
        h = h.view(B, T, -1) # [B, T, hidden_size]

        # Pass into GRU
        h_ctx, _ = self.gru(h)  # [B, T, gru_hidden]
        # h_ctx = self.dropout_head(h_ctx)
        
        # Classify
        logits = self.classifier(h_ctx) # [B, T, num_labels]

        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds # [B, T]