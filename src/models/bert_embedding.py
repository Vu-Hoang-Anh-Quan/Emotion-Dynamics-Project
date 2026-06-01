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
    for layer_idx in range(total_layers):
        for param in bert_model.encoder.layer[layer_idx].parameters():
            param.requires_grad = (layer_idx >= total_layers - k)

class BERTEmbedding(nn.Module):
    def __init__(self, bert_config):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            bert_config["model_name"]
        )

        freeze_bert_except_last_k(
            self.bert,
            k=bert_config["freeze_except_last_k"]
        )

        self.dropout = nn.Dropout(
            bert_config["dropout"]
        )

        self.hidden_size = self.bert.config.hidden_size
    
    def optimizer_groups(self):
        return {
            "bert": self.bert,
        }

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [N, L]
        attention_mask: [N, L]

        Returns:
            embeddings: [N, hidden_size]
        """

        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        token_embeddings = (
            bert_outputs.last_hidden_state
        ) # [N, L, hidden_size]

        mask = attention_mask.unsqueeze(-1).float() # [N, L, 1]

        summed = (
            token_embeddings * mask # [N, L, hidden_size]
        ).sum(dim=1) # [N, hidden_size]

        lengths = (
            mask.sum(dim=1) # [N, 1]
            .clamp(min=1e-9)
        )

        embeddings = summed / lengths # [N, hidden_size]

        embeddings = self.dropout(embeddings)

        return embeddings