import torch.nn as nn
from transformers import BertModel

class BERTEmbedding(nn.Module):
    def __init__(self, bert_config):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            bert_config["model_name"]
        )

        self.dropout = nn.Dropout(
            bert_config["dropout"]
        )

        self.hidden_size = self.bert.config.hidden_size
        self.output_dim = bert_config["output_dim"]

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU()
        )
    
    def optimizer_groups(self):
        return {
            "bert": self.bert,
        }
    
    def set_trainable_layers(self, k=4):
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(self.bert.encoder.layer)

        if k > total_layers:
            raise RuntimeError(
                f"Cannot unfreeze {k} layers. "
                f"BERT only has {total_layers} layers."
            )

        # Freeze all except last k layers
        for layer_idx in range(total_layers):
            trainable = layer_idx >= total_layers - k

            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [N, L]
        attention_mask: [N, L]

        Returns:
            embeddings: [N, hidden_size]
        """

        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        # # check if multiple dim -> flatten
        # if input_ids.dim() > 3:
        #     raise NotImplementedError
        # elif input_ids.dim() == 3:
        #     # [B, T, L]
        #     B, T, _ = input_ids.shape
        #     input_ids = input_ids.view(B * T, -1)
        #     # Assume that attention_mask should follow the same
        #     attention_mask = attention_mask.view(B * T, -1)


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

        output = self.projection(embeddings)

        output = self.dropout(output)

        return output