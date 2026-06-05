import torch.nn as nn

from .bert_embedding import BERTEmbedding

class UtteranceClassifier(nn.Module):
    def __init__(self, dataset_config, bert_config):
        super().__init__()

        self.embedding = BERTEmbedding(bert_config)

        self.classifier = nn.Linear(
            self.embedding.hidden_size,
            dataset_config["num_labels"]
        )

    def optimizer_groups(self):
        groups = {}

        groups.update(
            self.embedding.optimizer_groups()
        )

        groups["utterance_head"] = self.classifier

        return groups

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        """
        input_ids: [B, L]
        attention_mask: [B, L]

        Returns:
            logits: [B, num_labels]
        """

        h = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = self.classifier(h)

        return logits