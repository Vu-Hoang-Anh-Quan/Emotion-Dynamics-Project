import torch
import torch.nn as nn
from .attention import TransformerEncoder

class ConversationClassifier(nn.Module):
    def __init__(
            self, 
            embedding,
            dataset_config,
            attention_config,
            head_config
        ):
        super(ConversationClassifier, self).__init__()

        # Load embedding
        self.embedding = embedding

        # Hidden size of BERT (768 for base)
        bert_output_dim = self.embedding.output_dim

        # Self attention layer
        self.transformer = TransformerEncoder(
            attention_config=attention_config,
        )

        # Classification head
        # self.classifier = nn.Sequential(
        #     # nn.Linear(attention_config["dim"]+bert_output_dim, 128),        #     nn.Linear(128, dataset_config["num_labels"])

        #     nn.Linear(attention_config["dim"], 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(head_config["dropout"]),
        #     nn.Linear(128, dataset_config["num_labels"])
        # )
        self.classifier = nn.Linear(attention_config["hidden_size"], dataset_config["num_labels"])

        # Softmax for inference only (NOT used in training loss)
        self.softmax = nn.Softmax(dim=1)

    def optimizer_groups(self):
        groups = {}

        groups.update(
            self.embedding.optimizer_groups()
        )

        groups["attention"] = self.transformer
        groups["conversation_head"] = self.classifier

        return groups

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        utterance_mask = batch["utterance_mask"] # [B, T]
        utterance_ids = batch["utterance_ids"]
        speaker_ids = batch["speaker_ids"]

        B, T, L = input_ids.shape # [B, T, L]

        # Flatten for BERT
        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L) # [B * T, L]

        # Embedding
        h = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Reshape back to dialogue
        h = h.view(B, T, -1) # [B, T, hidden_size]
        
        # Pass into self attention
        attention_output = self.transformer.forward(
            h,
            utterance_mask=utterance_mask,
            utterance_ids=utterance_ids,
            speaker_ids=speaker_ids
        )
        h = attention_output["logits"]
        attention_probs = attention_output["attention_probs"] # [B, T, T]
        attention_outputs = attention_output["attention_outputs"] # Note the s difference
        
        # Classify
        logits = self.classifier(h) # [B, T, num_labels]

        # check_tensor("Logits", logits)

        return {
            "logits": logits,
            "attention_probs": attention_probs,
            "attention_outputs": attention_outputs,
            "utterance_mask": utterance_mask
        }