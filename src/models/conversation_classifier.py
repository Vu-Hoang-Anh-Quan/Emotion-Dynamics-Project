import torch
import torch.nn as nn
import math
from transformers import BertModel
from .bert_embedding import BERTEmbedding

def check_tensor(name, x):
    if torch.isnan(x).any():
        print(f"{name}: NaN")
    if torch.isinf(x).any():
        print(f"{name}: Inf")

    print(
        f"{name}: "
        f"min={x.min().item():.4f}, "
        f"max={x.max().item():.4f}, "
        f"mean={x.mean().item():.4f}"
    )

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_config, max_turns = 64): 
        super().__init__()

        self.input_dim = input_dim
        self.attention_dim = attention_config["dim"]

        # W_q, W_k, W_v
        self.query = nn.Linear(input_dim, self.attention_dim)
        self.key = nn.Linear(input_dim, self.attention_dim)
        self.value = nn.Linear(input_dim, self.attention_dim)
        for layer in [self.query, self.key, self.value]:
            # nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # max turns
        self.max_turns = max_turns

        # Relational embedding
        self.relative_bias = nn.Embedding(2 * max_turns - 1, 1)
        nn.init.normal_(self.relative_bias.weight, std=0.02)

        # Dropout
        self.dropout = nn.Dropout(attention_config["dropout"])

        # self.residual_proj = nn.Linear(input_dim, self.attention_dim)
        self.residual_proj = nn.Identity()
        self.layer_norm = nn.LayerNorm(input_dim)

        # self.self_bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, utterance_mask): # To do padding mask, we must pass utterance_mask in
        # x_norm = self.layer_norm(x)
        x_norm = x # NOTICE THIS LINE

        # x : [B, T, input_dim]
        B, T, D = x_norm.shape

        # [B, T, attention_dim]
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)
        # Q = torch.nn.functional.normalize(Q, dim=-1)
        # K = torch.nn.functional.normalize(K, dim=-1)

        # check_tensor("Q", Q)
        # check_tensor("K", K)
        # check_tensor("V", V)

        # Transpose K to [B, attention_dim, T]
        K_t = K.transpose(-1, -2)

        # [B, T, T]
        attention_scores = torch.matmul(Q, K_t)

        # Scale
        attention_scores = attention_scores / math.sqrt(self.attention_dim)

        # check_tensor("Attention scores before bias and mask", attention_scores)

        # Relative positions
        positions = torch.arange(
            T,
            device=x.device
        )

        # [T, T]
        relative_positions = (
            positions.unsqueeze(1)
            - positions.unsqueeze(0)
        )

        relative_positions += self.max_turns - 1 # Shift id to nonnegativeconfigs/default.json

        # [T, T]
        relative_bias = self.relative_bias(
            relative_positions
        ).squeeze(-1) # Squeeze as after embedding, each index map to a 1, produce [T, T, 1] -> squeeze the last dimension

        # Broadcast and add the relative_bias
        # [T,T] -> [B,T,T]
        attention_scores = attention_scores + relative_bias

        #Learnable self_bias
        # attention_scores += torch.eye(T, device=x.device) * self.self_bias

        # Casual mask that let utterance i only attend to <=i
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        attention_scores = attention_scores.masked_fill(
            causal_mask,
            -1e4
        )

        # Padding mask only on key
        # [B, T]
        padding_mask = (utterance_mask == 0) # Bool already
        # [B, 1, T] to broadcast to [B, T, T], mask all columns/keys of the attention scores
        padding_mask = padding_mask.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(
            padding_mask,
            -1e4
        )

        # check_tensor("Attention scores after mask", attention_scores)
        # finite_scores = attention_scores[
        #     torch.isfinite(attention_scores)
        # ]
        # print(configs/default.json
        #     finite_scores.min().item(),
        #     finite_scores.max().item()
        # )

        # Softmax
        # attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.softmax(
            attention_scores.float(),
            dim=-1
        ).to(attention_scores.dtype) # Force to FP 32 for softmax to avoid NaN, then convert back to original dtype (possibly FP16) for later matmul. This is a common practice when using mixed precision training, as softmax can produce NaN in FP16 if the input values are too large or too small.

        # attention_entropy = (
        #     -attention_probs *
        #     torch.log(attention_probs + 1e-12)
        # ).sum(dim=-1).mean()

        # print(attention_entropy.item())

        # diag_weight = attention_probs.diagonal(
        #     dim1=1,
        #     dim2=2
        # ).mean()
        # print(diag_weight.item())

        # check_tensor("Attention probabilities", attention_probs)

        # Dropout
        attention_probs = self.dropout(attention_probs)

        # Multiply with V to produce [B, T, attention_dim]
        output = torch.matmul(attention_probs, V)

        # Residual but with a projection layer
        output = output + self.residual_proj(x_norm)

        output = self.layer_norm(output) # LAYER NORM HERE

        return output

        # Residual by concatenate
        # [B, T, embedding_final_size + attention_size]
        residual_output = torch.cat((x_norm, output), dim=-1)
        return residual_output

class ConversationClassifier(nn.Module):
    def __init__(
            self, 
            embedding,
            dataset_config,
            attention_config,
            head_config,
        ):
        super(ConversationClassifier, self).__init__()

        # Load embedding
        self.embedding = embedding

        # Hidden size of BERT (768 for base)
        bert_output_dim = self.embedding.output_dim

        # Self attention layer
        self.self_attention = SelfAttention(
            input_dim=bert_output_dim,
            attention_config=attention_config,
            max_turns=dataset_config["max_turns"]
        )

        # Classification head
        self.classifier = nn.Sequential(
            # nn.Linear(attention_config["dim"]+bert_output_dim, 128),
            nn.Linear(attention_config["dim"], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(head_config["dropout"]),
            nn.Linear(128, dataset_config["num_labels"])
        )

        # Softmax for inference only (NOT used in training loss)
        self.softmax = nn.Softmax(dim=1)

    def optimizer_groups(self):
        groups = {}

        groups.update(
            self.embedding.optimizer_groups()
        )

        groups["attention"] = self.self_attention
        groups["conversation_head"] = self.classifier

        return groups

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        utterance_mask = batch["utterance_mask"] # [B, T]

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
        h = self.self_attention.forward(h, utterance_mask=utterance_mask) # [B, T, bert_output_dim+attention_dim]
        
        # Classify
        logits = self.classifier(h) # [B, T, num_labels]

        # check_tensor("Logits", logits)

        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds # [B, T]