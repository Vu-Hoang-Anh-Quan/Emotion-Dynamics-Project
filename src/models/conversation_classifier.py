import torch
import torch.nn as nn
import math
from ..utils.debug import check_tensor

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
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # max turns
        self.max_turns = max_turns

        # Relational embedding
        self.relative_bias = nn.Embedding(2 * max_turns - 1, 1)
        nn.init.normal_(self.relative_bias.weight, std=0.005)

        # Dropout
        self.dropout = nn.Dropout(attention_config["dropout"])
        # self.dropout = nn.Dropout(0.0)

        # self.residual_proj = nn.Linear(input_dim, self.attention_dim)
        self.residual_proj = nn.Identity()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.self_bias = nn.Parameter(torch.tensor(1.0))

    def apply_mask(self, attention_scores, utterance_mask):
        B, T, T = attention_scores.shape

        # Casual mask that let utterance i only attend to <=i
        causal_mask = torch.triu(
            torch.ones(T, T, device=attention_scores.device),
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

        # Just in case
        all_masked = (attention_scores == -1e4).all(dim=-1)
        if all_masked.any():
            print(f"There are ALL MASKED rows:\n{torch.nonzero()}")

        return attention_scores

    def forward(self, x, utterance_mask): # To do padding mask, we must pass utterance_mask in
        x_norm = self.layer_norm(x)

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

        # Learnable self bias        
        attention_scores += torch.eye(T, device=x.device) * self.self_bias
        # print(f"{self.self_bias}\n")
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

        relative_positions += self.max_turns - 1 # Shift id to nonnegative

        # [T, T]
        relative_bias = self.relative_bias(
            relative_positions
        ).squeeze(-1) # Squeeze as after embedding, each index map to a 1, produce [T, T, 1] -> squeeze the last dimension

        # Broadcast and add the relative_bias
        # [T,T] -> [B,T,T]
        attention_scores = attention_scores + relative_bias

        # Masking
        attention_scores = self.apply_mask(attention_scores, utterance_mask)

        # check_tensor("Attention scores after mask", attention_scores)
        # finite_scores = attention_scores[
        #     torch.isfinite(attention_scores)
        # ]
        # print(
        #     finite_scores.min().item(),
        #     finite_scores.max().item()
        # )

        if not torch.isfinite(attention_scores).all():
            print("Bad scores before softmax")

        # Softmax
        # attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.softmax(
            attention_scores.float(),
            dim=-1
        ).to(attention_scores.dtype) # Force to FP 32 for softmax to avoid NaN, then convert back to original dtype (possibly FP16) for later matmul. This is a common practice when using mixed precision training, as softmax can produce NaN in FP16 if the input values are too large or too small.

        if not torch.isfinite(attention_probs).all():
            print("Bad probs after softmax")

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
        output = torch.matmul(
            attention_probs, 
            V
        )

        # Residual but with a projection layer
        output = (output + self.residual_proj(x_norm))/2.0
        return output

        # Residual by concatenate
        # [B, T, embedding_final_size + attention_size]
        # residual_output = torch.cat((x_norm, output.to(x.dtype)), dim=-1)
        # return residual_output

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
        
        if not torch.isfinite(h).all():
            print("Classifier input bad")

        # Classify
        logits = self.classifier(h) # [B, T, num_labels]
        # x = h
        # for i in range(5):
        #     if i == 4:
        #         print(
        #             "Before final linear:",
        #             torch.isfinite(x).all(),
        #             x.dtype,
        #             x.abs().max()
        #         )

        #         print(
        #             "num inf:",
        #             torch.isinf(x).sum()
        #         )

        #         print(
        #             "num nan:",
        #             torch.isnan(x).sum()
        #         )
        #     x = self.classifier[i](x)

        # check_tensor("Logits", logits)

        return logits
        # return x

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds # [B, T]