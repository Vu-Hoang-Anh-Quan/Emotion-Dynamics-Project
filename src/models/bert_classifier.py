import torch
import torch.nn as nn
import math
from transformers import BertModel

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

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim, dropout_attention = 0.2, max_turns = 64): # Take a look at DailyDialog and specify max_turns
        super().__init__()

        self.input_dim = input_dim
        self.attention_dim = attention_dim

        # W_q, W_k, W_v
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
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
        self.dropout = nn.Dropout(dropout_attention)

        self.residual_proj = nn.Linear(input_dim, attention_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, utterance_mask): # To do padding mask, we must pass utterance_mask in
        x_norm = self.layer_norm(x)

        # x : [B, T, input_dim]
        B, T, D = x_norm.shape

        # [B, T, attention_dim]
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)
        Q = torch.nn.functional.normalize(Q, dim=-1)
        K = torch.nn.functional.normalize(K, dim=-1)

        check_tensor("Q", Q)
        check_tensor("K", K)
        check_tensor("V", V)

        # Transpose K to [B, attention_dim, T]
        K_t = K.transpose(-1, -2)

        # [B, T, T]
        attention_scores = torch.matmul(Q, K_t)

        # Scale
        attention_scores = attention_scores / math.sqrt(self.attention_dim)

        check_tensor("Attention scores before bias and mask", attention_scores)

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

        finite_scores = attention_scores[
            torch.isfinite(attention_scores)
        ]
        print(
            finite_scores.min().item(),
            finite_scores.max().item()
        )

        # Softmax
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Dropout
        attention_probs = self.dropout(attention_probs)

        check_tensor("Attention probabilities", attention_probs)

        # Multiply with V to produce [B, T, attention_dim]
        output = torch.matmul(attention_probs, V)

        # Residual but with a projection layer
        output = output + self.residual_proj(x_norm)
        return output


class BertClassifier(nn.Module):
    def __init__(
            self, 
            model_name="bert-base-uncased", 
            num_labels=7, 
            dropout_bert=0.1,
            dropout_attention=0.2,
            dropout_head=0.3, 
            attention_dim=512, 
            max_turns=64, 
            freeze_except_last_k=4
        ):
        super(BertClassifier, self).__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        freeze_bert_except_last_k(self.bert, k=freeze_except_last_k)
        self.dropout_bert = nn.Dropout(dropout_bert)

        # Hidden size of BERT (768 for base)
        bert_hidden_size = self.bert.config.hidden_size

        # Self attention layer
        self.self_attention = SelfAttention(
            input_dim=bert_hidden_size,
            attention_dim=attention_dim,
            max_turns=max_turns,
            dropout_attention=dropout_attention
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(128, num_labels)
        )

        # Softmax for inference only (NOT used in training loss)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, utterance_mask):
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

        # Pass into self attention
        h = self.self_attention.forward(h, utterance_mask=utterance_mask) # [B, T, attention_dim]
        
        # Classify
        logits = self.classifier(h) # [B, T, num_labels]

        check_tensor("Logits", logits)

        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds # [B, T]