import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, attention_config):
        super().__init__()
        

        self.hidden_size = attention_config["hidden_size"]
        self.num_heads = attention_config["num_heads"]

        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(attention_config["dropout"])

    def forward(self, x, utterance_mask):
        """
        x: [B, T, H]
        utterance_mask: [B, T]
            1 = valid token
            0 = padding
        """

        B, T, H = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, heads, T, T]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores /= math.sqrt(self.head_dim)

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
        # [B, 1, 1, T] to broadcast to [B, heads, T, T], mask all columns/keys of the attention scores
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
        attention_scores = attention_scores.masked_fill(
            padding_mask,
            -1e4
        )

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs_for_loss = attention_probs
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, H)

        output = self.out_proj(output)

        return {
            "logits": output,
            "attention_probs": attention_probs_for_loss
        }
    
class FeedForward(nn.Module):
    def __init__(self, attention_config):
        super().__init__()

        self.hidden_size = attention_config["hidden_size"]
        self.ff_dim = attention_config["ff_dim"]
        self.dropout = nn.Dropout(attention_config["dropout"])

        self.linear1 = nn.Linear(self.hidden_size, self.ff_dim)
        self.linear2 = nn.Linear(self.ff_dim, self.hidden_size)

    def forward(self, x):

        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        x = self.dropout(x)

        return {
            "logits": x
        }

class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention_config):
        super().__init__()

        self.hidden_size = attention_config["hidden_size"]

        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.self_attention = MultiHeadSelfAttention(attention_config)

        self.ffn = FeedForward(attention_config)

    def forward(self, x, utterance_mask):

        # Self-attention block
        residual = x

        x = self.norm1(x)

        attention_output = self.self_attention(
            x,
            utterance_mask,
        )
        attention_logits = attention_output["logits"]
        attention_probs = attention_output["attention_probs"]

        x = residual + attention_logits

        # Feed-forward block
        residual = x

        x = self.norm2(x)

        ffn_output = self.ffn(x)
        ffn_logits = ffn_output["logits"]

        x = residual + ffn_logits

        return {
            "logits": x,
            "attention_probs": attention_probs
        }
    
class TransformerEncoder(nn.Module):
    def __init__(self, attention_config):
        super().__init__()

        self.num_layers = attention_config["num_layers"]

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(attention_config)
            for _ in range(self.num_layers)
        ])

    def forward(self, x, utterance_mask=None):

        attention_probs_list = []

        for layer in self.layers:
            layer_output = layer(x, utterance_mask)
            x = layer_output["logits"]
            attention_probs = layer_output["attention_probs"]
            attention_probs_list.append(attention_probs)

        return {
            "logits": x,
            "attention_probs": attention_probs_list # [num_layers, B, heads, T, T]
        }

# Old self-attention module, not used anymore, but kept for reference. The new self-attention module is MultiHeadSelfAttention above.
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

        # x : [B, T, input_dim]
        B, T, D = x.shape

        # [B, T, attention_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Transpose K to [B, attention_dim, T]
        K_t = K.transpose(-1, -2)

        # [B, T, T]
        attention_scores = torch.matmul(Q, K_t)

        # Scale
        attention_scores = attention_scores / math.sqrt(self.attention_dim)


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

        # Softmax
        # attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.softmax(
            attention_scores.float(),
            dim=-1
        ).to(attention_scores.dtype) # Force to FP 32 for softmax to avoid NaN, then convert back to original dtype (possibly FP16) for later matmul. This is a common practice when using mixed precision training, as softmax can produce NaN in FP16 if the input values are too large or too small.

        attention_probs_for_loss = attention_probs

        # Dropout
        attention_probs = self.dropout(attention_probs)

        # Multiply with V to produce [B, T, attention_dim]
        output = torch.matmul(attention_probs, V)

        # Residual but with a projection layer
        output = output + self.residual_proj(x)

        output = self.layer_norm(output)

        return {
            "logits": output,
            "attention_probs": attention_probs_for_loss
        }

        # Residual by concatenate
        # [B, T, embedding_final_size + attention_size]
        residual_output = torch.cat((x_norm, output), dim=-1)
        return residual_output
