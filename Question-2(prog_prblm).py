import torch
import torch.nn as nn
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, d_ff=512):
        super(SimpleTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads


        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
   
 def forward(self, x):
      # x shape: (batch_size, seq_len, d_model)
        
        # --- Multi-head self-attention with residual & norm ---
        attn_output, _ = self.mha(x, x, x)            
        x = self.norm1(x + attn_output)

      # --- Feed-forward network with residual & norm ---
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
# --- Test the encoder block --
batch_size = 32
seq_len = 10
d_model = 128

# Random input: batch of 32 sentences, each with 10 tokens, embedding size 128
x = torch.randn(batch_size, seq_len, d_model)

encoder = SimpleTransformerEncoder(d_model=d_model, n_heads=8)
output = encoder(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
