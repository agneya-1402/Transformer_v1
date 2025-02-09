import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

# Multi-Head Attention 
# Computes attention between pair of position in input sequence

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define linear layers => query(Q), key(K), value(V), and output transformation
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Split last dimension to (num_heads, d_k) and transpose
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        # Combine heads by reversing split and transpose operations
        batch_size, num_heads, seq_length, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split into heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Compute attention output on each head
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply final linear layer
        output = self.W_o(self.combine_heads(attn_output))
        return output


# Position-wise Feed-Forward Network

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Positional Encoding
# Inject position information of each token in input, sine and cosine fucntions used to generate it

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer 
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]


# Encoder Layer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention sublayer with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward sublayer with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Decoder Layer

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self-attention for decoder input
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        # Cross-attention with encoder output
        x2 = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(x2))
        # Feed-forward network
        x2 = self.feed_forward(x)
        x = self.norm3(x + self.dropout(x2))
        return x


# Complete Transformer Model
# combine encode and decode layers

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6, max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def make_src_mask(self, src):
        # Create mask for source padding tokens 
        # (0 is used for padding)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # Create mask for target padding tokens
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)

        # Subsequent mask to prevent attending to future tokens
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
        return tgt_mask
        
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Embed and add positional encoding to source
        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src = self.positional_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # Embed and add positional encoding to target
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        output = self.fc_out(tgt)
        return output


# Usage 

if __name__ == "__main__":
    device = torch.device("cpu")

    # Define hyperparameters 
    SRC_VOCAB_SIZE = 10000   # Source vocabulary size
    TGT_VOCAB_SIZE = 10000   # Target vocabulary size
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    MAX_SEQ_LENGTH = 100
    DROPOUT = 0.1

    # Initialize Transformer model
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF,
                        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, MAX_SEQ_LENGTH, DROPOUT).to(device)
    print("Transformer model instantiated.")

    # dummy data 
    # (batch_size x seq_length)
    batch_size = 2
    src_seq = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, MAX_SEQ_LENGTH)).to(device)
    tgt_seq = torch.randint(1, TGT_VOCAB_SIZE, (batch_size, MAX_SEQ_LENGTH)).to(device)

    # Forward pass through model
    output = model(src_seq, tgt_seq)

    # batch_size, seq_length, TGT_VOCAB_SIZE
    print("Output shape:", output.shape)  
