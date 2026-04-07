import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        self.q_weight = nn.Linear(self.embedding_dim, self.num_heads*self.qk_length)
        self.k_weight = nn.Linear(self.embedding_dim, self.num_heads*self.qk_length)
        self.v_weight = nn.Linear(self.embedding_dim, self.num_heads*self.value_length)

        self.mha_layer = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.mha_layer_mask = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.ff_layer = FeedForwardNN(self.embedding_dim, self.ffn_hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.ln1_layer = nn.LayerNorm(self.embedding_dim)
        self.ln2_layer = nn.LayerNorm(self.embedding_dim)
        self.ln3_layer = nn.LayerNorm(self.embedding_dim)

        self.q_weight_cross = nn.Linear(self.embedding_dim, self.num_heads*self.qk_length)
        self.k_weight_cross = nn.Linear(self.embedding_dim, self.num_heads*self.qk_length)
        self.v_weight_cross = nn.Linear(self.embedding_dim, self.num_heads*self.value_length)


    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        Q, K, V = self.q_weight(x), self.k_weight(x), self.v_weight(x)
        out = x + self.dropout_layer(self.mha_layer_mask(Q, K, V, tgt_mask))
        out = self.ln1_layer(out)

        if enc_x is not None:
            Q_cross, K_cross, V_cross = self.q_weight_cross(out), self.k_weight_cross(enc_x), self.v_weight_cross(enc_x)
            out = out + self.dropout_layer(self.mha_layer(Q_cross, K_cross, V_cross, src_mask))
            out = self.ln2_layer(out)
        
        out = out + self.dropout_layer(self.ff_layer(out))
        out = self.ln3_layer(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pe = PositionalEncoding(self.embedding_dim, dropout, max_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.num_heads, self.embedding_dim, self.ffn_hidden_dim, self.qk_length, self.value_length, dropout) for _ in range(self.num_layers)])

        self.flinear = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        x = x.long()
        out = self.pe(self.embedding(x))
        for layer in self.decoder_layers:
            out = layer(out, enc_x, tgt_mask, src_mask)
        out = self.flinear(out)
        return out
