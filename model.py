import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    # word embedding
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout:float()) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        #to avoid overfit
        self.droupout = nn.Droupout(dropout)
        # create positional encoding, a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # (Seq_Len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0)/d_model) # log space for numerical stability.
        # print(torch.sin(position * div_term).shape)
        # print(pe[:,0::2].shape)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model) to process a batch of sentence
        self.register_buffer("pe", pe) # to save not-learned parameters inside the model
    def forward(self, x):
        # positional encoding wouldn't change
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.droupout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        # for numerical stability
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim_= True)
        return self.alpha * (x-mean)/math.sqrt(std + self.eps) + self. bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float) -> None:
        super().__init()
        self.linear_1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.droupout = nn.Droupout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2
    def forward(self, x):
        #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, D_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.droupout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.linear(d_model, d_model)
        self.w_k = nn.linear(d_model, d_model)
        self.w_v = nn.linear(d_model, d_model)
        self.w_o = nn.linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask = None, dropout = nn.Dropout):
        d_k = query.shape[-1]
        # (Batch ,h, seq_len, d_k) - > (Batch, h, seq_len, seq_len)
        attention_scores = (query@key.transpose(-2, -1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim  =-1) #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value, attention_scores # for visualizing

    def forward(self, q,k,v, mask):
        query = self.w_q(q) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # doing transpose, so we see each head with full sequence_len by d_k

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # doing transpose, so we see each head with full sequence_len by d_k

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, h, d_k) --> (Batch, h, Seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # doing transpose, so we see each head with full sequence_len by d_k

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,  key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k) 
        
        return self.w_o(x)
         
        # why use contigous?
        # When you perform certain operations (like .transpose()), the actual data in memory isn't rearranged; 
        # only the size and stride of the tensor are changed. 
        # This can lead to non-contiguous tensors where the data is not laid out in the regular, expected order in memory.
        # As a general rule, if you're changing the shape of a tensor with operations like view,
        #  and you've performed operations like transpose, permute, or slicing with non-continuous step before, you'll likely need .contiguous().


class ResidualConnection(nn.Module):
    
    def __init__(self,droupout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(droupout)
        self.norm = LayerNormalization()
    def forwward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # in the paper, it's sublayer first and then norm, but most of the implementation did it this way
    


class EncoderBlock(nn.Module):

    # it's called self_attention b/c it's the same input with different role
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        # we used the src_mask to hide the padding work 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.Modulelist) ->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, \
                 feed_forward_block: FeedForwardBlock, dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, x, encoder_output,src_mask,tgt_mask):
        # we used the src_mask to hide the padding work 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) 
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) 
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
class Decoder(nn.Module):
    def __init__(self, layers: nn.Modulelist) ->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output,src_mask,tgt_mask)
        return self.norm(x)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.proj = nn.linear(d_model, vocab_size)
    def forward(self, x):
        # (Batch, Sqe_Len, d_model) -> (Batch, Seq_len, Vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                  src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt= self.src_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) 
    def projct(self, x):
        return self.projection_layer(x)
# to combine all of the blocks, so given any hyperparam, we can initialize a transformer
        
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len:int, tgt_seq_len:int, d_model: int = 512, N: int = 6, h:int = 8, 
                      dropout: float = 0.1, d_ff: int=2048):
    # create the embedding layers:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # create the positional encoding layers:
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder blocks:
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff)
        decoder_block = EncoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,src_pos, tgt_pos, projection_layer)


    # Initialize the parameters:
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return transformer


        

