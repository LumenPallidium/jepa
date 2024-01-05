import torch
from einops import rearrange
from einops.layers.torch import Rearrange

def add_util_norm(module, norm = "weight", **norm_kwargs):
    """Adds a norm from torch.nn.utils to a module"""
    if norm == "weight":
        norm_f = torch.nn.utils.weight_norm
    elif norm == "spectral":
        norm_f = torch.nn.utils.spectral_norm
    else:
        norm_f = lambda x: x
    return norm_f(module, **norm_kwargs)

def remove_util_norm(module, norm = "weight"):
    if norm == "weight":
        norm_f = torch.nn.utils.remove_weight_norm
    elif norm == "spectral":
        norm_f = torch.nn.utils.remove_spectral_norm
    else:
        norm_f = lambda x: x
    return norm_f(module)

class Attention2d(torch.nn.Module):
    """Based on ViT implementation from Phil Wang:
    https://github.com/lucidrains/musiclm-pytorch/blob/main/musiclm_pytorch/musiclm_pytorch.py
    
    Parameters
    ----------
    dim : int
        The dimension of the input and output
    dim_head : int, optional
        The dimension of the subspace for each head, by default 64
    n_heads : int, optional
        The number of heads, by default 8
    dropout : float, optional
        The dropout rate, by default 0.
    bias : bool, optional
        Whether to use bias in the linear layers, by default False"""
    def __init__(self, 
                 dim,
                 n_heads = 8,
                 dropout = 0.,
                 bias = False,
                 cross = False,
                 flash_attention = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.cross = cross
        self.flash_attention = flash_attention

        self.dropout = dropout
        self.inner_dim = self.dim_head * n_heads

        self.norm = torch.nn.LayerNorm(dim)

        self.W_q = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_k = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_v = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_o = torch.nn.Linear(self.inner_dim, dim, bias = bias)

        if self.flash_attention:
            self.mha = torch.nn.MultiheadAttention(dim, n_heads, dropout = dropout, batch_first=True)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, y = None):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        if self.cross and (not y is None):
            q, k, v = self.W_q(x), self.W_k(y), self.W_v(y)
        else:
            q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        
        if self.flash_attention:
            output = self.mha(q, k, v)[0]
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

            attention = torch.einsum("b h i k, b h j k -> b h i j", q, k)
            attention = attention / (self.dim_head ** 0.5)
            attention = self.dropout(attention.softmax(dim = -1))

            output = torch.einsum("b h i j, b h j k -> b h i k", attention, v)
            output = rearrange(output, "b h n d -> b n (h d)")
        output = self.W_o(output)

        return self.dropout(output)
    
    def attention_single_head(self, x):
        """Interpretation function to filter output to a single head, seperate from forward to avoid
        if-else in main function"""

        x = self.norm(x)

        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

        attention = torch.einsum("b h i k, b h j k -> b h i j", q, k)

        attention = attention / (self.dim_head ** 0.5)
        attention = attention.softmax(dim = -1)

        return attention
    
class FeedForward(torch.nn.Module):
    """A feed forward layer for transformers.
    
    Parameters
    ----------
    dim : int
        The dimension of the input and output
    hidden_dim : int
        The dimension of the hidden layer
    dropout : float, optional
        The dropout rate, by default 0.
    activation : torch.nn.Module, optional
        The activation function, by default torch.nn.GELU"""
    def __init__(self, 
                 dim, 
                 hidden_dim, 
                 dropout = 0.,
                 activation = torch.nn.GELU):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(torch.nn.Module):
    """A residual transformer with attention and feed forward layers.
    
    Parameters
    ----------
    dim : int, optional
        The dimension of the residual stream
    depth : int, optional
        The number of attention and feed forward layers
    heads : int, optional
        The number of attention heads, by default 8
    head_dim : int, optional
        The dimension of the subspaces of the attention heads, by default 64
    dropout : float, optional
        The dropout rate, by default 0.
    positional_embedding : bool, optional
        Whether to use a positional embedding, by default True
    context : int, optional
        The number of context frames, by default None
    activation : torch.nn.Module, optional
        The activation function, by default torch.nn.GELU
    """
    def __init__(self, 
                 dim = 512, 
                 depth = 4, 
                 heads = 8, 
                 dropout = 0.4,
                 positional_embedding = True,
                 context = None,
                 activation = torch.nn.GELU,
                 ema_decay = 0.996,
                 first_layer_norm = True,
                 cross = False,
                 flash_attention = False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cross = cross

        self.ema_decay = ema_decay

        self.has_util_norm = False

        if first_layer_norm:
            self.norm = torch.nn.LayerNorm(dim)
        else:
            self.norm = torch.nn.Identity()

        if positional_embedding and (context is not None):
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, context, dim))
        else:
            self.register_buffer("pos_embedding", torch.zeros(1, 1, dim))

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention2d(dim, n_heads = heads, dropout = dropout, cross = cross,
                            flash_attention = flash_attention),
                FeedForward(dim, dim, dropout = dropout, activation = activation)
            ]))

    def forward(self, x, y = None, stop_at = None, pos_embedding = None):
        """Transformer forward. Can stop at a certain layer for layer-dropout,
        as well as be supplied with a positional embedding (e.g. for shared
        positional embeddings between models)"""
        if pos_embedding is None:
            pos_embedding = self.pos_embedding
        x = self.norm(x) + pos_embedding

        for i, (attention, ff) in enumerate(self.layers):
            x = x + attention(x, y = y)
            x = x + ff(x)

            y = None # disable cross attention after first layer
            if (stop_at is not None) and (i >= (stop_at - 1)):
                break
        return x
    
    def filtered_forward(self, x, indices):
        """Given a tensor of the form [masked target, context] and indices describing where on the image
        they come from, add positional embedding and pass through the transformer."""
        pos_embedding = self.pos_embedding[:, indices, :]
        return self.forward(x, pos_embedding = pos_embedding)
    
    def get_attentions(self, x):
        """Modified forward for just getting the output of the nth attention layers, for use in the
        attention visualization."""
        attentions = []
        x = x + self.pos_embedding
        for i, (attention, ff) in enumerate(self.layers):
            attention_i = attention.attention_single_head(x)
            attentions.append(attention_i)

            x = x + attention(x)
            x = x + ff(x)
        return attentions
    
    def ema_update(self, new_model):
        for ema_param, new_param in zip(self.parameters(), new_model.parameters()):
            ema_param.data.copy_(ema_param.data * self.ema_decay + (1 - self.ema_decay) * new_param.data)

    def add_util_norm(self, norm_name = "weight"):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                add_util_norm(module, norm_name)
        self.has_util_norm = True
    
    def remove_util_norm(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                remove_util_norm(module)
        self.has_util_norm = False




        