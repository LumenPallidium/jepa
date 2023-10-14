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
                 bias = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads

        self.dropout = dropout
        self.inner_dim = self.dim_head * n_heads

        self.norm = torch.nn.LayerNorm(dim)

        self.W_q = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_k = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_v = torch.nn.Linear(dim, self.inner_dim, bias = bias)
        self.W_o = torch.nn.Linear(self.inner_dim, dim, bias = bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """Input shape is (batch, seq_len, dim)"""
        x = self.norm(x)

        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
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
                 dropout = 0.,
                 positional_embedding= True,
                 context = None,
                 activation = torch.nn.GELU,
                 ema_decay = 0.996,):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.heads = heads

        self.ema_decay = ema_decay

        self.has_util_norm = False

        if positional_embedding:
            assert context is not None, "Context must be provided if positional embedding is used"
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, context, dim))
        else:
            self.pos_embedding = torch.zeros(1, 1, dim)

        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention2d(dim, n_heads = heads, dropout = dropout),
                FeedForward(dim, dim, dropout = dropout, activation = activation)
            ]))

    def forward(self, x):
        x = x + self.pos_embedding

        for attention, ff in self.layers:
            x = x + attention(x)
            x = x + ff(x)
        return x
    
    def filtered_forward(self, x, indices):
        """Given a tensor of the form [masked target, context] and indices describing where on the image
        they come from, add positional embedding and pass through the transformer."""
        pos_embedding = self.pos_embedding[:, indices, :]

        x = x + pos_embedding
        for attention, ff in self.layers:
            x = x + attention(x)
            x = x + ff(x)
        return x
    
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




        