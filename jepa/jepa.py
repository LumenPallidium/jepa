import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from patcher import MaskedEmbedder
from transformers import Transformer

class IJepa(torch.nn.Module):
    def __init__(self,
                 h, w,
                 in_channels = 3,
                 patch_size = 16,
                 embed_dim = 256,
                 n_targets = 4,
                 context_scale_fraction_range = (0.85, 1),
                 context_aspect_ratio_range = (1, 1),
                 target_scale_fraction_range = (0.15, 0.25),
                 target_aspect_ratio_range = (0.75, 1.5),
                 transformer_depth = 6,
                 transformer_heads = 8,
                 transformer_head_dim = 64,
                 transformer_dropout = 0.2,
                 transformer_activation = torch.nn.GELU,):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.masked_embedder = MaskedEmbedder(h, w,
                                              in_channels = in_channels,
                                              patch_size = patch_size,
                                              embed_dim = embed_dim,
                                              n_targets = n_targets,
                                              context_scale_fraction_range = context_scale_fraction_range,
                                              context_aspect_ratio_range = context_aspect_ratio_range,
                                              target_scale_fraction_range = target_scale_fraction_range,
                                              target_aspect_ratio_range = target_aspect_ratio_range,)
        
        self.context_encoder = Transformer(embed_dim,
                                           transformer_depth,
                                           heads = transformer_heads,
                                           head_dim = transformer_head_dim,
                                           dropout = transformer_dropout,
                                           context = h * w // patch_size[0] // patch_size[1],
                                           activation = transformer_activation)
        # no grad, this is updated via EMA
        self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)

        self.predictor = Transformer(embed_dim,
                                     transformer_depth,
                                     heads = transformer_heads,
                                     head_dim = transformer_head_dim,
                                     dropout = transformer_dropout,
                                     context = h * w // patch_size[0] // patch_size[1],
                                     activation = transformer_activation)
        
    def forward(self, x):
        x_patched = self.masked_embedder(x)
        # using single indices for each batch - consistency is easier
        context, targets = self.masked_embedder.get_indices()

        target_encoded = self.target_encoder(x_patched)
        x_targets = [target_encoded[:, target, :] for target in targets]

        x_context = x_patched.clone()
        x_context[:, ~context, :] = self.masked_embedder.mask_token
        # note I filter to just the context patch after running through the encoder - seems consistent with paper figures
        context_encoded = self.context_encoder(x_context)[:, context, :]

        # need these for filtering the posemb to the right spots
        indice_pairs = [torch.cat((target, context), dim = 0) for target in targets]
        # create masks of right shape
        pred_targets = [self.masked_embedder.mask_token.repeat(context_encoded.shape[0], target.shape[0], 1) for target in targets]
        # since shape is (batch, tokens, dim), we join at dim=1
        pred_pairs = [torch.cat((pred_target, context_encoded), dim = 1) for pred_target in pred_targets]

        preds = [self.predictor.predict(pred_pair, indice_pair) for pred_pair, indice_pair in zip(pred_pairs, indice_pairs)]
        # filter to just the predicted target
        preds = [pred[:, :target.shape[0], :] for pred, target in zip(preds, targets)]

        self.target_encoder.ema_update(self.context_encoder)
        return preds, x_targets
    
    def encode(self, x):
        x = self.masked_embedder(x)
        x = self.target_encoder(x)
        # not sure if i should run this through the predictor - paper suggests no and that makes sense
        return x

#TODO : energy transformers!!
    