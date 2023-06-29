import torch
import einops
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from patcher import MaskedEmbedder
from saccade import SaccadeCropper
from torchvision.models import resnet50
from transformers import Transformer
from utils import ema_update
try:
    from energy_transformer import EnergyTransformer
    ET_AVAILABLE = True
except ImportError:
    ET_AVAILABLE = False
    print("EnergyTransformer not available. See the readme if you want to install it.")


class JepaSkeleton(torch.nn.Module):
    #TODO : use ABCs?
    def __init__(self,
                 embedder,
                 context_encoder,
                 predictor):
        super().__init__()

        self.embed_dim = context_encoder.dim
        self.mask_token = torch.nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.embedder = embedder

        self.context_encoder = context_encoder
        # no grad, this is updated via EMA
        self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)

        self.predictor = predictor

    def embed_get_indices(self, x):
        raise NotImplementedError("The JepaSkeleton class is abstract. Please use a subclass.")

    def encode_context_targets(self, x_patched, context, targets):
        raise NotImplementedError("The JepaSkeleton class is abstract. Please use a subclass.")
    
    def predict(self, context_encoded, x_targets, targets):
        raise NotImplementedError("The JepaSkeleton class is abstract. Please use a subclass.")

    def forward_part(self, x):
        """The forward operation for a single item in a batch."""
        x_patched, context, targets = self.embed_get_indices(x)

        context_encoded, x_targets = self.encode_context_targets(x_patched, context, targets)

        preds = self.predict(context_encoded, context, targets)
        
        return preds, x_targets, context_encoded
    
    def forward(self, x):
        """The forward operation. Uses vmap so that we can have different contexts,targets for each image in the batch."""
        preds, x_targets, context_encoded = torch.vmap(self.forward_part, randomness = "different")(x)
        return preds, x_targets, context_encoded
    
    def encode(self, x):
        x = self.embedder(x)
        x = self.target_encoder(x)
        # not sure if i should run this through the predictor - paper suggests no and that makes sense
        return x
    
    def enable_util_norm(self, util_norm_name = "weight"):
        self.context_encoder.add_util_norm(norm_name = util_norm_name)
        self.target_encoder.add_util_norm(norm_name = util_norm_name)
    
    def save(self, path):
        had_util_norm = False
        if self.context_encoder.has_util_norm:
            self.context_encoder.remove_util_norm()
            self.target_encoder.remove_util_norm()
            had_util_norm = True
        torch.save(self.state_dict(), path)

        if had_util_norm:
            self.enable_util_norm()
    
class IJepa(JepaSkeleton):
    """An image JEPA. This is slightly more abstract than the paper implementation (can be used with models other than classic ViTs).
    This model fills in some of the unimplemented methods in the skeleton. See ViTJepa below for an implementation more akin to the paper."""
    def __init__(self,
                 embedder,
                 context_encoder,
                 predictor):
        super().__init__(embedder,
                         context_encoder,
                         predictor)
        
    def embed_get_indices(self, x):
        x_patched = self.embedder(x)
        context, targets = self.embedder.get_indices()
        return x_patched, context, targets

    def encode_context_targets(self, x_patched, context, targets):
        with torch.no_grad():
            target_encoded = self.target_encoder(x_patched)
            x_targets = [target_encoded[:, target, :] for target in targets]

        # .filtered_forward method filters the posemb to the context
        context_encoded = self.context_encoder.filtered_forward(x_patched[context, :], 
                                                                context)
        return context_encoded, x_targets
    
    def predict(self, context_encoded, context, targets):
        # need these for filtering the posemb to the right spots
        indice_pairs = [torch.cat((target, context), dim = 0) for target in targets]
        # create masks of right shape
        pred_targets = [self.mask_token.repeat(context_encoded.shape[0], target.shape[0], 1) for target in targets]
        # since shape is (batch, tokens, dim), we join at dim=1
        pred_pairs = [torch.cat((pred_target, context_encoded), dim = 1) for pred_target in pred_targets]

        preds = [self.predictor.filtered_forward(pred_pair, indice_pair) for pred_pair, indice_pair in zip(pred_pairs, indice_pairs)]
        # filter to just the predicted target
        preds = [pred[:, :target.shape[0], :] for pred, target in zip(preds, targets)]

        return preds
    
    def visualize_attention(self, x):
        x = self.embedder(x)

        #TODO : maybe need to make attention patch-level
        attentions = self.target_encoder.get_attentions(x)

        return attentions

class ViTJepa(IJepa):
    """The original IJepa model from the paper. Uses a VIT-style transformer for the context encoder.
    Essentially initializes the skeleton with VITs and the methods needed. Default is a small model
    for testing."""
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
                 transformer_dropout = 0.2,
                 transformer_activation = torch.nn.GELU,):

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        masked_embedder = MaskedEmbedder(h, w,
                                         in_channels = in_channels,
                                         patch_size = patch_size,
                                         embed_dim = embed_dim,
                                         n_targets = n_targets,
                                         context_scale_fraction_range = context_scale_fraction_range,
                                         context_aspect_ratio_range = context_aspect_ratio_range,
                                         target_scale_fraction_range = target_scale_fraction_range,
                                         target_aspect_ratio_range = target_aspect_ratio_range,)
        
        context_encoder = Transformer(dim = embed_dim,
                                     depth = transformer_depth,
                                     heads = transformer_heads,
                                     dropout = transformer_dropout,
                                     context = h * w // patch_size[0] // patch_size[1],
                                     activation = transformer_activation)

        predictor = Transformer(dim = embed_dim,
                                depth = transformer_depth,
                                heads = transformer_heads,
                                dropout = transformer_dropout,
                                context = h * w // patch_size[0] // patch_size[1],
                                activation = transformer_activation)
        
        super().__init__(masked_embedder,
                         context_encoder,
                         predictor)
        
class EnergyIJepa(IJepa):
    def __init__(self,
                 h, w,
                 in_channels = 3,
                 patch_size = 16,
                 embed_dim = 256,
                 hopfield_hidden_dim = 2048,
                 n_heads = 8,
                 n_iters_default = 6,
                 alpha = 0.1,
                 beta = None,
                 hopfield_type = "relu",
                 n_targets = 4,
                 context_scale_fraction_range = (0.85, 1),
                 context_aspect_ratio_range = (1, 1),
                 target_scale_fraction_range = (0.15, 0.25),
                 target_aspect_ratio_range = (0.75, 1.5),):
        
        assert ET_AVAILABLE, "EnergyIJepa requires the energy transformer to be installed. See readme."

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        masked_embedder = MaskedEmbedder(h, w,
                                         in_channels = in_channels,
                                         patch_size = patch_size,
                                         embed_dim = embed_dim,
                                         n_targets = n_targets,
                                         context_scale_fraction_range = context_scale_fraction_range,
                                         context_aspect_ratio_range = context_aspect_ratio_range,
                                         target_scale_fraction_range = target_scale_fraction_range,
                                         target_aspect_ratio_range = target_aspect_ratio_range,)
        
        context_encoder = EnergyTransformer(dim = embed_dim,
                                            hidden_dim = hopfield_hidden_dim,
                                            n_heads = n_heads,
                                            n_iters_default = n_iters_default,
                                            alpha = alpha,
                                            beta = beta,
                                            context = h * w // patch_size[0] // patch_size[1],
                                            hopfield_type = hopfield_type)
        
        predictor = EnergyTransformer(dim = embed_dim,
                                      hidden_dim = hopfield_hidden_dim,
                                      n_heads = n_heads,
                                      n_iters_default = n_iters_default,
                                      alpha = alpha,
                                      beta = beta,
                                      context = h * w // patch_size[0] // patch_size[1],
                                      hopfield_type = hopfield_type)
        
        super().__init__(masked_embedder,
                         context_encoder,
                         predictor)

class SaccadeJepa(torch.nn.Module):
    def __init__(self,
                 embed_dim = (64, 24),
                 model_input_size = (224, 224),
                 full_input_size = (304, 304),
                 transformer_depth = 12,
                 transformer_heads = 8,
                 transformer_dropout = 0.0,
                 transformer_activation = torch.nn.GELU,
                 ):
        super().__init__()
        self.model_input_size = model_input_size
        self.full_input_size = full_input_size
        self.embed_dim = embed_dim
        self.class_dim = embed_dim[0] * embed_dim[1]

        translation_max = (min(full_input_size) - max(model_input_size)) // 2

        self.saccade_cropper = SaccadeCropper(input_h = full_input_size[0],
                                              input_w = full_input_size[1],
                                              target_h = model_input_size[0],
                                              target_w = model_input_size[1],
                                              max_translation = translation_max)

        self.context_encoder = resnet50(num_classes = self.class_dim)
        self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)

        predictor = Transformer(dim = embed_dim[-1],
                                depth = transformer_depth,
                                heads = transformer_heads,
                                dropout = transformer_dropout,
                                context = embed_dim[0],
                                activation = transformer_activation)
        self.predictor = predictor

        # TODO: test if this is better than the opposite : feeding affine info to the predictor
        self.affine_predictor = torch.nn.Sequential(
                                                    transformer_activation(),
                                                    torch.nn.Linear(self.class_dim, 3),
                                             )

    def forward(self, x):
        x_view_1, x_view_2, affines = self.saccade_cropper(x)
        context = self.context_encoder(x_view_1)
        target = self.target_encoder(x_view_2)

        context = context.view(context.shape[0], *self.embed_dim)
        target = target.view(target.shape[0], *self.embed_dim)

        target_pred = self.predictor(context)
        affines_pred = self.affine_predictor(target_pred.view(target_pred.shape[0], -1))

        return target, target_pred, affines, affines_pred
    
if __name__ == "__main__":
    sj = SaccadeJepa()
    x = torch.randn(1, 3, 304, 304)

    target, target_pred, affines, affines_pred = sj(x)

    assert target.shape == target_pred.shape, "Target and target prediction should have the same shape."
    assert affines.shape == affines_pred.shape, "Affines and affine predictions should have the same shape."