import torch
import einops
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from patcher import MaskedEmbedder
from saccade import SaccadeCropper
from torchvision.models import efficientnet_v2_l, resnet50, inception_v3, convnext_tiny
from transformer import Transformer
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
        return x
    
    def embed(self, x):
        """Slightly misleading name, but for backward compatiblity with other models.
        Encodes with the context encoder.
        """
        x = self.embedder(x)
        x = self.context_encoder(x)
        return x
    
    def enable_util_norm(self, util_norm_name = "weight"):
        self.context_encoder.add_util_norm(norm_name = util_norm_name)
        self.target_encoder.add_util_norm(norm_name = util_norm_name)
        self.predictor.add_util_norm(norm_name = util_norm_name)
    
    def save(self, path):
        had_util_norm = False
        if self.context_encoder.has_util_norm:
            self.context_encoder.remove_util_norm()
            self.target_encoder.remove_util_norm()
            self.predictor.remove_util_norm()
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
        context_encoded = self.context_encoder.filtered_forward(x_patched[:, context, :], 
                                                                context)
        return context_encoded, x_targets
    
    def predict(self, context_encoded, context, targets):
        # need these for filtering the posemb to the right spots
        indice_pairs = [torch.cat((target, context), dim = 0) for target in targets]
        # create masks of right shape
        pred_targets = [self.mask_token.repeat(context_encoded.shape[0], target.shape[0], 1) for target in targets]
        # since shape is (batch, tokens, dim), we join at dim=1, i.e. token concatenation
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
                 d = None,
                 in_channels = 3,
                 patch_size = 16,
                 embed_dim = 256,
                 n_targets = 4,
                 context_scale_fraction_range = (0.85, 1),
                 context_aspect_ratio_range = (1, 1),
                 target_scale_fraction_range = (0.15, 0.25),
                 target_aspect_ratio_range = (0.75, 1.5),
                 hw_reduction = None,
                 transformer_depth = 6,
                 transformer_heads = 8,
                 transformer_dropout = 0.2,
                 transformer_activation = torch.nn.GELU,
                 cls_token = True,):

        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size] if d is None else [patch_size, patch_size, patch_size]
        if d is None:
            context_size = h * w // patch_size[0] // patch_size[1]
        else:
            context_size = h * w * d // patch_size[0] // patch_size[1] // patch_size[2]


        masked_embedder = MaskedEmbedder(h, w,
                                         d = d,
                                         in_channels = in_channels,
                                         patch_size = patch_size,
                                         embed_dim = embed_dim,
                                         n_targets = n_targets,
                                         hw_reduction=hw_reduction,
                                         context_scale_fraction_range = context_scale_fraction_range,
                                         context_aspect_ratio_range = context_aspect_ratio_range,
                                         target_scale_fraction_range = target_scale_fraction_range,
                                         target_aspect_ratio_range = target_aspect_ratio_range,)
        
        context_encoder = Transformer(dim = embed_dim,
                                      depth = transformer_depth,
                                      heads = transformer_heads,
                                      dropout = transformer_dropout,
                                      context = context_size,
                                      activation = transformer_activation)

        predictor = Transformer(dim = embed_dim,
                                depth = transformer_depth,
                                heads = transformer_heads,
                                dropout = transformer_dropout,
                                context = context_size,
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
                 model_input_size = (128, 128),
                 full_input_size = (200, 200),
                 in_channels = 3,
                 expected_in_channels = 3,
                 predictor_depth = 3,
                 predictor_activation = torch.nn.GELU,
                 shift_percent = 0.9,
                 predict_affines = False,
                 transformer_predictor = False,
                 transformer_heads = 8,
                 transformer_dropout = 0.0,
                 use_cycle_consistency = True,
                 model = None
                 ):
        super().__init__()
        self.model_input_size = model_input_size
        self.full_input_size = full_input_size
        self.embed_dim = embed_dim
        self.class_dim = embed_dim[0] * embed_dim[1]
        self.transformer_predictor = transformer_predictor
        self.use_cycle_consistency = use_cycle_consistency

        if expected_in_channels != in_channels:
            self.input_transform = torch.nn.Conv2d(in_channels, 
                                                   expected_in_channels, 
                                                   1)
        else:
            self.input_transform = torch.nn.Identity()

        translation_max = int((min(full_input_size) - max(model_input_size)) * shift_percent)

        self.saccade_cropper = SaccadeCropper(input_h = full_input_size[0],
                                              input_w = full_input_size[1],
                                              target_h = model_input_size[0],
                                              target_w = model_input_size[1],
                                              max_translation = translation_max)
        
        if model is None:
            self.context_encoder = convnext_tiny(num_classes = self.class_dim)
            self.context_encoder.dim = self.class_dim
            self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)
        else:
            self.context_encoder = model(num_classes = self.class_dim)
            self.context_encoder.dim = self.class_dim
            self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)


        if transformer_predictor:
            predictor = Transformer(dim = embed_dim[-1],
                                    depth = predictor_depth,
                                    heads = transformer_heads,
                                    dropout = transformer_dropout,
                                    context = embed_dim[0],
                                    activation = predictor_activation)
        else:
            predictor = [torch.nn.Linear(self.class_dim, self.class_dim), predictor_activation()] * (predictor_depth - 1)
            predictor = torch.nn.Sequential(*predictor, torch.nn.Linear(self.class_dim, self.class_dim))
        self.predictor = predictor


        self.predict_affines = predict_affines
        assert not (self.predict_affines and self.use_cycle_consistency), "predict_affines and use_cycle_consistency can not both be True."
        if predict_affines:
            self.affine_predictor = torch.nn.Sequential(
                                                        predictor_activation(),
                                                        torch.nn.Linear(self.class_dim,
                                                                        self.saccade_cropper.affine_embed_dim),
                                                )
        else:
            self.affine_embedder = torch.nn.Sequential(
                                                        predictor_activation(),
                                                        torch.nn.Linear(self.saccade_cropper.affine_embed_dim, 
                                                                        self.class_dim),
                                                )
        

    def forward(self, x):
        x = self.input_transform(x)
        x_view_1, x_view_2, affines = self.saccade_cropper(x)

        # randomly swap views
        if np.random.rand() > 0.5:
            x_view_1, x_view_2 = x_view_2, x_view_1
            affines = -affines

        context = self.context_encoder(x_view_1)
        with torch.no_grad():
            target = self.target_encoder(x_view_2)

        if type(self.context_encoder).__name__ == "Inception3":
            context = context[0]
            target = target[0]
            
        context_copy = context.clone()

        # add affines as a positional encoding
        if not self.predict_affines:
            affines_emb = self.affine_embedder(affines)
            context += affines_emb

        if self.transformer_predictor:
            context = context.view(context.shape[0], *self.embed_dim)
        target_pred = self.predictor(context)

        # undo the predictor shaping
        if self.transformer_predictor:
            target_pred = target_pred.view(target_pred.shape[0], -1)

        if self.use_cycle_consistency:
            # the inverse affine embedding, note that since a mix of sins and cos are used, we can't just take the negative of affines
            affines_minus = self.affine_embedder(-affines)
            cycled_context = self.predictor(target_pred + affines_minus)
            # ensure the network properly "undoes" itself
            cycle_loss = torch.nn.functional.mse_loss(cycled_context, context_copy)
        else:
            cycle_loss = 0

        if self.predict_affines:
            affines_pred = self.affine_predictor(target_pred.view(target_pred.shape[0], -1))
            return target, target_pred, affines, affines_pred, cycle_loss

        return target, context_copy, target_pred, cycle_loss
    
    def encode(self, x):
        return self.target_encoder(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def enable_util_norm(self):
        raise NotImplementedError
    
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import os
    from einops import rearrange

    im2tensor = torchvision.transforms.ToTensor()

    def collate(x, im2tensor = im2tensor):
        y = [x_i[1] for x_i in x]
        x = [im2tensor(x_i[0]) for x_i in x]
        return torch.stack(x, dim = 0), y

    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    # data root
    data_root = "D:/Projects/" # you should only need to change this
    # image export frequency
    output_every = 100

    # data and patcher params
    h, w = 32, 32
    in_channels = 3
    patch_size = 2
    patch_dim = patch_size**2 * 3
    embed_dim = patch_dim * 2
    depth = 4
    n_patches = (h // patch_size) * (w // patch_size)
    batch_size = 64
    flash_attention = False
    test_saccade = False

    if test_saccade:
        sj = SaccadeJepa()
        x = torch.randn(1, 3, 304, 304)

        target, target_pred, affines, affines_pred = sj(x)

        assert target.shape == target_pred.shape, "Target and target prediction should have the same shape."
        assert affines.shape == affines_pred.shape, "Affines and affine predictions should have the same shape."
    else:
        from sklearn.neighbors import KNeighborsClassifier

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ViTJepa(h, w, 
                        patch_size = patch_size, 
                        embed_dim = embed_dim,
                        transformer_depth = depth).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = 1e-3,
                                     weight_decay = 1e-6)
        
        # training params
        n_epochs = 10
        lr = 1e-3

        cifar = torchvision.datasets.CIFAR100(root = data_root, train = True, download = True)
        
        losses = []
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")
            dataloader = torch.utils.data.DataLoader(cifar, 
                                                     batch_size = batch_size, 
                                                     shuffle = True,
                                                     collate_fn = collate)
            ys = []
            embeddings = []
            epoch_losses = []
            for i, (x, y) in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                ys.extend(y)
                x = x.to(device)
                preds, x_targets, context_encoded = model(x)

                loss = 0
                for pred, target in zip(preds, x_targets):
                    loss += torch.nn.functional.mse_loss(pred, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                embeddings.extend([c.mean(dim = (0, 1)).detach().cpu().numpy() for c in context_encoded])
                epoch_losses.append(loss.item())
                
            # run knn on the embeddings
            X = np.stack(embeddings)
            ys = np.array(ys)
            knn = KNeighborsClassifier(n_neighbors = 4)
            knn.fit(X, ys)
            print(f"KNN score: {knn.score(embeddings, ys)}")
            print(f"Epoch loss: {np.mean(epoch_losses)}")
            losses.extend(epoch_losses)

