import torch
from copy import deepcopy
from patcher import MaskedEmbedder
from transformers import Transformer

class IJepa(torch.nn.Module):
    def __init__(self,
                 h, w,
                 in_channels = 3,
                 patch_size = 16,
                 embed_dim = 256,
                 predictor_dim = 256,
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
        self.target_encoder = deepcopy(self.context_encoder).requires_grad_(False)

        self.predictor = Transformer(predictor_dim,
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
        context_encoded = self.context_encoder(x_context)[:, context, :]

        # since shape is (batch, context, dim), we join at dim=1
        indice_pairs = [torch.cat((target, context), dim = 0) for target in targets]
        pred_targets = [self.masked_embedder.mask_token.repeat(context_encoded.shape[0], target.shape[0], 1) for target in targets]
        pred_pairs = [torch.cat((pred_target, context_encoded), dim = 1) for pred_target in pred_targets]

        preds = [self.predictor.predict(pred_pair, indice_pair) for pred_pair, indice_pair in zip(pred_pairs, indice_pairs)]
        preds = [pred[:, :target.shape[0], :] for pred, target in zip(preds, targets)]

        self.target_encoder.ema_update(self.context_encoder)
        return preds, x_targets


if __name__ == "__main__":
    import torchvision
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    im2tensor = torchvision.transforms.ToTensor()

    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)

    def collate(x):
        x = [x_i[0] for x_i in x]
        return torch.stack(x, dim = 0)
    
    h = 500
    w = 500
    patch_size = 20
    n_targets = 4

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((h, w)),
                                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                                torchvision.transforms.ToTensor()])

    flowers = torchvision.datasets.Flowers102(root = "data/", 
                                              split = "train", 
                                              download = True,
                                              transform = transform)

    dataloader = torch.utils.data.DataLoader(flowers, 
                                             batch_size = 8, 
                                             shuffle = True,
                                             collate_fn = collate)
    
    model = IJepa(h, w, patch_size = patch_size, n_targets = n_targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4, amsgrad = True)

    losses = []

    for i, x in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        preds, x_targets = model(x)

        loss = 0
        for pred, x_target in zip(preds, x_targets):
            loss += torch.nn.functional.mse_loss(pred, x_target)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i == 100:
            break

    plt.plot(losses)
    
