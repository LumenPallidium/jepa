import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from patcher import MaskedEmbedder
from transformers import Transformer
from utils import WarmUpScheduler

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

def validation_test(model, 
                    dataset,
                    device,
                    n_categories = 101,
                    probe_lr = 1e-3,
                    probe_weight_decay = 1e-4,
                    val_epochs = 1,):
    """Tests the model at a point in its training. Freezes the weights,
    stops the gradients, and trains a linear probe on the dataset"""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle = True)
    model.eval()
    linear_probe = torch.nn.Linear(model.target_encoder.dim, n_categories).to(device)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr = probe_lr, weight_decay = probe_weight_decay, amsgrad = True)

    for epoch in range(val_epochs):
        epoch_scores = []
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = model.encode(x)
            
            optimizer.zero_grad()
            logits = linear_probe(x).mean(dim = 1)
            loss = torch.nn.functional.cross_entropy(logits, y)

            top1_acc = (logits.argmax(dim = 1) == y).float().mean()
            epoch_scores.append(top1_acc.item())

            loss.backward()
            optimizer.step()
        print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")
    
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    im2tensor = torchvision.transforms.ToTensor()

    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)

    def collate(x):
        x = [x_i[0] for x_i in x]
        return torch.stack(x, dim = 0)
    
    h = 224
    w = 224
    patch_size = 16
    n_targets = 4
    n_epochs = 10
    val_every = 5
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((h, w)),
                                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                                torchvision.transforms.ToTensor()])

    food = torchvision.datasets.Food101(root = "C:/Projects/image_datasets/", 
                                              split = "train", 
                                              download = True,
                                              transform = transform)
    food_val = torchvision.datasets.Food101(root = "C:/Projects/image_datasets/",
                                                  split = "test",
                                                  download = True,
                                                  transform = transform)


    
    model = IJepa(h, w, patch_size = patch_size, n_targets = n_targets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4, amsgrad = True)

    scheduler = WarmUpScheduler(optimizer = optimizer,
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
                                warmup_iter = 128 * 15,
                                total_iter = 128 * n_epochs) # 15 epochs of warmup

    losses = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        if epoch % val_every == 0:
           validation_test(model, food_val, device)

        dataloader = torch.utils.data.DataLoader(food, 
                                                batch_size = batch_size, 
                                                shuffle = True,
                                                collate_fn = collate)
        for i, x in tqdm(enumerate(dataloader)):
            x = x.to(device)
            optimizer.zero_grad()
            preds, x_targets = model(x)

            loss = 0
            for pred, x_target in zip(preds, x_targets):
                loss += torch.nn.functional.mse_loss(pred, x_target)

            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f"\t\tDone. Final loss: {loss.item()}")

    model.train()
    plt.plot(losses)
    
