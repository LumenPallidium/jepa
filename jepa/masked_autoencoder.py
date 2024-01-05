import torch
import numpy as np
from patcher import ConvPatcher, ImagePatcher, HybridPatcher, ConvPatcher3d
from transformer import Transformer
from copy import deepcopy

def min_scale(x):
    x = x - x.min() + 1e-6
    return x / x.sum()

def normalized_image_loss(x, x_out, loss_f):
    overall_mean = x.mean(dim = 0, keepdim = True)
    overall_std = x.std(dim = 0, keepdim = True)

    return loss_f((x - overall_mean) / overall_std,
                  (x_out - overall_mean) / overall_std)

class MaskedAutoencoder(torch.nn.Module):
    def __init__(self,
                 h,
                 w = None,
                 in_channels = 1,
                 dim = 384,
                 depth = 8,
                 patch_size = 8,
                 cls_token = True,
                 mask_prob = 0.6,
                 mask_drop = 0.1,
                 loss = torch.nn.MSELoss(),
                 final_activation = torch.nn.GELU(),
                 patcher_type = "hybrid"):
        super().__init__()

        if w is None:
            w = h
        self.h = h
        self.w = w
        self.dim = dim
        self.depth = depth
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.mask_drop = mask_drop
        self.has_cls = cls_token

        self.loss = loss

        if patcher_type == "conv":
            self.patcher = ConvPatcher(h, w = w, 
                                    patch_size = patch_size,
                                    in_channels = in_channels,
                                    out_channels = dim)
        elif patcher_type == "hybrid":
            self.patcher = HybridPatcher(h, w = w, 
                                    patch_size = patch_size,
                                    in_channels = in_channels,
                                    out_channels = dim)
        else:
            self.patcher = ImagePatcher(h, w = w, 
                                    patch_size = patch_size,
                                    in_channels = in_channels,
                                    out_channels = dim)
        self.num_patches = self.patcher.num_patches
        
        if cls_token:
            self.register_buffer("cls_token", torch.randn(1, dim))
            self.num_patches += 1

        self.transformer = Transformer(dim = dim,
                                       depth = depth,
                                       context = self.num_patches)
        self.final_activation = final_activation
        
        self.register_buffer("mask_token", torch.randn(1, dim))

    def forward(self, x, mask_indices = None, transformer_dropout = None):
        x = self.embed(x,
                       mask_indices = mask_indices,
                       transformer_dropout = transformer_dropout)
        if self.has_cls:
            x = x[:, 1:]
        x = self.patcher.inverse(x)
        x = self.final_activation(x)
        return x
    
    def project_deproject(self, x):
        x = self.patcher(x)
        x = self.patcher.inverse(x)
        return x

    def embed(self, x, mask_indices = None, transformer_dropout = None):
        x = self.patcher(x)
        if not mask_indices is None:
            x[mask_indices] = self.mask_token
        if self.has_cls:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim = 1)
        x = self.transformer(x, stop_at = transformer_dropout)
        return x
    
    def get_loss(self, x):
        cls_token_int = int(self.has_cls)
        mask_sample = torch.rand(x.shape[0], self.num_patches - cls_token_int)
        mask = (mask_sample < self.mask_prob)
        # trains better when some mask tokens aren't applied
        mask_dropped_out = mask_sample < self.mask_prob * (1 - self.mask_drop)

        x_out = self.forward(x, mask_indices = mask_dropped_out)

        # convert mask to image space
        img_mask = self.patcher.mask_inverse(mask).bool()
        img_mask = img_mask.repeat(1, self.in_channels, 1, 1)
        # compute loss only on masked pixels
        loss = normalized_image_loss(x[img_mask],
                                     x_out[img_mask],
                                     self.loss)
        return loss, x_out
    
    def update_mask_prob(self, amount = None, percent = None):
        current_prob = self.mask_prob
        if percent is not None:
            amount = current_prob * percent
        assert not amount is None, "Must provide either amount or percent"
        # mask shouldn't exceed 80% imo
        self.mask_prob = max(current_prob + amount, 0.8)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

class SelfDistillMAE(torch.nn.Module):
    def __init__(self,
                 h,
                 w = None,
                 d = None,
                 in_channels = 1,
                 dim = 384,
                 encoder_depth = 4,
                 regressor_depth = 2,
                 decoder_depth = 4,
                 patch_size = 8,
                 cls_token = False,
                 mask_prob = 0.6,
                 flash_attention = False,
                 loss = torch.nn.MSELoss(),
                 patcher_type = "conv",
                 student_temp = 0.1,
                 teacher_temp = 0.04,
                 center_alpha = 0.996,
                 downsample_3d = (2, 2, 2),
                 n_layers_3d = 3):
        super().__init__()
        if w is None:
            w = h
        self.h = h
        self.w = w
        if d is None:
            d = 1
        self.d = d
        self.dim = dim
        self.encoder_depth = encoder_depth
        self.regressor_depth = regressor_depth
        self.decoder_depth = decoder_depth
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.has_cls = cls_token
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_alpha = center_alpha

        self.loss = loss
        if patcher_type == "conv":
            self.patcher = ConvPatcher(h, w = w,
                                       d = d,
                                       patch_size = patch_size,
                                       in_channels = in_channels,
                                       out_channels = dim)
        elif patcher_type == "hybrid":
            assert d is None, "Hybrid patcher doesn't support 3D images"
            self.patcher = HybridPatcher(h, w = w, 
                                    patch_size = patch_size,
                                    in_channels = in_channels,
                                    out_channels = dim)
        elif patcher_type == "conv3d":
            assert d is not None, "Conv3d patcher requires d to be specified"
            ds_factor = [x ** n_layers_3d for x in downsample_3d]
            h_patcher = h * ds_factor[1]
            w_patcher = w * ds_factor[2]
            d_patcher = d * ds_factor[0]
            self.patcher = ConvPatcher3d(h_patcher, w = w_patcher,
                                         d = d_patcher,
                                         in_channels = in_channels,
                                         out_channels = dim,
                                         layer_mults = downsample_3d,
                                         n_layers= n_layers_3d ,)    
        
        else:
            self.patcher = ImagePatcher(h, w = w,
                                        d = d,
                                        patch_size = patch_size,
                                        in_channels = in_channels,
                                        out_channels = dim,
                                        activation = torch.nn.GELU())
        self.num_patches = self.patcher.num_patches
        
        #TODO: not actually implemented
        # if cls_token:
        #     self.register_buffer("cls_token", torch.randn(1, dim))
        #     self.num_patches += 1

        self.pos_emb = torch.nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.encoder= Transformer(dim = dim,
                                  depth = encoder_depth,
                                  positional_embedding = False,
                                  context = None,
                                  flash_attention = flash_attention)
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)

        self.regressor = Transformer(dim = dim,
                                     depth = regressor_depth,
                                     context = None,
                                     positional_embedding = False,
                                     cross = True,
                                     flash_attention = flash_attention)
        self.decoder = Transformer(dim = dim,
                                   depth = decoder_depth,
                                   context = None,
                                   positional_embedding = False,
                                   flash_attention = flash_attention)
        
        self.mask_token = torch.nn.Parameter(torch.randn(1, dim))
        self.register_buffer("center", torch.zeros(1, 1, dim))

    def tokenize(self, x):
        x = self.patcher(x)
        #TODO : implement cls token
        #if self.has_cls:
        #    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #    x = torch.cat((cls_tokens, x), dim = 1)
        return x
    
    def project_deproject(self, x):
        x = self.patcher(x)
        x = self.patcher.inverse(x)
        return x
    
    def embed(self, x, type = "ema"):
        if type == "ema":
            return self.encoder_ema(self.tokenize(x),
                                    pos_embedding = self.pos_emb)
        elif type == "normal":
            return self.encoder(self.tokenize(x),
                                pos_embedding = self.pos_emb)
        else:
            raise ValueError(f"Unknown type {type}")
        
    def decode(self, x):
        x = self.decoder(x)
        return self.patcher.inverse(x)
    
    def mask_strategy(self, random = False):
        roll = np.random.rand()
        h_patches = self.patcher.h_patched
        w_patches = self.patcher.w_patched
        d = self.d
        # random patches
        if (not random) or roll < 0.5:
            mask_sample = torch.rand(self.num_patches)
        # 2x2 block patches
        elif roll < 0.8:
            mask_sample = torch.rand(d, h_patches // 2, w_patches // 2)
            mask_sample = torch.nn.functional.interpolate(mask_sample.unsqueeze(0),
                                                          size = (h_patches, w_patches),
                                                          mode = "nearest").squeeze()
            mask_sample = mask_sample.flatten()
        # 4x4 block patches
        else:
            mask_sample = torch.rand(d, h_patches // 4, w_patches // 4)
            mask_sample = torch.nn.functional.interpolate(mask_sample.unsqueeze(0),
                                                          size = (h_patches, w_patches),
                                                          mode = "nearest").squeeze()
            mask_sample = mask_sample.flatten()
        # trains better when some mask tokens aren't applied
        mask = mask_sample < self.mask_prob
        if mask.sum() == 0:
            # at least one mask token
            mask[0] = True
        elif mask.sum() == mask.shape[0]:
            # at least one non-mask token
            mask[0] = False
        mask_inverse = ~mask
        return mask, mask_inverse

    #TODO: look into not repeating same mask for batch
    def get_loss(self, x, rep_loss_weight = 2):
        batch_size = x.shape[0]
        x_toks = self.tokenize(x)
        #cls_token_int = int(self.has_cls)
        mask, mask_inverse = self.mask_strategy()

        pos_emb_mask = self.pos_emb.repeat(batch_size, 1, 1)[:, mask, :].clone()
        pos_emb_inverse = self.pos_emb.repeat(batch_size, 1, 1)[:, mask_inverse, :].clone()

        encoding = self.encoder(x_toks[:, mask_inverse, :], 
                                pos_embedding = pos_emb_inverse)
        with torch.no_grad():
            encoding_ema = self.encoder_ema(x_toks[:, mask, :],
                                            pos_embedding = pos_emb_mask)
        
        masked_query = self.mask_token.expand_as(encoding_ema).clone()
        
        encoding_ema_hat = self.regressor(masked_query,
                                          y = encoding,
                                          pos_embedding = pos_emb_mask)
        
        rep_loss = torch.nn.functional.cross_entropy(((encoding_ema - self.center) / self.teacher_temp).softmax(dim = -1),
                                                     (encoding_ema_hat / self.student_temp).softmax(dim = -1))
        rep_loss *= rep_loss_weight

        self.center = self.center * self.center_alpha + encoding_ema.mean(dim = (0, 1), keepdim = True) * (1 - self.center_alpha)

        last_embedding = self.decoder(encoding_ema_hat,
                                      pos_embedding = pos_emb_mask)
        # detach here so that encoder only gets grad through regressor
        with torch.no_grad():
            encoder_decode = self.decoder(encoding,
                                          pos_embedding = pos_emb_inverse)
        
        # reconstruct image
        full_embedding = torch.zeros(batch_size, self.num_patches, self.dim,
                                     device = x.device)
        full_embedding[:, mask, :] = last_embedding
        full_embedding[:, mask_inverse, :] = encoder_decode
        x_out = self.patcher.inverse(full_embedding)
        # convert mask to image space
        img_mask = self.patcher.mask_inverse(mask.repeat(batch_size, 1)).bool()
        img_mask = img_mask.repeat(1, self.in_channels, 1, 1)
        # compute loss only on masked pixels
        loss = normalized_image_loss(x[img_mask],
                                     x_out[img_mask],
                                     self.loss)
        return loss + rep_loss, x_out
    
    def update_mask_prob(self, amount = None, percent = None):
        current_prob = self.mask_prob
        if percent is not None:
            amount = current_prob * percent
        assert not amount is None, "Must provide either amount or percent"
        # mask shouldn't exceed 80% imo
        self.mask_prob = max(current_prob + amount, 0.8)

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == "__main__":
    import torchvision
    import os
    from tqdm import tqdm
    from einops import rearrange
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import ema_update

    im2tensor = torchvision.transforms.ToTensor()

    def collate(x, im2tensor = im2tensor):
        x = [im2tensor(x_i[0]) for x_i in x]
        return torch.stack(x, dim = 0)

    def tensor2im(x):
        return torchvision.transforms.ToPILImage()(x)

    def save_im(x, path):
        tensor2im(x).save(path)

    # making a tmp folder to store the images
    os.makedirs("tmp/", exist_ok = True)

    # data root
    data_root = "D:/Projects/" # you should only need to change this
    test_mae = False
    # image export frequency
    output_every = 100

    # data and patcher params
    h, w = 32, 32
    in_channels = 3
    patch_size = 1
    patch_dim = patch_size**2 * 3
    embed_dim = patch_dim * 2
    depth = 6
    n_patches = (h // patch_size) * (w // patch_size)
    batch_size = 256
    mask_prob = 0.6
    flash_attention = False

    if test_mae:
        model = MaskedAutoencoder(h,
                                  w = w,
                                  in_channels = in_channels,
                                  depth = depth,
                                  dim = embed_dim,
                                  patch_size = patch_size,
                                  mask_prob = mask_prob).to(device)
    else:
        model = SelfDistillMAE(h,
                               w = w,
                               in_channels = in_channels,
                               encoder_depth = depth,
                               dim = embed_dim,
                               patch_size = patch_size,
                               mask_prob = mask_prob,
                               flash_attention = flash_attention).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-3,
                                 weight_decay = 1e-6)
    
    # training params
    n_epochs = 5
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar = torchvision.datasets.CIFAR100(root = data_root, train = True, download = True)
       
    losses = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        dataloader = torch.utils.data.DataLoader(cifar, 
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            collate_fn = collate)
        epoch_losses = []
        for i, x in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            loss, x_out = model.get_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if not test_mae:
                model.encoder_ema = ema_update(model.encoder_ema,
                                            model.encoder)

            epoch_losses.append(loss.item())
            
            if i % output_every == 0:
                str_i = str(i).zfill(5)
                b, c, h, w = x.shape

                x_out = torch.stack([x, x_out], dim = 0)
                x_out = rearrange(x_out, "n b c h w -> (b n) c h w")

                x_out = torchvision.utils.make_grid(x_out, nrow = 8, padding = 2, pad_value = 1)
                save_im(x_out, f"tmp/epoch_{epoch}_{str_i}.png")
        losses.extend(epoch_losses)
        print(f"Loss: {np.mean(epoch_losses)}")
    plt.plot(losses)