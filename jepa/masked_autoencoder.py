import torch
from patcher import ConvPatcher
from transformer import Transformer

class MaskedAutoencoder(torch.nn.Module):
    def __init__(self,
                 h,
                 w = None,
                 in_channels = 1,
                 dim = 512,
                 depth = 6,
                 patch_size = 4,
                 cls_token = True,
                 mask_prob = 0.4,
                 mask_drop = 0.1,
                 loss = torch.nn.MSELoss()):
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
        self.patcher = ConvPatcher(h, w = w, 
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
        
        self.register_buffer("mask_token", torch.randn(1, dim))

    def forward(self, x, mask_indices = None):
        x = self.embed(x, mask_indices = mask_indices)
        if self.has_cls:
            x = x[:, 1:]
        x = self.patcher.inverse(x)
        return x
    
    def embed(self, x, mask_indices = None):
        x = self.patcher(x)
        if not mask_indices is None:
            x[mask_indices] = self.mask_token
        if self.has_cls:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim = 1)
        x = self.transformer(x)
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
        # compute loss only on masked pixels
        loss = self.loss(x_out[img_mask], x[img_mask])
        return loss
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoder(128,
                              128,
                              patch_size = 16).to(device)
    
    x = torch.randn(1, 1, 128, 128).to(device)

    print(model(x).shape)
    print(model.get_loss(x))