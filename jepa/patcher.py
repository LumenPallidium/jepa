import torch
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from warnings import warn

class ImagePatcher(torch.nn.Module):
    """Convenience layer for converting image to patches and vice versa.
    
    Parameters
    ----------
    h : int
        The height of the image
    w : int
        The width of the image
    in_channels : int, optional
        The number of channels in the image, by default 3
    patch_size : int or tuple of int, optional
        The size of the patches, by default 16
    """
    def __init__(self,
                 h, w,
                 d = None,
                 in_channels = 3,
                 out_channels = None,
                 activation = torch.nn.Identity(),
                 patch_size = 16,):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size[0] * patch_size[1] * in_channels

        self.h = h
        self.w = w
        if d is None:
            d = 1
        self.d = d
        self.h_patched = h // patch_size[0]
        self.w_patched = w // patch_size[1]
        self.num_patches = self.h_patched * self.w_patched * self.d

        if out_channels is not None:
            self.activation = activation
            self.projection = torch.nn.Linear(self.patch_dim , out_channels)
            self.deprojection = torch.nn.Linear(out_channels, self.patch_dim )
        else:
            self.activation = torch.nn.Identity()
            self.projection = torch.nn.Identity()
            self.deprojection = torch.nn.Identity()

        if self.d == 1:
            self.layer = Rearrange("... c (h p1) (w p2) -> ... (h w) (p1 p2 c)",
                                                    p1 = patch_size[0],
                                                    p2 = patch_size[1])
            
            self.inverse_layer = Rearrange("... (h1 w1) (p1 p2 c) -> ... c (h1 p1) (w1 p2)",
                                                            h1 = h // patch_size[0],
                                                            w1 = w // patch_size[1],
                                                            p1 = patch_size[0],
                                                            p2 = patch_size[1],
                                                            c = in_channels)
        else:
            self.layer = Rearrange("... d (h p1) (w p2) -> ... (h w d) (p1 p2)",
                                        p1 = patch_size[0],
                                        p2 = patch_size[1])
            
            self.inverse_layer = Rearrange("... (h1 w1 d) (p1 p2) -> ... d (h1 p1) (w1 p2)",
                                                            h1 = h // patch_size[0],
                                                            w1 = w // patch_size[1],
                                                            p1 = patch_size[0],
                                                            p2 = patch_size[1],
                                                            d = self.d)
    
    def forward(self, x):
        x = self.layer(x)
        return self.activation(self.projection(x))
    
    def inverse(self, x):
        x = self.deprojection(x)
        return self.inverse_layer(x)
    
    def mask_inverse(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b (h w) -> b h w", 
                      h = self.h_patched, 
                      w = self.w_patched)
        # upsampling innacurate for bools converted to floats, so we do this
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.repeat_interleave(self.patch_size[0], dim=-1)
        x = x.repeat_interleave(self.patch_size[1], dim=-2)
        x = x.reshape(batch_size, 1, self.h, self.w)
        return x
    
class ConvPatcher(torch.nn.Module):
    def __init__(self,
                 h,
                 w = None,
                 d = None,
                 in_channels = 3,
                 out_channels = 3,
                 patch_size = 4,):
        super().__init__()
        if w is None:
            w = h
        self.h = h
        self.w = w
        if d is None:
            d = 1
        self.d = d

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.patch_size = patch_size
        self.h_patched = h // patch_size
        self.w_patched = w // patch_size
        self.num_patches = self.h_patched * self.w_patched * self.d

        self.conv = torch.nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = patch_size,
                                    stride = patch_size)
        self.upsample = torch.nn.Upsample(scale_factor = patch_size)
        self.out_conv = torch.nn.Conv2d(in_channels = out_channels,
                                        out_channels = in_channels,
                                        kernel_size = 2 * patch_size - 1,
                                        padding = "same")
        
        if self.d == 1:
            self.layer = Rearrange("... c h w -> ... (h w) c")
            self.inverse_layer = Rearrange("... (h w) c -> ... c h w",
                                           h = self.h_patched,
                                           w = self.w_patched)

        else:
            self.layer = Rearrange("(b d) c h w -> b (h w d) c",
                                   d = self.d)
            self.inverse_layer = Rearrange("b (h w d) c -> (b d) c h w",
                                           h = self.h_patched,
                                           w = self.w_patched,
                                           d = self.d)
        
    def forward(self, x):
        if self.d != 1:
            x = rearrange(x, "b d h w -> (b d) 1 h w")
        patches = self.conv(x)
        return self.layer(patches)
    
    def inverse(self, x):
        x = self.inverse_layer(x)
        x = self.upsample(x)
        x = self.out_conv(x)
        if self.d != 1:
            x = rearrange(x, "(b d) 1 h w -> b d h w", d = self.d)
        return x
    
    def mask_inverse(self, x):
        batch_size = x.shape[0]
        if self.d == 1:
            x = rearrange(x, "b (h w) -> b h w", 
                        h = self.h_patched, 
                        w = self.w_patched)
            # upsampling innacurate for bools converted to floats, so we do this
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = x.repeat_interleave(self.patch_size, dim=-1)
            x = x.repeat_interleave(self.patch_size, dim=-2)
            x = x.reshape(batch_size, 1, self.h, self.w)
        else:
            x = rearrange(x, "b (h w d) -> b d h w", 
                        h = self.h_patched, 
                        w = self.w_patched,
                        d = self.d)
            x = x.unsqueeze(-1)
            x = x.repeat_interleave(self.patch_size, dim=-1)
            x = x.repeat_interleave(self.patch_size, dim=-2)
            x = x.reshape(batch_size, self.d, self.h, self.w)
        return x
    
class HybridPatcher(torch.nn.Module):
    def __init__(self,
                 h,
                 w = None,
                 in_channels = 3,
                 out_channels = 3,
                 patch_size = 4,
                 intermediate_channels = None,
                 activaiton = torch.nn.GELU()):
        """
        This patcher balances the speed of the linear patcher with the locality bias
        and filter-like nature of the convolutional patcher.
        """
        super().__init__()
        if w is None:
            w = h
        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = out_channels // 4

        self.patch_size = patch_size
        self.h_patched = h // patch_size
        self.w_patched = w // patch_size
        self.num_patches = self.h_patched * self.w_patched

        self.linear_in = torch.nn.Linear(in_channels, intermediate_channels)
        self.linear_out = torch.nn.Linear(out_channels, intermediate_channels)

        self.in_conv = torch.nn.Conv2d(in_channels = intermediate_channels,
                                       out_channels = out_channels,
                                       kernel_size = patch_size,
                                       stride = patch_size,
                                       groups = intermediate_channels // 4)
        self.out_conv = torch.nn.Conv2d(in_channels = intermediate_channels,
                                        out_channels = in_channels,
                                        kernel_size = 2 * patch_size - 1,
                                        padding = "same")
        self.upsample = torch.nn.Upsample(scale_factor = patch_size)
        self.activation = activaiton

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.activation(self.linear_in(x))
        x = rearrange(x, "b h w c-> b c h w")
        x = self.in_conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x
    
    def inverse(self, x):
        x = self.linear_out(x)
        x = self.activation(x)
        x = rearrange(x, "b (h w) c -> b c h w", 
                      h = self.h_patched, 
                      w = self.w_patched)
        x = self.upsample(x)
        x = self.out_conv(x)
        return x
    
    def mask_inverse(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b (h w) -> b h w", 
                      h = self.h_patched, 
                      w = self.w_patched)
        # upsampling innacurate for bools converted to floats, so we do this
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.repeat_interleave(self.patch_size, dim=-1)
        x = x.repeat_interleave(self.patch_size, dim=-2)
        x = x.reshape(batch_size, 1, self.h, self.w)
        return x
    
class ImagePatcher3D(torch.nn.Module):
    """Similar to above, but 3D. Theoretically could make a general ND patcher
    using custom einops strings, but over 3D does seem excessive.
    
    Parameters
    ----------
    h : int
        The height of the image
    w : int
        The width of the image
    d : int
        The depth of the image
    in_channels : int, optional
        The number of channels in the image, by default 1
    patch_size : int or tuple of int, optional
        The size of the patches, by default 16
    """
    def __init__(self,
                 h, w, d,
                 in_channels = 1,
                 patch_size = 16,
                 hw_reduction = None):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hw_reduction = hw_reduction

        # reduce dimensionality of hw by a factor of hw_reduction
        if hw_reduction is not None:
            p1, p2 = patch_size[0], patch_size[1]
            self.layer = torch.nn.Sequential(Rearrange("... c (h p1) (w p2) d -> ... c (h w) d (p1 p2)",
                                                       p1 = p1, p2 = p2),
                                             torch.nn.Linear(p1 * p2, p1 * p2 // hw_reduction),
                                             torch.nn.GELU(),
                                             Rearrange("... c (h w) (d p3) p -> ... c (h w d) (p p3)",
                                                        h = h // p1, w = w // p2, p3 = patch_size[2])
                                             )
            self.inverse_layer = torch.nn.Sequential(Rearrange("... c (h w d) (p p3) -> ... c (h w) (d p3) p",
                                                                h = h // p1, w = w // p2, p3 = patch_size[2]),
                                                     torch.nn.Linear(p1 * p2 // hw_reduction, p1 * p2),
                                                     torch.nn.GELU(),
                                                     Rearrange("... c (h w) d (p1 p2) -> ... c (h p1) (w p2) d",
                                                               p1 = p1, p2 = p2, h = h // p1, w = w // p2))
                                                               
        else:
            self.layer = Rearrange("... c (h p1) (w p2) (d p3) -> ... (h w d) (p1 p2 p3 c)",
                                                    p1 = patch_size[0],
                                                    p2 = patch_size[1],
                                                    p3 = patch_size[2])
            
            self.inverse_layer = Rearrange("... (h1 w1 d1) (p1 p2 p3 c) -> ... c (h1 p1) (w1 p2) (d1 p3)",
                                                            h1 = h // patch_size[0],
                                                            w1 = w // patch_size[1],
                                                            d1 = d // patch_size[2],
                                                            p1 = patch_size[0],
                                                            p2 = patch_size[1],
                                                            p3 = patch_size[2],
                                                            c = in_channels)
        
    def forward(self, x):
        return self.layer(x)
    
    def inverse(self, x):
        return self.inverse_layer(x)
    
class ConvPatcher3d(torch.nn.Module):
    def __init__(self,
                 h, w, d,
                 in_channels = 1,
                 out_channels = 512,
                 n_layers = 3,
                 layer_mults = (2, 4, 4),
                 activation = torch.nn.GELU()):
        super().__init__()
        self.h = h
        self.w = w
        self.d = d

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intermediate_channels = self._get_channels(n_layers)

        if isinstance(layer_mults, int):
            layer_mults = (layer_mults, layer_mults, layer_mults)
        self.layer_mults = layer_mults

        self.n_layers = n_layers
        self.patch_size = [layer_mult ** n_layers for layer_mult in layer_mults]

        self.h_patched = h // self.patch_size[1]
        self.w_patched = w // self.patch_size[2]
        self.d_patched = d // self.patch_size[0]
        self.num_patches = self.h_patched * self.w_patched * self.d_patched

        layer_sizes = [(d // (layer_mults[0] ** i), h // (layer_mults[1] ** i), w // (layer_mults[2] ** i)) for i in range(n_layers)]

        layers_in = []
        layers_out = []

        for i in range(n_layers):
            layers_in.append(torch.nn.Sequential(torch.nn.LayerNorm(layer_sizes[i]),
                                                 torch.nn.Conv3d(in_channels = self.intermediate_channels[i],
                                                 out_channels = self.intermediate_channels[i + 1],
                                                 kernel_size = self.layer_mults,
                                                 stride = self.layer_mults)))
            layers_out.append(torch.nn.Sequential(torch.nn.LayerNorm(layer_sizes[-i - 1]),
                                                  torch.nn.Conv3d(in_channels = self.intermediate_channels[-i - 1],
                                                                  out_channels = self.intermediate_channels[-i - 2],
                                                                  kernel_size = [x + 1 for x in self.layer_mults],
                                                                  padding = "same")))
                        
        self.conv = torch.nn.ModuleList(layers_in)
        self.conv_out = torch.nn.ModuleList(layers_out)

        self.activation = activation
        self.upsample = torch.nn.Upsample(scale_factor = self.layer_mults)

        self.layer = Rearrange("b c d h w -> b (d h w) c")
        self.inverse_layer = Rearrange("b (d h w) c -> b c d h w",
                                        h = self.h_patched,
                                        w = self.w_patched,
                                        d = self.d_patched)
            
    def _get_channels(self, n_layers):
        channel_ratio = self.out_channels // self.in_channels
        scale_factor = np.log2(channel_ratio) / n_layers
        scale_per_step = int(np.exp2(scale_factor))
        channels = [self.in_channels * scale_per_step ** i for i in range(n_layers)]
        channels.append(self.out_channels)
        return channels
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv:
            x = conv(x)
            x = self.activation(x)
        return self.layer(x)
    
    def inverse(self, x):
        x = self.inverse_layer(x)
        for conv in self.conv_out:
            x = self.upsample(x)
            x = conv(x)
            x = self.activation(x)
        return x.squeeze(1)
    
    def mask_inverse(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b (d h w) -> b d h w", 
                      h = self.h_patched, 
                      w = self.w_patched,
                      d = self.d_patched)
        x = x.repeat_interleave(self.patch_size[1], dim=-1)
        x = x.repeat_interleave(self.patch_size[0], dim=-2)
        x = x.repeat_interleave(self.patch_size[2], dim=-3)
        x = x.reshape(batch_size, self.d, self.h, self.w)
        return x
        
class RectangleExtractor:
    """Given an already patched image, extracts rectangles from it. Returns
    indices of the patches composing the rectangle. Does not feel efficient.
    
    Parameters
    ----------
    h : int
        The height of the image
    w : int
        The width of the image
    scale_fraction_range : tuple of float, optional
        Range for the ratio of the extracted rectangle's width to the image's width
    aspect_ratio_range : tuple of float, optional
        Range for rectangle aspect ratios (height / width)"""
    def __init__(self,
                h, w,
                d = None,
                scale_fraction_range = (0.15, 0.2),
                aspect_ratio_range = (0.75, 1.5)):

        self.h = h
        self.w = w
        self.d = d

        self.n_patches = h * w
        if d is not None:
            self.n_patches *= d

        self.scale_fraction_range = scale_fraction_range
        self.aspect_ratio_range = aspect_ratio_range

        expected_size = scale_fraction_range[0] * min(h, w) * aspect_ratio_range[0]
        if expected_size < 1:
            warn("Expected size of rectangles is less than 1, you will likely encounter errors.\
                  Try a smaller patch size or using larger images.")

    def get_indices(self):

        low_w = int(self.w * self.scale_fraction_range[0])
        high_w = int(self.w * self.scale_fraction_range[1])
        rec_width = np.random.randint(low = max(low_w, 1),
                                      high = high_w + 1)
        
        low_h = int(rec_width * self.aspect_ratio_range[0])
        high_h = int(rec_width * self.aspect_ratio_range[1])
        rec_height = np.random.randint(low = max(low_h, 1),
                                       high = high_h + 1)

        start_index_w = np.random.randint(low = 0,
                                          high = self.w - rec_width + 1)
        start_index_h = np.random.randint(low = 0,
                                          high = self.h - rec_height + 1)
        patch_start_ = start_index_h * self.w + start_index_w
        indices = [torch.arange(patch_start_ + i * self.w, 
                                patch_start_ + i * self.w + rec_width) for i in range(rec_height)]
        indices = torch.cat(indices, dim = 0)
        if self.d is not None:
            # allow whole depth to be used
            start_index_d = np.random.randint(low = 0,
                                              high = self.d)
            if start_index_d < self.d - 2:
                end_index_d = np.random.randint(low = start_index_d + 1,        
                                                high = self.d)
            else:
                end_index_d = start_index_d
            d_range = torch.arange(start_index_d, end_index_d + 1)
            indices = torch.cat([indices + i * self.h * self.w for i in d_range], dim = 0)

        return indices
     
def difference_of_indices(indices1, *indices2):
    """Returns the indices in indices1 that are not in indices2 (which can be multiple tensors).
    
    Parameters
    ----------
    indices1 : torch.Tensor
        The indices to be returned, with chunks taken out
    indices2 : torch.Tensor
        The indices to be removed from indices1, can be multiple tensors"""
    indices2 = torch.cat(indices2, dim = 0)
    bool_mask = torch.isin(indices1, indices2, invert = True)
    return indices1[bool_mask]

class MaskedEmbedder(torch.nn.Module):
    """This class handles patching, embedding patches, and pulling out context/target
    patches of the right size"""
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
                 hw_reduction = None):
        super().__init__()

        self.h = h
        self.w = w
        self.d = d
        self.hw_reduction = hw_reduction

        if isinstance(patch_size, int):
            if d is None:
                patch_size = (patch_size, patch_size)
            else:
                patch_size = (patch_size, patch_size, patch_size)

        self.patch_size = patch_size
        self.in_dim = in_channels * np.prod(patch_size) // hw_reduction

        self.n_targets = n_targets

        if d is None:
            self.patcher = ImagePatcher(h, w, 
                                        in_channels = in_channels,
                                        patch_size = patch_size)
            rectangle_d = None
        else:
            self.patcher = ImagePatcher3D(h, w, d,
                                          in_channels = in_channels,
                                          patch_size = patch_size,
                                          hw_reduction = hw_reduction)
            rectangle_d = d // patch_size[2]

        if self.in_dim != embed_dim:    
            self.embedding = torch.nn.Linear(self.in_dim, embed_dim)
        else:
            self.embedding = torch.nn.Identity()

        self.context_extractor = RectangleExtractor(h // patch_size[0],
                                                    w // patch_size[1],
                                                    d = rectangle_d,
                                                    scale_fraction_range = context_scale_fraction_range,
                                                    aspect_ratio_range = context_aspect_ratio_range)
        self.target_extractor = RectangleExtractor(h // patch_size[0],
                                                   w // patch_size[1],
                                                   d = rectangle_d,
                                                   scale_fraction_range = target_scale_fraction_range,
                                                   aspect_ratio_range = target_aspect_ratio_range)
    
    def get_indices(self):
        context = self.context_extractor.get_indices()
        targets = [self.target_extractor.get_indices() for _ in range(self.n_targets)]

        context = difference_of_indices(context, *targets)

        return context, targets
        
    def forward(self, x):
        x_patched = self.patcher(x)
        x_patched = self.embedding(x_patched)

        return x_patched
        
        
# TODO: method to mask out the non-context, non-target patches
if __name__ == "__main__":
        import torchvision
        
        im2tensor = torchvision.transforms.ToTensor()

        def tensor2im(x):
            return torchvision.transforms.ToPILImage()(x)

        def collate(x, im2tensor = im2tensor):
            x = [im2tensor(x_i[0]) for x_i in x]
            return torch.stack(x, dim = 0)

        flowers = torchvision.datasets.Flowers102(root = "../data/", split = "train", download = True)
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(500),
                                                    torchvision.transforms.RandomHorizontalFlip(0.5),
                                                    torchvision.transforms.ToTensor()])
        # dataloader = torch.utils.data.DataLoader(cifar, 
        #                                         batch_size = 32, 
        #                                         shuffle = True,
        #                                         collate_fn = collate)
        
        h = 500
        w = 500
        patch_size = 20
        n_targets = 4

        # testing general classes

        context_extractor = RectangleExtractor(h // patch_size, 
                                               w // patch_size, 
                                               scale_fraction_range=(0.85, 1),
                                               aspect_ratio_range=(1, 1))
        target_extractor = RectangleExtractor(h // patch_size, 
                                              w // patch_size,
                                              scale_fraction_range=(0.15, 0.25))

        patcher = ImagePatcher(h, w, patch_size = patch_size)

        test = transform(flowers[0][0])
        test_patched = patcher(test)

        context_im = test_patched.clone()

        context = context_extractor.get_indices()

        context_im[context, :] = 1

        targets = []
        for i in range(n_targets):
            targets.append(target_extractor.get_indices())
        context = difference_of_indices(context, *targets)

        
        context_im[context, :] = 0
        context_im = tensor2im(patcher.inverse(context_im))

        # testing MaskedEmbedder
        mask_embedder = MaskedEmbedder(h, w, patch_size = patch_size, n_targets = n_targets)
        x_patched = mask_embedder(test.unsqueeze(0))
        context, targets = mask_embedder.get_indices()
