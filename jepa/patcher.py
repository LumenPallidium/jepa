import torch
import numpy as np
from einops.layers.torch import Rearrange

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
                 in_channels = 3,
                 patch_size = 16,):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.layer = Rearrange("... c (h p1) (w p2) -> ... (h w) (p1 p2 c)",
                                                   p1 = patch_size[0],
                                                   p2 = patch_size[1])
        
        self.inverse_layer = Rearrange("... (h1 w1) (p1 p2 c) -> ... c (h1 p1) (w1 p2)",
                                                           h1 = h // patch_size[0],
                                                           w1 = w // patch_size[1],
                                                           p1 = patch_size[0],
                                                           p2 = patch_size[1],
                                                           c = in_channels)
    
    def forward(self, x):
        return self.layer(x)
    
    def inverse(self, x):
        return self.inverse_layer(x)
    
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
                 patch_size = 16,):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels

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
    
# TODO : make this more than 2d, probably need a for loop
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
                scale_fraction_range = (0.15, 0.2),
                aspect_ratio_range = (0.75, 1.5)):

        self.h = h
        self.w = w

        self.n_patches = h * w

        self.scale_fraction_range = scale_fraction_range
        self.aspect_ratio_range = aspect_ratio_range

    def get_indices(self):

        low_w = int(self.w * self.scale_fraction_range[0])
        high_w = int(self.w * self.scale_fraction_range[1])
        rec_width = np.random.randint(low = low_w,
                                      high = high_w + 1)
        
        # TODO: this might be an issue - eg 0.2 * 1.5 = 0.3, 0.75 * 0.15 ~ 0.11
        low_h = int(rec_width * self.aspect_ratio_range[0])
        high_h = int(rec_width * self.aspect_ratio_range[1])
        rec_height = np.random.randint(low = low_h,
                                       high = high_h + 1)

        start_index_w = np.random.randint(low = 0,
                                          high = self.w - rec_width + 1)
        start_index_h = np.random.randint(low = 0,
                                          high = self.h - rec_height + 1)
        patch_start_ = start_index_h * self.w + start_index_w
        indices = [torch.arange(patch_start_ + i * self.w, 
                                patch_start_ + i * self.w + rec_width) for i in range(rec_height)]
        indices = torch.cat(indices, dim = 0)

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
                 in_channels = 3,
                 patch_size = 16,
                 embed_dim = 256,
                 n_targets = 4,
                 context_scale_fraction_range = (0.85, 1),
                 context_aspect_ratio_range = (1, 1),
                 target_scale_fraction_range = (0.15, 0.25),
                 target_aspect_ratio_range = (0.75, 1.5),):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_dim = patch_size[0] * patch_size[1] * in_channels

        self.n_targets = n_targets
        self.mask_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

        self.patcher = ImagePatcher(h, w, 
                                    in_channels = in_channels,
                                    patch_size = patch_size)
        self.embedding = torch.nn.Linear(self.in_dim, embed_dim)

        self.context_extractor = RectangleExtractor(h // patch_size[0],
                                                    w // patch_size[1],
                                                    scale_fraction_range = context_scale_fraction_range,
                                                    aspect_ratio_range = context_aspect_ratio_range)
        self.target_extractor = RectangleExtractor(h // patch_size[0],
                                                    w // patch_size[1],
                                                    scale_fraction_range = target_scale_fraction_range,
                                                    aspect_ratio_range = target_aspect_ratio_range)
    def mask_indices(self, x, indices):
        x_masked = x.clone()
        x_masked[indices, :] = self.mask_token
        return x_masked
    
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
