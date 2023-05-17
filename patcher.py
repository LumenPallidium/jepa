import torch
import einops
from einops.layers.torch import Rearrange

class ImagePatcher(torch.nn.Module):
    """Convenience layer for converting image to patches and vice versa."""
    def __init__(self,
                 h, w,
                 patch_size = 16,):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.layer = Rearrange("... (h p1) (w p2) -> ... (h w) (p1 p2)",
                                                   p1 = patch_size[0],
                                                   p2 = patch_size[1])
        
        self.inverse_layer = Rearrange("... (h1 w1) (p1 p2) -> ... (h1 p1) (w1 p2)",
                                                           h1 = h // patch_size[0],
                                                           w1 = w // patch_size[1],
                                                           p1 = patch_size[0],
                                                           p2 = patch_size[1])
    
    def forward(self, x):
        return self.layer(x)
    
    def inverse(self, x):
        return self.inverse_layer(x)
    
class RectangleExtractor:
    """Given an already patched image, extracts rectangles from it. Returns
    indices of the patches composing the rectangle. Does not feel efficient."""
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
        rec_width = torch.randint(low = int(self.w * self.scale_fraction_range[0]),
                                    high = int(self.w * self.scale_fraction_range[1]),
                                    size = (1,)).item()
        low_h = int(rec_width * self.aspect_ratio_range[0])
        high_h = int(rec_width * self.aspect_ratio_range[1])
        if low_h == high_h:
            rec_height = low_h
        else:
            rec_height = torch.randint(low = low_h,
                                        high = high_h,
                                        size = (1,)).item()

        start_index_w = torch.randint(low = 0,
                                        high = self.w - rec_width,
                                        size = (1,)).item()
        start_index_h = torch.randint(low = 0,
                                high = self.h - rec_height,
                                size = (1,)).item()
        patch_start_ = start_index_h * self.w + start_index_w
        indices = [torch.arange(patch_start_ + i * self.w, 
                                patch_start_ + i * self.w + rec_width) for i in range(rec_height)]
        indices = torch.cat(indices, dim = 0)

        return indices
     
def difference_of_indices(indices1, *indices2):
    """Returns the indices in indices1 that are not in indices2 (which can be multiple tensors)"""
    indices2 = torch.cat(indices2, dim = 0)
    bool_mask = torch.isin(indices1, indices2, invert = True)
    return indices1[bool_mask]
        

if __name__ == "__main__":
        import torchvision
        
        im2tensor = torchvision.transforms.ToTensor()

        def tensor2im(x):
            return torchvision.transforms.ToPILImage()(x)

        def collate(x, im2tensor = im2tensor):
            x = [im2tensor(x_i[0]) for x_i in x]
            return torch.stack(x, dim = 0)

        flowers = torchvision.datasets.Flowers102(root = "data/", split = "train", download = True)
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

        context_im[:, context, :] = 1

        targets = []
        for i in range(n_targets):
            targets.append(target_extractor.get_indices())
        context = difference_of_indices(context, *targets)

        
        context_im[:, context, :] = 0
        context_im = tensor2im(patcher.inverse(context_im))
