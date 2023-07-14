import torch
import torchvision

def rotate_and_translate_batch(images, translation, theta, interp_mode = "nearest"):
    """Custom rotation operation that works on batches of images with batches of angles.
    Assumes theta has shape (batch_size, 1) and translation has shape (batch_size, 2)."""
    # unsqueeze shapes for generating affines
    translation = translation.unsqueeze(-1)
    theta = theta.unsqueeze(-1)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot_mat = torch.cat([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
    rot_mat = rot_mat.view(-1, 2, 2)
    
    # concat rotation with translation
    rot_mat = torch.cat([rot_mat, translation], dim = -1)
    
    grid = torch.nn.functional.affine_grid(rot_mat, images.size(), align_corners = False)
    return torch.nn.functional.grid_sample(images, grid, mode = interp_mode, align_corners = False)

class SaccadeCropper(torch.nn.Module):
    def __init__(self,
                 input_h,
                 input_w,
                 target_h = 224,
                 target_w = 224,
                 max_translation = 40,
                 max_rotation = 2,
                 interp_mode = "nearest",
                 default_device = "cuda" if torch.cuda.is_available() else "cpu",
                 embed_affines = True,
                 translation_L = 4,
                 rotation_L = 10
                 ):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.target_h = target_h
        self.target_w = target_w

        self.max_translation = max_translation
        self.max_translation_frac = max_translation / input_h

        self.max_rotation = max_rotation
        self.max_rotation_rad = max_rotation * torch.pi / 180
        self.interp_mode = interp_mode

        # coefficients for the NeRF-like encoder, see method
        self.coeff = {}
        self.embed_affines = embed_affines
        self.translation_L = translation_L
        self.rotation_L = rotation_L
        # since there are two translation dims and one rotation dim
        self.affine_embed_dim = translation_L * 4 + rotation_L * 2

        self.default_device = default_device
        self.requires_grad_(False)


    def forward(self, x):
        with torch.no_grad():
            b, c, h, w = x.shape
            # generate rotations of shape (b, 1)
            rotations = torch.empty(b, 1, device = self.default_device).uniform_(-self.max_rotation_rad, self.max_rotation_rad)
            translations = torch.empty(b, 2, device = self.default_device).uniform_(-self.max_translation_frac, self.max_translation_frac)

            # apply the roations
            x_affined = rotate_and_translate_batch(x, 
                                                   translations, 
                                                   rotations, 
                                                   interp_mode = self.interp_mode)
            if self.embed_affines:
                rotations = self.nerf_like_encoder(rotations, L = self.rotation_L)
                translations = self.nerf_like_encoder(translations, L = self.translation_L)
            transform_values = torch.cat([rotations, translations], dim = -1)

            # generate two views via center crop
            x1 = torchvision.transforms.functional.center_crop(x_affined, (self.target_h, self.target_w))
            x2 = torchvision.transforms.functional.center_crop(x, (self.target_h, self.target_w))

        return x1, x2, transform_values
    
    def nerf_like_encoder(self, in_tensor, L = 10):
        """This embeds the translation or rotation values into a sinuisoidal space.
        This is based on the NeRF paper, where the translation and rotation values are embedded
        in a higher-dimensional sinuisoidal space, which improves performance.
        https://arxiv.org/pdf/2003.08934.pdf

        Parameters
        ----------
        in_tensor : torch.Tensor
            Tensor of shape (batch_size, n) containing the translation or rotation values.

        L : int
            Half the number of dimensions to use for the embedding.
        """
        # array of the coefficients for the sinuisoidal embedding
        batch_size = in_tensor.shape[0]

        # compute coefficient only once since this gets called multiple times
        if L in self.coeff:
            coeff = self.coeff[L].to(in_tensor.device)
        else:
            coeff = (torch.pi * (2 ** torch.arange(L, dtype = torch.float32))).unsqueeze(0).unsqueeze(0)
            self.coeff[L] = coeff

            coeff = coeff.to(in_tensor.device)


        embedding = torch.concat([torch.sin(in_tensor.unsqueeze(-1) * coeff),
                                torch.cos(in_tensor.unsqueeze(-1) * coeff)],
                                dim = -1)

        return embedding.view(batch_size, -1)




if __name__ == "__main__":
    from PIL import Image
    sc = SaccadeCropper(304, 304)
    x = torch.rand(64, 3, 304, 304)

    y1, y2, affines = sc(x)

    img_rot = torchvision.transforms.ToPILImage()(y1[0])
    img = torchvision.transforms.ToPILImage()(y2[0])
    