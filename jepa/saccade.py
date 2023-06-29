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
    
    grid = torch.nn.functional.affine_grid(rot_mat, images.size())
    return torch.nn.functional.grid_sample(images, grid, mode = interp_mode)

class SaccadeCropper(torch.nn.Module):
    def __init__(self,
                 input_h,
                 input_w,
                 target_h = 224,
                 target_w = 224,
                 max_translation = 40,
                 max_rotation = 2,
                 interp_mode = "nearest"
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

        self.requires_grad_(False)


    def forward(self, x):
        with torch.no_grad():
            b, c, h, w = x.shape
            # generate rotations of shape (b, 1)
            rotations = torch.FloatTensor(b, 1).uniform_(-self.max_rotation_rad, self.max_rotation_rad)
            translations = torch.FloatTensor(b, 2).uniform_(-self.max_translation_frac, self.max_translation_frac)

            # apply the roations
            x_affined = rotate_and_translate_batch(x, 
                                                   translations, 
                                                   rotations, 
                                                   interp_mode = self.interp_mode)
            transform_values = torch.cat([rotations, translations], dim = -1)

            # generate two views via center crop
            x1 = torchvision.transforms.functional.center_crop(x_affined, (self.target_h, self.target_w))
            x2 = torchvision.transforms.functional.center_crop(x, (self.target_h, self.target_w))

            # TODO: look into shuffling x1 and x2 ACROSS BATCH
            #views_stack = torch.stack([x1, x2], dim = 0)

            #rand = torch.rand(views_stack.shape[0], views_stack.shape[1])
            #rand_perm = rand.argsort(dim=0)

            #x1, x2 = views_stack[0], views_stack[1]
        return x1, x2, transform_values

if __name__ == "__main__":
    from PIL import Image
    sc = SaccadeCropper(304, 304)
    x = torch.rand(64, 3, 304, 304)

    y1, y2, affines = sc(x)

    img_rot = torchvision.transforms.ToPILImage()(y1[0])
    img = torchvision.transforms.ToPILImage()(y2[0])
    