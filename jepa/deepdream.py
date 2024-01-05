import torch
import torchvision
from tqdm import tqdm
import numpy as np
from train import tensor2im, config, get_model

def get_leaf_modules(model):
    """Easiest way to get all layers in a pretrained model."""
    leaf_modules = []
    for n, module in model.named_modules():
        if len(list(module.children())) == 0:
            leaf_modules.append((n, module))
    return leaf_modules

def optimize_step(input, lr, blur_kernel = None):
    # add gradient by hand
    grad = input.grad.data
    if torch.any(grad != 0):
        grad = grad / grad.abs().mean() # normalize gradient

        #print(grad)
        if blur_kernel is not None:
            in_c = grad.shape[1]
            blur_kernel = blur_kernel.expand(in_c, 1, 3, 3)

            grad = torch.nn.functional.conv2d(grad, blur_kernel, padding = 1, groups = in_c)
    

    input.data.add_(grad * lr)
    input.grad.data.zero_()
    return input

def get_imagenet_stats():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    denormalize = torchvision.transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                   std = [1/0.229, 1/0.224, 1/0.255])

    return normalize, denormalize

def deepdream_ijepa(input, layer_n, jepa_model,
                     neuron_j = None, 
                     feedforward_layer = False,
                     n_iters = 100, 
                     lr = 0.1,
                     max_jitter = 32):

    model = jepa_model.target_encoder
    normalize, denormalize = get_imagenet_stats()

    input.requires_grad_(True)

    for i in tqdm(range(n_iters)):
        input_embedded = normalize(input)

        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        input_embedded = torch.roll(torch.roll(input_embedded, shift_x, -1), shift_y, -2)

        input_embedded = jepa_model.embedder(input_embedded) + model.pos_embedding
        for n, (attention, ff) in enumerate(model.layers):
            if n == layer_n:
                if feedforward_layer:
                    input_embedded = input_embedded + ff(input_embedded)
                else:
                    input_embedded = input_embedded + attention(input_embedded)
                break
            else:
                input_embedded = input_embedded + attention(input_embedded)
                input_embedded = input_embedded + ff(input_embedded)

        input_embedded = torch.roll(torch.roll(x, -shift_x, -1), -shift_y, -2)

        if neuron_j is None:
            loss = input_embedded.norm()
        else:
            loss = input_embedded[0, neuron_j, :].norm()
             
        loss.backward()

        input = optimize_step(input, lr, 
                              blur_kernel = blur_kernel)

    input = denormalize(input).detach()

    return torch.clamp(input, 0, 1)

def deepdream_resnet(input, 
                     model, 
                     layer_n = 1, 
                     channel = None, 
                     neuron_idx_i = None, 
                     neuron_idx_j = None, 
                     n_iters = 128, 
                     lr = 0.1,
                     max_jitter = 32,
                     blur_kernel = None,
                     leafs_or_children = "leafs"):
    normalize, denormalize = get_imagenet_stats()   

    input.requires_grad_(True)

    for i in tqdm(range(n_iters)):
        x = normalize(input)

        # shift the image to find translation invariant activation maximizers
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        x = torch.roll(torch.roll(x, shift_x, -1), shift_y, -2)

        if leafs_or_children == "leafs":
            layers = get_leaf_modules(model)
        else:
            layers = list(model.named_children())
        n_layers = len(layers)

        if layer_n >= n_layers:
            # this avoids a weird operation in the last layer
            x = model(x)
            name = "full model"
            n = n_layers
        else:
            curr_layer = 0
            prev_layer = 0
            downsampling = False
            for n, (name, layer) in enumerate(layers):
                # you don't have to tell me how awful this is
                # but resnet downsampling means you can't just iterate through layers
                if "layer" in name:
                    layer_num = name.split(".")[0][-1]
                    curr_layer = int(layer_num)

                # store inital for the downsampling
                if curr_layer != prev_layer:
                    x_store = x

                if not downsampling:
                    if "downsample" in name:
                        # downsample
                        downsampling = True
                        x_store = layer(x_store)
                    else:
                        x = layer(x)
                else:
                    if "downsample" in name:
                        # downsample
                        x_store = layer(x_store)
                    else:
                        x = layer(x + x_store)
                        downsampling = False

                prev_layer = curr_layer
                if n == layer_n:
                    break

        if len(x.shape) == 4:
            if channel is None:
                loss = x.norm()
            else:
                if neuron_idx_i is None:
                    loss = (x[0, channel, :, :].norm())
                else:
                    if neuron_idx_j is None:
                        loss = (x[0, channel, neuron_idx_i, :].norm())
                    else:
                        loss = (x[0, channel, neuron_idx_i, neuron_idx_j].norm())
        elif len(x.shape) == 2:
            if neuron_idx_i is None:
                loss = x.norm()
            else:
                loss = (x[0, neuron_idx_i].norm())

        loss.backward()

        input = optimize_step(input, lr, 
                              blur_kernel = blur_kernel)

    print(f"Stopping at layer {name} ({n}))")
    input = denormalize(input).detach()

    return torch.clamp(input, 0, 1)

def deepdream_octaves(input, model, octave_scale = 1.4, n_octaves = 6, **kwargs):
    octaves = [input]
    for i in range(n_octaves - 1):
        octaves.append(torch.nn.functional.interpolate(octaves[-1], scale_factor = 1 / octave_scale, mode = "bilinear", align_corners = False))

    detail = torch.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        if octave > 0:
            detail = torch.nn.functional.interpolate(detail, 
                                                     size = octave_base.shape[-2:], 
                                                     mode = "bilinear", 
                                                     align_corners = False)
        input = octave_base + detail
        input = deepdream_resnet(input, model, **kwargs)
        detail = input - octave_base
    return input

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blur_kernel = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], device = device, dtype = torch.float32) / 16

    jepa_model, _, _ = get_model(config, device)
    sj = jepa_model.target_encoder

    x = torch.rand(1, 3, 500, 500, device = device)

    if config["model"] == "ijepa":
        y = deepdream_ijepa(x, 5, jepa_model, neuron_j = 0, feedforward_layer = False)
        tensor2im(y.squeeze(0))
    elif config["model"] == "saccade":
        #resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device).requires_grad_(False)
        inception = torchvision.models.inception_v3(weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
                                                    ).to(device).requires_grad_(False)
        y = deepdream_octaves(x, sj, 
                              layer_n = 1,  
                              channel = 1,
                              neuron_idx_i = 0,
                              neuron_idx_j = 0,
                              blur_kernel = blur_kernel,
                              leafs_or_children = "children")
        tensor2im(y.squeeze(0))

        x = torch.rand(1, 3, 299, 299, device = device)
        y = deepdream_octaves(x, sj, 
                              layer_n = 1, 
                              blur_kernel = blur_kernel,
                              leafs_or_children = "children")
        tensor2im(y.squeeze(0))
