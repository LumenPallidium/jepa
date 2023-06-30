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


def deepdream_ijepa(input, layer_n, jepa_model, neuron_j = None, feedforward_layer = False, n_iters = 100):

    optimizer = torch.optim.SGD([input], lr = 0.1)

    model = jepa_model.target_encoder

    for i in tqdm(range(n_iters)):
        input_embedded = jepa_model.embedder(input) + model.pos_embedding
        optimizer.zero_grad()
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

        if neuron_j is None:
            loss = input_embedded.norm()
        else:
            loss = input_embedded[0, neuron_j, :].norm()
             
        loss.backward()
        optimizer.step()
    return input.detach()

def deepdream_resnet(input, model, layer_n = 1, channel = None, neuron_idx_i = None, neuron_idx_j = None, n_iters = 100, lr = 0.1):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    denormalize = torchvision.transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                   std = [1/0.229, 1/0.224, 1/0.255])

    input.requires_grad_(True)

    for i in tqdm(range(n_iters)):
        x = normalize(input)

        #shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        #x = torch.roll(torch.roll(x, shift_x, -1), shift_y, -2)

        layers = get_leaf_modules(model)
        n_layers = len(layers)

        if layer_n >= n_layers:
            # this avoids a weird operation in the last layer
            x = model(x)
        else:
            curr_layer = 0
            prev_layer = 0
            downsampling = False
            for n, (name, layer) in enumerate(layers):
                # you don't have to tell me how awful this is
                # but the downsampling means you can't just iterate through layers
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
                loss = -x.norm()
            else:
                if neuron_idx_i is None:
                    loss = -(x[0, channel, :, :].norm())
                else:
                    neuron_idx_j = neuron_idx_i if neuron_idx_j is None else neuron_idx_j
                    loss = -(x[0, channel, neuron_idx_i, neuron_idx_j].norm())
        elif len(x.shape) == 2:
            if neuron_idx_i is None:
                loss = -x.norm()
            else:
                loss = -(x[0, neuron_idx_i].norm())
             
        # add reg term
        #loss += 0.1 * input.norm()

        loss.backward()

        # add gradient by hand
        grad = input.grad.data / input.grad.mean()
        input.data += grad * lr
        input.data = torch.clamp(input.data, -1, 1)
        input.grad.data.zero_()

    return denormalize(input).detach()

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

    #jepa_model, _, _ = get_model(config, device)

    x = torch.zeros(1, 3, 224, 224, device = device)

    if config["model"] == "ijepa":
        y = deepdream_ijepa(x, 5, jepa_model, neuron_j = 42, feedforward_layer = False)
        tensor2im(y.squeeze(0))
    elif config["model"] == "saccade":
        resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
        y = deepdream_octaves(x, resnet, layer_n = 13,  channel = 3)
        tensor2im(y.squeeze(0))
