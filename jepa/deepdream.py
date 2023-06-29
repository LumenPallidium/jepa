import torch
from tqdm import tqdm
from train import tensor2im, config, get_model

def deepdream(input, layer_n, jepa_model, neuron_j = None, feedforward_layer = False, n_iters = 100):

    optimizer = torch.optim.SGD([input], lr = 0.1)

    model = jepa_model.target_encoder

    for i in tqdm(range(n_iters)):
        input_embedded = jepa_model.embedder(input) + model.pos_embedding
        for n, (attention, ff) in enumerate(model.layers):
            optimizer.zero_grad()
            if n == layer_n:
                if feedforward_layer:
                    input_embedded = input_embedded + ff(input_embedded)
                else:
                    input_embedded = input_embedded + attention(input_embedded)
            else:
                input_embedded = input_embedded + attention(input_embedded)
                input_embedded = input_embedded + ff(input_embedded)

            if neuron_j is None:
                loss = input_embedded.norm()
            else:
                loss = input_embedded[0, neuron_j, :].norm() - input_embedded[0, ~neuron_j, :].mean()
             
        loss.backward()
        optimizer.step()
    return input.detach()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_model, _ = get_model(config, device)

    x = torch.zeros(1, 3, config["h"], config["w"], requires_grad = True, device = device)
    y = deepdream(x, 5, jepa_model, neuron_j = 42, feedforward_layer = False)
    tensor2im(y.squeeze(0))
