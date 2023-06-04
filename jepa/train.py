import torch
import torchvision
import yaml
import os
import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm
from jepa import ViTJepa, EnergyIJepa
from utils import WarmUpScheduler, losses_to_running_loss
from datasets import get_imagenet, ImageNet2017
# need to do this for downloading on windows
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

im2tensor = torchvision.transforms.ToTensor()

def tensor2im(x):
    return torchvision.transforms.ToPILImage()(x)

def collate(x):
    x = [x_i[0] for x_i in x]
    return torch.stack(x, dim = 0)


def linear_probe_test(model, 
                    dataset,
                    device,
                    n_categories = 101,
                    probe_lr = 1e-3,
                    probe_weight_decay = 1e-4,
                    val_epochs = 1,):
    """Tests the model at a point in its training. Freezes the weights,
    stops the gradients, and trains a linear probe on the dataset"""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle = True)
    model.eval()
    linear_probe = torch.nn.Linear(model.target_encoder.dim, n_categories).to(device)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr = probe_lr, weight_decay = probe_weight_decay, amsgrad = True)

    for epoch in range(val_epochs):
        epoch_scores = []
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = model.encode(x)
            
            optimizer.zero_grad()
            logits = linear_probe(x).mean(dim = 1)
            loss = torch.nn.functional.cross_entropy(logits, y)

            top1_acc = (logits.argmax(dim = 1) == y).float().mean()
            epoch_scores.append(top1_acc.item())

            loss.backward()
            optimizer.step()
        print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")
    model.train()

def generate_val_data(model, dataset, device, batch_size = 64):
    dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size = batch_size, 
                                                shuffle = True)
    model.eval()

    X, y = [], []
    for x, y_i in tqdm(dataloader):
        x = x.to(device)
        with torch.no_grad():
            x = model.encode(x)
        X.append(x)
        y.append(y_i)
    X = torch.cat(X, dim = 0)
    y = torch.cat(y, dim = 0)

    model.train()

    return X, y

def coerce_shapes(X, y = None, reduce = False, sample = 0.1):
    if reduce:
        X = X.mean(dim = 1)
    else:
        if y is not None:
            lengths = X.shape[1]
            y = y.unsqueeze(1).repeat(1, lengths).flatten()

        X = einops.rearrange(X, "b l d -> (b l) d")

        if sample is not None:
            if sample < 1:
                n_samples = int(sample * X.shape[0])
            else:
                n_samples = int(sample)
            perm = torch.randperm(X.shape[0])[:n_samples]
            X = X[perm, :]
            if y is not None:
                y = y[perm]
    return X, y

def knn_test(X : torch.Tensor, 
             y : torch.Tensor,
             k : int = 5,
             coerce_shape = True,
             reduce = False,
             sample = 0.002):
    # importing here cause it's unnecessary for the rest of the code
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    if coerce_shape and (len(X.shape) > 2):
        X, y = coerce_shapes(X, y = y, reduce = reduce, sample = sample)
    
    X_np = X.cpu().numpy()
    X_np = StandardScaler().fit_transform(X_np)
    y_np = y.cpu().numpy()
    
    knn = KNeighborsClassifier(n_neighbors = k)
    
    print("All images embedded, fitting KNN...", end = "")
    knn.fit(X_np, y_np)
    print(f"Done.\n\tKNN score: {knn.score(X_np, y_np)}\nEmbedding in 2D and plotting...", end = "")

def plot_embedding(X, y, k = 20, coerce_shape = True, reduce = True, sample = 0.002, epoch = None):
    from umap import UMAP
    from sklearn.preprocessing import StandardScaler
    if coerce_shape and (len(X.shape) > 2):
        X, y  = coerce_shapes(X, y = y, reduce = reduce, sample = sample)

    if epoch is None:
        epoch = ""

    X_np = X.cpu().numpy()
    X_np = StandardScaler().fit_transform(X_np)
    y_np = y.cpu().numpy()

    X_embedded = UMAP(n_neighbors = k).fit_transform(X_np)
    # plot it, with colors corresponding to the true labels
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y_np, cmap = "tab10", s = 1, alpha = 0.1)
    os.makedirs("../plots/", exist_ok = True)
    fig.savefig(f"../plots/embedding_{epoch}.png", dpi = 300)


def corr_dimension(X, 
                   log_eps : np.array = None,
                   n_points = 10,
                   base = 10, 
                   plot = False, 
                   coerce_shape = True, 
                   reduce = False, 
                   sample = 0.002):
    """Correlation dimension, assumes X has already been stacked or reduced and is shape (n, dim)"""
    if coerce_shape and (len(X.shape) > 2):
        X, _ = coerce_shapes(X, reduce = reduce, sample = sample)

    lens = X.shape[0]
    X = X # for compatibility with torch.cdist
    denominator = lens ** 2
    with torch.no_grad():
        dists = torch.nn.functional.pdist(X)

        if log_eps is None:
            min_log_eps = np.log(dists.min().item()) / np.log(base)
            max_log_eps = np.log(dists.max().item()) / np.log(base)
            log_eps = np.linspace(min_log_eps, max_log_eps, n_points)

        eps = list(base ** log_eps)
        log_eps = list(log_eps)

        corr_integrals = []
        max_i = 0
        for i, eps_i in enumerate(eps):
            numerator = (dists < eps_i).sum().item()

            corr_integrals.append(numerator / denominator)
            if numerator == 0:
                max_i = i

    log_eps = log_eps[max_i + 1:]
    corr_integrals = corr_integrals[max_i + 1:]
    log_corr_integrals = np.log(corr_integrals)
    if plot:
        os.makedirs("../plots/", exist_ok = True)
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(log_eps, log_corr_integrals)
        ax.set_xlabel("log(eps)")
        ax.set_ylabel("log(C(eps))")
        fig.savefig("../plots/corr_integrals.png", dpi = 300)
    if len(log_eps) <= 1:
        log_slope = np.nan
    else:
        log_slope = np.polyfit(log_eps, log_corr_integrals, 1)[0]
    print(f"\tCorrelation Dimension: {np.round(log_slope, 4)}")


def run_tests(test_list, model, data_val, device, epoch, sample = 4096):
    """Slightly inefficent and a bit verbose, but makes a nice wrapper so you can 
    specify tests in a list"""
    data_generated = False
    for test in test_list:
        if test == "knn":
            if not data_generated:
                X, y = generate_val_data(model, data_val, device)
                data_generated = True
            knn_test(X, y, sample = sample)
        elif test == "corr_dim":
            if not data_generated:
                X, y = generate_val_data(model, data_val, device)
                data_generated = True
            corr_dimension(X, sample = sample)
        elif test == "plot":
            # not technically a test :)
            if not data_generated:
                X, y = generate_val_data(model, data_val, device)
                data_generated = True
            plot_embedding(X, y, epoch = epoch, sample = sample)
        elif test == "linear_probe":
            linear_probe_test(model, data_val, device)
        else:
            raise ValueError(f"Unknown test {test}")

#TODO : break into functions
#TODO : saving, loading pts
#TODO : add function to print singular value count for network

if __name__ == "__main__":
    config = yaml.safe_load(open("../config/training.yml", "r"))

    warmup_epochs = config["n_epochs"] / config["warmup_epoch_fraction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((config["h"], 
                                                                                          config["w"])),
                                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                                torchvision.transforms.ToTensor()])

    data = get_imagenet(config["data_path"],
                        split = "train",
                        transform = transform,)
    data_val = get_imagenet(config["data_path"],
                            split = "val",
                            transform = transform,)

    mini_epochs_per_epoch = len(data) // config["mini_epoch_len"]
    steps_per_mini_epoch = config["mini_epoch_len"] // (config["batch_size"] * config["accumulation_steps"])
    steps_per_epoch = mini_epochs_per_epoch * steps_per_mini_epoch

    model = ViTJepa(config["h"], 
                  config["w"], 
                  patch_size = config["patch_size"], 
                  n_targets = config["n_targets"]).to(device)
    os.makedirs("../models", exist_ok = True)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config["lr"], 
                                  weight_decay = config["weight_decay"],
                                  betas = (config["beta1"], config["beta2"]),
                                  amsgrad = True)

    scheduler = WarmUpScheduler(optimizer = optimizer,
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
                                warmup_iter = int(steps_per_epoch * warmup_epochs),
                                total_iter = steps_per_epoch * config["n_epochs"],)

    losses = []

    for epoch in range(config["n_epochs"]):
        print(f"Epoch {epoch + 1}")
        if (config["val_every"] != 0) and (epoch % config["val_every"] == 0):
            run_tests(config["tests"], model, data_val, device, epoch)
            

        dataloader = iter(torch.utils.data.DataLoader(data, 
                                                 batch_size = config["batch_size"], 
                                                 shuffle = True,
                                                 collate_fn = collate))
        epoch_losses = []

        for mini_epoch in range(mini_epochs_per_epoch):

            print(f"\tMini Epoch {mini_epoch + 1}")
            torch.save(model.state_dict(), f"../models/ijepa_{epoch}_{mini_epoch}.pt")

            mini_epoch_losses = []

            for i in tqdm(range(steps_per_mini_epoch)):
                optimizer.zero_grad()

                for j in range(config["accumulation_steps"]):
                    x = next(dataloader)
                    x = x.to(device)
                    
                    preds, x_targets = model(x)

                    loss = 0
                    for pred, x_target in zip(preds, x_targets):
                        loss += torch.nn.functional.mse_loss(pred, x_target)
                    # scale by number of targets and accumulation steps
                    loss /= len(preds) * config["accumulation_steps"]

                    loss.backward()
                    mini_epoch_losses.append(loss.item())

                optimizer.step()
                scheduler.step()

            print(f"\t\tDone. Mean Loss: {np.mean(mini_epoch_losses)}")
            epoch_losses.extend(mini_epoch_losses)
                    
        print(f"\tDone. Mean Loss: {np.mean(epoch_losses)}")
        losses.extend(epoch_losses)

    running_losses = losses_to_running_loss(losses)
    log_losses = np.log(running_losses)
    plt.plot(running_losses)
