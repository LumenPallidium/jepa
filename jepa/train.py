import torch
import torchvision
import yaml
import os
import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm
from jepa import ViTJepa, EnergyIJepa, SaccadeJepa
from masked_autoencoder import MaskedAutoencoder, SelfDistillMAE
from einops import rearrange
from utils import WarmUpScheduler, losses_to_running_loss, get_latest_file, ema_update
try:
    from datasets import get_imagenet, ImageNet2017, MAIN_TRANSFORM
except:
    print("Datasets not found - please create your own as needed")
# need to do this for downloading on windows - SAD!
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

im2tensor = torchvision.transforms.ToTensor()

def tensor2im(x):
    return torchvision.transforms.ToPILImage()(x)

def save_im(x, path):
    tensor2im(x).save(path)

def pair_and_save_ims(x, x_hat, model_save_name, epoch, suffix = "", mean = False):
    if x.shape[1] > 1:
        if mean:
            x = x.mean(dim = 1, keepdim = True)
            x_hat = x_hat.mean(dim = 1, keepdim = True)
        else:
            # take middle image
            x = x[:, x.shape[1] // 2, :, :].unsqueeze(1)
            x_hat = x_hat[:, x_hat.shape[1] // 2, :, :].unsqueeze(1)
    x_hat = x_hat.clamp(0, 1)
    pairs = torch.stack((x, x_hat), dim = 1)
    pairs = rearrange(pairs, "b n c h w -> (b n) c h w")

    pairs = torchvision.utils.make_grid(pairs, nrow = 8, padding = 2)
    os.makedirs("../plots/", exist_ok = True)
    save_im(pairs, f"../plots/{model_save_name}_{epoch + 1}{suffix}.png")

def collate(x):
    x = [x_i[0] for x_i in x]
    return torch.stack(x, dim = 0)

def collatev(x):
    x = [x_i for x_i in x]
    return torch.stack(x, dim = 0)

def linear_probe_test(model, 
                    dataset,
                    device,
                    n_categories = 1000,
                    probe_lr = 1e-3,
                    probe_weight_decay = 1e-4,
                    val_epochs = 5,
                    batch_size = 64):
    """Tests the model at a point in its training. Freezes the weights,
    stops the gradients, and trains a linear probe on the dataset"""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    model.eval()
    linear_probe = torch.nn.Linear(model.target_encoder.dim, n_categories).to(device)

    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr = probe_lr, weight_decay = probe_weight_decay, amsgrad = True)

    for epoch in range(val_epochs):
        epoch_scores = []
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = model.embed(x)
            
            optimizer.zero_grad()
            logits = linear_probe(x)
            if x.shape == 3:
                logits = logits.mean(dim = 1)
            loss = torch.nn.functional.cross_entropy(logits, y)

            top1_acc = (logits.argmax(dim = 1) == y).float().mean()
            epoch_scores.append(top1_acc.item())

            loss.backward()
            optimizer.step()
        print(f"\tVal Epoch {epoch + 1} - score: {np.mean(epoch_scores)}")
    model.train()

def generate_val_data(model, dataset, device, batch_size = 16, collater = collatev, n_steps = 300):
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size = batch_size, 
                                             shuffle = True,
                                             collate_fn = collater)
    model.eval()

    X, y = [], []
    for x in tqdm(range(n_steps)):
        x = next(dataloader)
        if len(x) == 2:
            x, y_i = x
            y.append(y_i)
        x = x.to(device).permute(0, 2, 3, 1)
        with torch.no_grad():
            x = model.embed(x)
        X.append(x)
        
    X = torch.cat(X, dim = 0)
    if y:
        y = torch.cat(y, dim = 0)

    model.train()

    return X, y

def coerce_shapes(X, y = None, reduce = False, sample = 0.1, n_labels = None):
    """Coerces the final shapes for the various tests run. Reduce means away
    patches, yielding a single embedding per image. Samples the embeddings
    to reduce the number of points involved, making classic ML methods tractable."""
    if X.shape == 3:
        if reduce:
            X = X.mean(dim = 1)
        else:

            X = einops.rearrange(X, "b l d -> (b l) d")

    if y is not None:
        lengths = X.shape[1]
        y = y.unsqueeze(1).repeat(1, lengths).flatten()

        if n_labels is not None:
            X = X[y < n_labels]
            y = y[y < n_labels]

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
    print(f"Done.\n\tKNN score: {knn.score(X_np, y_np)}", end = "")

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
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y_np, cmap = "turbo", s = 1, alpha = 0.1)
    os.makedirs("../plots/", exist_ok = True)
    fig.savefig(f"../plots/embedding_{epoch}.png", dpi = 300)
    plt.close(fig)

def corr_dimension(X, 
                   log_eps : np.array = None,
                   n_points = 10,
                   base = 10, 
                   plot = True, 
                   coerce_shape = True, 
                   reduce = False, 
                   sample = 0.002,
                   n_labels = 20,
                   eps = 1e-12):
    """Correlation dimension, assumes X has already been stacked or reduced and is shape (n, dim)"""
    if coerce_shape and (len(X.shape) > 2):
        X, _ = coerce_shapes(X, reduce = reduce, sample = sample, n_labels = n_labels)

    lens = X.shape[0]
    X = X # for compatibility with torch.cdist
    denominator = lens ** 2
    with torch.no_grad():
        dists = torch.nn.functional.pdist(X)

        if log_eps is None:
            min_log_eps = np.log(dists.min().item() + eps) / np.log(base)
            max_log_eps = np.log(dists.max().item() + eps) / np.log(base)
            # use midpoint cause max skews corr dim calculation
            midpoint_log_eps = (min_log_eps + max_log_eps) / 2
            log_eps = np.linspace(min_log_eps, midpoint_log_eps, n_points)

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
        plt.close(fig)
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

def ijepa_loss(config, model, x):
    preds, x_targets, context_encoded = model(x)

    loss = 0
    for pred, x_target in zip(preds, x_targets):
        loss += torch.nn.functional.mse_loss(pred, x_target)
    # scale by number of targets and accumulation steps
    loss /= len(preds) * config["accumulation_steps"]

    return loss, preds

def saccade_loss(config, model, x, vicreg = True, eps = 1e-8):
    if model.predict_affines:
        target, target_pred, affines, affines_pred, cycle_loss = model(x)

        loss = torch.nn.functional.mse_loss(target_pred, target)
        loss += cycle_loss * config["cycle_loss_weight"]
        # doing seperate angle/cosine losses and magnitude losses
        true_affine_magnitude = torch.linalg.norm(affines, dim = -1)
        pred_affine_magnitude = torch.linalg.norm(affines_pred, dim = -1)
        affine_cos_loss = torch.dot(affines, affines_pred) / (true_affine_magnitude * pred_affine_magnitude)
        loss += affine_cos_loss.mean() * config["affine_cos_loss_weight"]

        loss += (pred_affine_magnitude - true_affine_magnitude).abs().mean() * config["affine_mag_loss_weight"]
    else:
        target, context, target_pred, cycle_loss = model(x)

        loss = torch.nn.functional.huber_loss(target_pred, target)
        loss += cycle_loss * config["cycle_loss_weight"]

    if vicreg:
        context_mean = context.mean(dim = 0, keepdim = True)

        # variance term
        variance_context = config["vicreg_gamma"] - (context.var(dim = 0) + eps).sqrt()
        variance = torch.nn.functional.relu(variance_context).mean()

        # covariance term
        # (d x batch) @ (batch x d) = (d x d)
        covariance_context = (context - context_mean).T @ (context - context_mean) / context.shape[0]

        # second vicreg term
        covariance = covariance_context.triu().pow(2).sum()
        covariance /= covariance_context.shape[0]
        loss += config["variance_weight"] * variance + config["covariance_weight"] * covariance

    loss /= config["accumulation_steps"]

    return loss, target_pred

def mae_loss(config, model, x,):
    # this is just a wrapper for interoperability
    loss, x_hat = model.get_loss(x)
    loss /= config["accumulation_steps"]
    return loss, x_hat

def get_model(config, device):
    if config["model"] == "ijepa":
        model = ViTJepa(config["h"], 
                        config["w"], 
                        config["d"],
                        target_scale_fraction_range = config["target_scale_fraction_range"],
                        in_channels = config["in_channels"],
                        embed_dim = config["dim"],
                        patch_size = config["patch_size"], 
                        n_targets = config["n_targets"],
                        hw_reduction = config["hw_reduction"]).to(device)
        loss_f = ijepa_loss
    elif config["model"] == "saccade":
        model = SaccadeJepa(in_channels = config["in_channels"]).to(device)
        loss_f = saccade_loss
    elif config["model"] == "mae":
        model = MaskedAutoencoder(config["h"],
                                  w = config["w"],
                                  patch_size = config["patch_size"],
                                  in_channels = config["in_channels"],
                                  patcher_type = config["patcher_type"]).to(device)
        loss_f = mae_loss
    elif config["model"] == "sdm":
        model = SelfDistillMAE(config["h"],
                               w = config["w"],
                               d = config["d"],
                               dim = config["dim"],
                               patch_size = config["patch_size"],
                               mask_prob = config["start_mask_prob"],
                               in_channels = config["in_channels"],
                               patcher_type = config["patcher_type"]).to(device)
        loss_f = mae_loss

    if os.path.exists("../models"):
        model_path = get_latest_file("../models", config["model"])
        if model_path is not None:
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            try:
                start_epoch = int(model_path.split("_")[-2])
            except:
                start_epoch = 1
        else:
            start_epoch = 0
    else:
        os.makedirs("../models", exist_ok = True)
        start_epoch = 0

    if config["use_util_norm"]:
        model.enable_util_norm()

    return model, start_epoch, loss_f

# global for use anywhere
config = yaml.safe_load(open("../config/training.yml", "r"))

#TODO : break into functions
#TODO : saving and loading scheduler + optimizer
#TODO : add function to print singular value count for network
#TODO : vmap may not be working how i want it to in the network forward
#TODO : should output of encoders be normalized? paper says nothing = no?
#TODO : add cls token to jepa
#TODO : for saccade jepa, add centering and temperature diff a la DINO

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["dataset"] == "imagenet":
        transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((config["crop_h"], 
                                                                                         config["crop_w"]),
                                                                                         interpolation = torchvision.transforms.InterpolationMode.NEAREST),
                                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                                torchvision.transforms.ToTensor()])
        data = get_imagenet(config["data_path"],
                            split = "train",
                            transform = transform,)
        data_val = get_imagenet(config["data_path"],
                                split = "val",
                                transform = transform,)

    mini_epochs_per_epoch = len(data) // config["mini_epoch_len"]
    n_mini_epochs = config["n_epochs"] * mini_epochs_per_epoch
    steps_per_mini_epoch = config["mini_epoch_len"] // (config["batch_size"] * config["accumulation_steps"])
    steps_per_epoch = mini_epochs_per_epoch * steps_per_mini_epoch

    model, start_epoch, loss_f = get_model(config, device)

    optimizer = torch.optim.Adam(model.parameters(), 
                                  lr = config["lr"], 
                                  weight_decay = config["weight_decay"],
                                  betas = (config["beta1"], config["beta2"]),
                                  amsgrad = True)
    
    scheduler = WarmUpScheduler(optimizer = optimizer,
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
                                warmup_iter = int(steps_per_epoch * config["n_epochs"] / config["warmup_epoch_fraction"]),
                                total_iter = steps_per_epoch * config["n_epochs"],
                                min_lr = config["min_lr"])

    losses = []

    if (start_epoch == 0) and \
        isinstance(model, (SelfDistillMAE, MaskedAutoencoder)) and \
        config["pretrain_projections"]:
        print("Pretraining projections...")
        for epoch in range(config["pretrain_projections"]):
            dataloader = torch.utils.data.DataLoader(data, 
                                                     batch_size = config["batch_size"], 
                                                     shuffle = True,
                                                     collate_fn = collatev)
            for x in tqdm(dataloader):
                optimizer.zero_grad()
                x = x.to(device)
                if len(x.shape) == 3:
                    # add channel dim
                    x = x.unsqueeze(1)

                x_hat = model.project_deproject(x)

                loss = torch.nn.functional.mse_loss(x_hat,
                                                    x)
                loss.backward()
                optimizer.step()

            pair_and_save_ims(x, x_hat, "projector", epoch)

    for epoch_i in range(config["n_epochs"]):
        epoch = epoch_i + start_epoch

        print(f"Epoch {epoch + 1}")

        dataloader = iter(torch.utils.data.DataLoader(data, 
                                                 batch_size = config["batch_size"], 
                                                 shuffle = True,
                                                 collate_fn = collatev))
        epoch_losses = []

        for mini_epoch in range(mini_epochs_per_epoch):
            print(f"\tMini Epoch {mini_epoch + 1}")

            model_save_name = config["model_save_name"]
            if config["save_mini_epoch"]:
                save_path = f"../models/{model_save_name}_{epoch}_{mini_epoch}.pt"
                model.save(save_path)

            if (config["val_every"] != 0) and (mini_epoch % config["val_every"] == 0) and (data_val is not None):
                run_tests(config["tests"], model, data_val, device, epoch)

            mini_epoch_losses = []

            for i in tqdm(range(steps_per_mini_epoch)):
                optimizer.zero_grad()

                for j in range(config["accumulation_steps"]):
                    x = next(dataloader)
                    x = x.to(device)
                    x = x.permute(0, 2, 3, 1)

                    # shape arithmetic :o
                    rank_4 = (len(x.shape) == 4) and (config["d"] is not None)
                    if (len(x.shape) == 3) or rank_4:
                        # add channel dim
                        x = x.unsqueeze(1)
                    
                    loss, x_hat = loss_f(config, model, x)

                    loss.backward()
                    mini_epoch_losses.append(loss.item())

                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config["clip_grad_norm"])

                optimizer.step()
                scheduler.step()

                if (i % config["save_images_every_mini_epoch"] == 0) & \
                    isinstance(model, (MaskedAutoencoder, SelfDistillMAE)):
                    pair_and_save_ims(x, x_hat, model_save_name, epoch, suffix = f"_{mini_epoch}")

                if not isinstance(model, MaskedAutoencoder):
                    # update after step
                    if isinstance(model, SelfDistillMAE):
                        model.encoder_ema = ema_update(model.encoder_ema, model.encoder, ema_decay = config["ema_decay"])
                    else:
                        model.target_encoder = ema_update(model.target_encoder, model.context_encoder, ema_decay = config["ema_decay"])

            print(f"\t\tDone. Mean Loss: {np.mean(mini_epoch_losses)}")
            epoch_losses.extend(mini_epoch_losses)
                    
        print(f"\tDone. Mean Loss: {np.mean(epoch_losses)}")
        losses.extend(epoch_losses)

        if isinstance(model, (MaskedAutoencoder, SelfDistillMAE)):
            model.update_mask_prob(amount = config["mask_prob_update_amount"])
            if epoch_i % config["save_images_every"] == 0:
                pair_and_save_ims(x, x_hat, model_save_name, epoch)

        save_path = f"../models/{model_save_name}_{epoch + 1}_start.pt"
        if epoch_i % config["save_every"] == 0:
            model.save(save_path)

    print("Done.")
    save_path = f"../models/{model_save_name}_end.pt"
    model.save(save_path)

    running_losses = losses_to_running_loss(losses)
    log_losses = np.log(running_losses)
    plt.plot(running_losses)


