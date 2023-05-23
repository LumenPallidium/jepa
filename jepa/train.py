import torch
import torchvision
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from jepa import IJepa
from utils import WarmUpScheduler, losses_to_running_loss
# need to do this for downloading on windows
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

im2tensor = torchvision.transforms.ToTensor()

def tensor2im(x):
    return torchvision.transforms.ToPILImage()(x)

def collate(x):
    x = [x_i[0] for x_i in x]
    return torch.stack(x, dim = 0)


def validation_test(model, 
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

def knn_test(model,
             dataset,
             device,
             batch_size = 64,
             k = 5):
    # importing here cause it's unnecessary for the rest of the code
    from sklearn.neighbors import KNeighborsClassifier

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size = batch_size, 
                                             shuffle = True)
    model.eval()

    knn = KNeighborsClassifier(n_neighbors = k)
    X, y = [], []
    for x, y_i in tqdm(dataloader):
        x = x.to(device)
        with torch.no_grad():
            x = model.encode(x)
            x = x.mean(dim = -2).cpu().numpy()
        X.append(x)
        y.append(y_i)
    X = np.concatenate(X, axis = 0)
    y = np.concatenate(y, axis = 0)

    knn.fit(X, y)
    print(f"KNN score: {knn.score(X, y)}")
    
    model.train()


#TODO : break into functions
#TODO : saving, loading pts
#TODO : better metric logging
#TODO : more probes - kNN probably better than linear (also generally curious about the embedding of the full dataset)

if __name__ == "__main__":
    config = yaml.safe_load(open("../config/training.yml", "r"))

    test_function = knn_test if config["test_type"] == "knn" else validation_test

    warmup_epochs = config["n_epochs"] / config["warmup_epoch_fraction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((config["h"], 
                                                                                          config["w"])),
                                                torchvision.transforms.RandomHorizontalFlip(0.5),
                                                torchvision.transforms.ToTensor()])

    data = torchvision.datasets.Food101(root = config["data_path"], 
                                        split = "train", 
                                        download = True,
                                        transform = transform)
    data_val = torchvision.datasets.Food101(root = config["data_path"],
                                            split = "test",
                                            download = True,
                                            transform = transform)

    steps_per_epoch = len(data) // config["batch_size"]

    model = IJepa(config["h"], 
                  config["w"], 
                  patch_size = config["patch_size"], 
                  n_targets = config["n_targets"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = config["lr"], 
                                  weight_decay = config["weight_decay"], 
                                  amsgrad = True)

    scheduler = WarmUpScheduler(optimizer = optimizer,
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
                                warmup_iter = int(steps_per_epoch * warmup_epochs),
                                total_iter = steps_per_epoch * config["n_epochs"],)

    losses = []

    for epoch in range(config["n_epochs"]):
        print(f"Epoch {epoch + 1}")
        if (config["val_every"] != 0) and (epoch % config["val_every"] == 0):
            test_function(model, data_val, device)

        dataloader = torch.utils.data.DataLoader(data, 
                                                 batch_size = config["batch_size"], 
                                                 shuffle = True,
                                                 collate_fn = collate)
        for i, x in tqdm(enumerate(dataloader)):
            x = x.to(device)
            optimizer.zero_grad()
            preds, x_targets = model(x)

            loss = 0
            for pred, x_target in zip(preds, x_targets):
                loss += torch.nn.functional.mse_loss(pred, x_target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f"\tDone. Final loss: {loss.item()}")

    running_losses = losses_to_running_loss(losses)
    log_losses = np.log(running_losses)
    plt.plot(running_losses)