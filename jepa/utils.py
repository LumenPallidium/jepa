import os
from sympy.ntheory import factorint

class WarmUpScheduler(object):
    """Copilot wrote this, made some small tweaks though."""
    def __init__(self, 
                 optimizer, 
                 scheduler, 
                 warmup_iter, 
                 total_iter = 300000,
                 min_lr = 1e-6):
        self.optimizer = optimizer
        self.scheduler = scheduler(optimizer, 
                                   total_iter - warmup_iter, 
                                   eta_min = min_lr,)
        self.warmup_iter = warmup_iter
        self.iter = 0
    
    def step(self):
        if self.iter < self.warmup_iter:
            lr = self.iter / self.warmup_iter * self.scheduler.get_last_lr()[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
        self.iter += 1

def losses_to_running_loss(losses, alpha = 0.95):
    running_losses = []
    running_loss = losses[0]
    for loss in losses:
        running_loss = (1 - alpha) * loss + alpha * running_loss
        running_losses.append(running_loss)
    return running_losses

def get_latest_file(path, name):
    """Util to get the most recent model checkpoints easily."""
    try:
        files = [os.path.join(path, f) for f in os.listdir(path) if name in f]
        files = [f for f in files if ".pt" in f]
        file = max(files, key = os.path.getmtime)
        # replacing backslashes with forward slashes for windows
        file = file.replace("\\","/")
    except (ValueError, FileNotFoundError):
        file = None
    return file

def ema_update(updated_model, new_model, ema_decay = 0.996):
    for ema_param, new_param in zip(updated_model.parameters(), new_model.parameters()):
        ema_param.data.copy_(ema_param.data * ema_decay + (1 - ema_decay) * new_param.data)
        ema_param.requires_grad_(False)
    return updated_model

def tuple_checker(item, length):
    """Checks if an item is a tuple or list, if not, converts it to a list of length length.
    Also checks that an input tuple is the correct length.
    Useful for giving a function a single item when it requires a iterable."""
    if isinstance(item, (int, float, str)):
        item = [item] * length
    elif isinstance(item, (tuple, list)):
        assert len(item) == length, f"Expected tuple of length {length}, got {len(item)}"
    return item

def approximate_square_root(x):
    factor_dict = factorint(x)
    factors = []
    for key, item in factor_dict.items():
        factors += [key] * item
    factors = sorted(factors)

    a, b = 1, 1
    for factor in factors:
        if a <= b:
            a *= factor
        else:
            b *= factor
    return a, b