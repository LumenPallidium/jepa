import os

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
        file = max(files, key = os.path.getmtime)
        # replacing backslashes with forward slashes for windows
        file = file.replace("\\","/")
    except (ValueError, FileNotFoundError):
        file = None
    return file

def ema_update(updated_model, new_model):
    for ema_param, new_param in zip(updated_model.parameters(), new_model.parameters()):
        ema_param.data.copy_(ema_param.data * updated_model.ema_decay + (1 - updated_model.ema_decay) * new_param.data)