from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, base_lr, warmup_ratio=0.1, min_lr=1e-6, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for _ in self.optimizer.param_groups:
            if step < self.warmup_steps:
                lr = self.min_lr + (self.base_lr - self.min_lr) * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs