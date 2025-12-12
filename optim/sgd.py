import torch
from torch.optim import Optimizer

# looking for a version of SGD with dual decay option
# comparing proximal descent vs LMO (LMO would yield normalized sgd)

class SGD(Optimizer):
	def __init__(self, params, lr, momentum=0.0, weight_decay=0.1, dual_decay=True):
		if not 0.0 <= lr:
			raise ValueError(f'Invaid learing rate: {lr}')
		if not 0.0 <= momentum or not momentum <= 1.0:
			raise ValueError(f'Invaid momentum: {momentum}')
		if not 0.0 <= weight_decay:
			raise ValueError(f'Invaid weight decay: {weight_decay}')

		defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dual_decay=dual_decay)
		super(SGD, self).__init__(params, defaults)

	@torch.no_grad()
	def step(self):
		for group in self.param_groups:
			alpha = group['lr']
			momentum = group['momentum']
			weight_decay = group['weight_decay']
			dual_decay = group['dual_decay']

			for p in group['params']:
				if p.grad is None:
					continue
				param_state = self.state[p]

				# weight Decay
				p.mul_(1 - alpha * weight_decay)

				# m initialization
				if 'm' not in param_state:
					param_state['m'] = p.grad.detach().clone()

				# decay momentum
				m = param_state['m']
				m.mul_(momentum).add_(p.grad, alpha=(1.0 - momentum))

				# if not dual decay, normalize the gradient by its norm
				if dual_decay:
					#normal gradient descent
					update = m
				else:
					# lmo gives normalized update
					# take the rms norm
					norm = m.norm() / (m.numel() ** 0.5)
					
					update = m / norm if norm != 0 else m

				p.add_(update, alpha=-alpha)

				