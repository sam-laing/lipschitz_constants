import torch 
import torch.nn as nn  
from optim import init_optimizer
from typing import Dict, Any

class Engine(nn.Module):
    """  
    if cfg.track_lipschitz: 
        Looking at norm*(grad_W f(W_{k+1}; X_k,y_k) - grad_W f(W_k; X_k,y_k)) / norm(W_{k+1} - W_k) as a lipschitz constant proxy

        at each step: need to store previous weights and gradients, then update and get new ones after an optimizer step
    otherwise standard training and val engine to keep train.py script a bit nicer
    """
    def __init__(self, model, cfg):
        super(Engine, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        #check the number of model params by layer and total and print
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total model parameters: {total_params}")
        print("Model parameters by layer:") 
        for name, param in self.model.named_parameters():
            print(f"Param: {name}, Shape: {param.shape}, Numel: {param.numel()}")

        self.named_params = list(self.model.named_parameters())
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        opt = init_optimizer(cfg, model)
        self.optimizer = list(opt) if isinstance(opt, (list, tuple)) else [opt]
        self.iteration = 0
        if cfg.track_lipschitz:
            #initialize previous weights 
            self.prev_weights = { name: param.clone().detach() for name, param in self.model.named_parameters() }
    
    def step(self, x,y):
        self.model.train()

        x, y = x.to(self.device), y.to(self.device)
        
        for opt in self.optimizer:
            opt.zero_grad()
        

        output = self.model(x)
        loss = self.criterion(output, y)
        #do the update to the weights
        loss.backward()

        metrics_dict = {"loss": loss.item()}

        if self.cfg.track_lipschitz:
            #store previous gradients
            self.grads_Wk = { name: p.grad.detach().clone() for name, p in self.model.named_parameters() }

        #step optimizer(s)
        for opt in self.optimizer:
            opt.step()

        # now look at the updated weights and gradients if tracking lipschitz
        if self.cfg.track_lipschitz:
            # W_{k+1}
            self.updated_weights = { name: param.clone().detach() for name, param in self.model.named_parameters() }
            #compute loss with new grads and backward for new grads
            for opt in self.optimizer:
                opt.zero_grad()
            new_loss = self.criterion(self.model(x), y)
            new_loss.backward()
            self.grads_Wkp1 = { name: p.grad.detach().clone() for name, p in self.model.named_parameters() }

            constants_dict = self.compute_grad_stats(norm_types=["2", "spec"])
            metrics_dict.update(constants_dict)
            #weight update
            self.prev_weights = self.updated_weights

        self.iteration += 1
        return metrics_dict


    @torch.no_grad()
    def eval(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == y).sum().item()

        metrics_dict = {
            "loss": total_loss / len(val_loader.dataset),
            "accuracy": total_correct / len(val_loader.dataset)
        }
        return metrics_dict

    def compute_grad_stats(self, norm_types=["2"]) -> Dict[str, Any]:
        """
        Compute Lipschitz proxies per-parameter using precise SVD-based norms.
        Returns layerwise estimates for L2 (vectorized Frobenius) and spec (operator/nuclear).
        """
        stats_dict = {
            "lipschitz_layerwise": {"L2": {}, "spec": {}},
            "grad_layerwise": {"L2": {}, "spec": {}}
        }
        
        for name, _ in self.named_params:
            param_name = name
            Wk = self.prev_weights[param_name]
            Wkp1 = self.updated_weights[param_name]
            grad_Wk = self.grads_Wk[param_name]
            grad_Wkp1 = self.grads_Wkp1[param_name]

            delta_grad = grad_Wkp1 - grad_Wk
            delta_W = Wkp1 - Wk

            # --- L2 (vectorized Frobenius) ---
            denom_l2 = torch.norm(delta_W.reshape(-1), p=2)
            lipschitz_2 = (torch.norm(delta_grad.reshape(-1), p=2) / denom_l2).item() if denom_l2 > 0 else float('inf')
            grad_norm_2 = torch.norm(grad_Wk.reshape(-1), p=2).item()

            stats_dict["lipschitz_layerwise"]["L2"][param_name] = lipschitz_2
            stats_dict["grad_layerwise"]["L2"][param_name] = grad_norm_2

            # --- Spectral/Nuclear (only for 2D matrices) ---
            if delta_W.ndim == 2 and "spec" in norm_types:
                # Compute singular values once for both norms
                S_grad = torch.linalg.svdvals(delta_grad.float())  # [min(m,n)]
                S_W = torch.linalg.svdvals(delta_W.float())

                # Operator norm = max singular value
                spec_norm_W = S_W[0].item() if S_W.numel() > 0 else 0.0
                # Nuclear norm = sum of singular values
                nuclear_norm_grad = S_grad.sum().item()

                lipschitz_spec = (nuclear_norm_grad / spec_norm_W) if spec_norm_W > 0 else float('inf')

                # Grad spectral: nuclear norm of grad_Wk
                S_grad_k = torch.linalg.svdvals(grad_Wk.float())
                spec_norm_grad = S_grad_k.sum().item()

                stats_dict["lipschitz_layerwise"]["spec"][param_name] = lipschitz_spec
                stats_dict["grad_layerwise"]["spec"][param_name] = spec_norm_grad
            else:
                stats_dict["lipschitz_layerwise"]["spec"][param_name] = None
                stats_dict["grad_layerwise"]["spec"][param_name] = None

        return stats_dict