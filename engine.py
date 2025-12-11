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

            constants_dict = self.compute_grad_stats(norm_types=["1", "2", "inf", "frob", "spec"])
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

    def compute_grad_stats(self, norm_types=["1", "2", "inf", "frob", "spec"]) -> Dict[str, Any]:
        """
        Compute Lipschitz proxies with proper norm duality:
        - L_∞ on weights ↔ L_1 on gradients
        - L_1 on weights ↔ L_∞ on gradients
        - L_2 (Frobenius) on weights ↔ L_2 (Frobenius) on gradients (self-dual)
        - Spectral (operator) on weights ↔ Nuclear on gradients
        
        Uses precise SVD for spectral/nuclear norms.
        """
        # Norm duality mapping
        dual_map = {
            "1": "inf",      # L1 dual is L∞
            "inf": "1",      # L∞ dual is L1
            "2": "2",        # L2 is self-dual
            "frob": "frob",  # Frobenius is self-dual
            "spec": "nuc",   # Spectral dual is Nuclear
        }
        
        stats_dict = {
            "lipschitz_layerwise": {nt: {} for nt in norm_types},
            "grad_layerwise": {nt: {} for nt in norm_types}
        }
        
        for name, _ in self.named_params:
            param_name = name
            Wk = self.prev_weights[param_name]
            Wkp1 = self.updated_weights[param_name]
            grad_Wk = self.grads_Wk[param_name]
            grad_Wkp1 = self.grads_Wkp1[param_name]

            delta_grad = grad_Wkp1 - grad_Wk
            delta_W = Wkp1 - Wk
            
            is_matrix = (delta_W.ndim == 2)
            
            # Precompute SVD for spectral/nuclear (only once per tensor)
            svd_cache = {}
            if is_matrix and "spec" in norm_types:
                svd_cache["delta_W"] = torch.linalg.svdvals(delta_W.float())
                svd_cache["delta_grad"] = torch.linalg.svdvals(delta_grad.float())
                svd_cache["grad_Wk"] = torch.linalg.svdvals(grad_Wk.float())

            for norm_type in norm_types:
                dual_norm = dual_map[norm_type]
                
                # Compute weight norm (denominator)
                if norm_type == "1":
                    weight_norm = torch.norm(delta_W.reshape(-1), p=1)
                elif norm_type == "inf":
                    weight_norm = torch.norm(delta_W.reshape(-1), p=float('inf'))
                elif norm_type == "2":
                    weight_norm = torch.norm(delta_W.reshape(-1), p=2)
                elif norm_type == "frob":
                    if is_matrix:
                        weight_norm = torch.linalg.norm(delta_W, ord='fro')
                    else:
                        weight_norm = torch.norm(delta_W.reshape(-1), p=2)
                elif norm_type == "spec":
                    if is_matrix:
                        # Spectral = largest singular value
                        weight_norm = svd_cache["delta_W"][0] if svd_cache["delta_W"].numel() > 0 else torch.tensor(0.0)
                    else:
                        weight_norm = None
                
                # Compute gradient norm (numerator - using DUAL norm)
                if dual_norm == "1":
                    grad_norm = torch.norm(delta_grad.reshape(-1), p=1)
                    grad_Wk_norm = torch.norm(grad_Wk.reshape(-1), p=1)
                elif dual_norm == "inf":
                    grad_norm = torch.norm(delta_grad.reshape(-1), p=float('inf'))
                    grad_Wk_norm = torch.norm(grad_Wk.reshape(-1), p=float('inf'))
                elif dual_norm == "2":
                    grad_norm = torch.norm(delta_grad.reshape(-1), p=2)
                    grad_Wk_norm = torch.norm(grad_Wk.reshape(-1), p=2)
                elif dual_norm == "frob":
                    if is_matrix:
                        grad_norm = torch.linalg.norm(delta_grad, ord='fro')
                        grad_Wk_norm = torch.linalg.norm(grad_Wk, ord='fro')
                    else:
                        grad_norm = torch.norm(delta_grad.reshape(-1), p=2)
                        grad_Wk_norm = torch.norm(grad_Wk.reshape(-1), p=2)
                elif dual_norm == "nuc":
                    if is_matrix:
                        # Nuclear = sum of singular values
                        grad_norm = svd_cache["delta_grad"].sum()
                        grad_Wk_norm = svd_cache["grad_Wk"].sum()
                    else:
                        grad_norm = None
                        grad_Wk_norm = None
                
                # Compute Lipschitz estimate
                if weight_norm is not None and grad_norm is not None:
                    if weight_norm > 0:
                        lipschitz = (grad_norm / weight_norm).item()
                    else:
                        lipschitz = float('inf')
                    
                    stats_dict["lipschitz_layerwise"][norm_type][param_name] = lipschitz
                    stats_dict["grad_layerwise"][norm_type][param_name] = grad_Wk_norm.item() if torch.is_tensor(grad_Wk_norm) else grad_Wk_norm
                else:
                    # Non-matrix parameters don't support spectral/nuclear
                    stats_dict["lipschitz_layerwise"][norm_type][param_name] = None
                    stats_dict["grad_layerwise"][norm_type][param_name] = None

        return stats_dict