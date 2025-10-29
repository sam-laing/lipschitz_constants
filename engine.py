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

        for opt in self.optimizer:
            opt.step()

        # now look at the updated weights and gradients if tracking lipschitz
        if self.cfg.track_lipschitz:
            # W_{k+1}
            self.updated_weights = { name: param.clone().detach() for name, param in self.model.named_parameters() }
            #compute loss with new grads and backward for new grads
            new_loss = self.criterion(self.model(x), y)
            new_loss.backward()
            self.grads_Wkp1 = { name: p.grad.detach().clone() for name, p in self.model.named_parameters() }

            constants_dict = self.compute_grad_stats(norm_types=["1", "2", "spec"])
            metrics_dict.update(constants_dict)

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
        compute global lipschitz constant estimate based on all parameters

        returns a nice json style dict with layerwise and global lipshitz constants and grad norms
        """
        # dual norm:
        norm_dual_dict = {
            "2" : 2, 
            "frob": "frob",
            "inf" : 1,
            "1" : float('inf'),
            "spec": "nuc", 
        }

        for p in [3,4,5,6,7,8,9]:
            norm_dual_dict[str(p)] = p/(p-1)

        #vfor lipschitz wrt norm n(), we do 
        # L_n = norm*(grad_W f(W_{k+1}; X_k,y_k) - grad_W f(W_k; X_k,y_k)) / norm(W_{k+1} - W_k) 
        # where norm* is the dual norm of norm
        stats_dict = {
            "lipschitz_global": {"L2": 0.0, "L1": 0.0, "spec": 0.0, "frob": 0.0}, 
            "grad_global": {"L2": 0.0, "L1": 0.0, "spec": 0.0, "frob": 0.0}, 
            "lipschitz_layerwise": {"L2": {}, "L1": {}, "spec": {}, "frob": {}},
            "grad_layerwise": {"L2": {}, "L1": {}, "spec": {}, "frob": {}}
            }
        for name in self.named_params:
            #literally just compute the 2 norm (which is self dual) by flattening the 2d to 1d if necessary
            # just want a poc
            param_name = name[0]
            Wk = self.prev_weights[param_name]
            Wkp1 = self.updated_weights[param_name]
            grad_Wk = self.grads_Wk[param_name]
            grad_Wkp1 = self.grads_Wkp1[param_name]

            delta_grad = grad_Wkp1 - grad_Wk
            delta_W = Wkp1 - Wk

            lipschitz_2 = torch.norm(delta_grad.view(-1), p=2) / torch.norm(delta_W.view(-1), p=2) if torch.norm(delta_W.view(-1), p=2) > 0 else float('inf')
            grad_norm_2 = torch.norm(grad_Wk.view(-1), p=2)
            lipschitz_frob = torch.norm(delta_grad, p='fro') / torch.norm(delta_W, p='fro') if torch.norm(delta_W, p='fro') > 0 else float('inf')
            grad_norm_frob = torch.norm(grad_Wk, p='fro')
            #get the spec norm/nuclear norm lipschitz only if matrix layer
            if len(delta_W.shape) == 2:
                #do max sing value norm and nuclear norm which is sum of sing values
                nuclear_norm_delta_grad = torch.norm(delta_grad, p='nuc')
                max_sing_value_delta_W = torch.norm(delta_W, p=2)
                lipschitz_spec = nuclear_norm_delta_grad / max_sing_value_delta_W if max_sing_value_delta_W > 0 else float('inf')
                spec_norm_grad = torch.norm(grad_Wk, p='nuc')
            else:
                lipschitz_spec = None
                spec_norm_grad = None

            stats_dict["lipschitz_layerwise"]["L2"][param_name] = lipschitz_2.item()
            stats_dict["grad_layerwise"]["L2"][param_name] = grad_norm_2.item()
            stats_dict["lipschitz_layerwise"]["frob"][param_name] = lipschitz_frob.item()
            stats_dict["grad_layerwise"]["frob"][param_name] = grad_norm_frob.item()
            stats_dict["lipschitz_layerwise"]["spec"][param_name] = lipschitz_spec.item() if lipschitz_spec is not None else None
            stats_dict["grad_layerwise"]["spec"][param_name] = spec_norm_grad.item() if spec_norm_grad is not None else None


        return stats_dict


        """ 
        stats_dict = {}
        for name in self.named_params:
            param_name = name[0]

            Wk = self.prev_weights[param_name]
            Wkp1 = self.updated_weights[param_name]
            grad_Wk = self.grads_Wk[param_name]
            grad_Wkp1 = self.grads_Wkp1[param_name]

            delta_grad = grad_Wkp1 - grad_Wk
            delta_W = Wkp1 - Wk

            stats_dict[param_name] = {}

            for n in norm_types:
                dual_n = norm_dual_dict[str(n)]


                grad_norm = torch.norm(delta_grad, p=dual_n).item()
                weight_norm = torch.norm(delta_W, p=n).item()

                if weight_norm > 0:
                    L_n = grad_norm / weight_norm
                else:
                    L_n = float('inf')

                stats_dict[param_name][f"grad_norm_p{dual_n}"] = grad_norm
                stats_dict[param_name][f"weight_norm_p{n}"] = weight_norm
                stats_dict[param_name][f"lipschitz_estimate_L{n}"] = L_n
        """ 




def _norm(t, norm_key):
    if isinstance(norm_key, int):
        return torch.norm(t, p=norm_key)
    elif norm_key=="frob":
        return torch.norm(t, p='fro')
    elif norm_key=="spec":
        assert len(t.shape) == 2, "Spectral norm only defined for matrices"
        return torch.norm(t, p=2)
    
    elif norm_key == "nuc":
        assert len(t.shape) == 2, "Nuclear norm only defined for matrices"
        return torch.norm(t, p='nuc')


    
 




    

