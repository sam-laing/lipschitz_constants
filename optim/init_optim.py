import torch

def init_optimizer(cfg, model):
    if cfg.optimizer == 'muon':
        from optim.muon import Muon
        if cfg.separate_biases:
            biases = []
            non_biases = []
            for name, param in model.named_parameters():
                if len(param.shape) == 1 or 'bias' in name:
                    biases.append(param)
                else:
                    non_biases.append(param)
            
            muon_optimizer = Muon(
                non_biases, 
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                ns_steps=cfg.ns_steps,
                orthogonalize=cfg.orthogonalize,
                weight_decay=cfg.weight_decay,
                adjust_lr=cfg.adjust_lr
            )
            adamw_optimizer = torch.optim.AdamW(
                biases,
                lr=cfg.lr,
                weight_decay=0.0,
                betas = (cfg.beta1, cfg.beta2),
                eps=cfg.eps
            )
            return muon_optimizer, adamw_optimizer
        else:
            muon_optimizer = Muon(
                model.parameters(), 
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                ns_steps=cfg.ns_steps,
                orthogonalize=cfg.orthogonalize,
                weight_decay=cfg.weight_decay,
                adjust_lr=cfg.adjust_lr
            )
            return muon_optimizer
    elif cfg.optimizer == "kj_muon":   
        from optim.kj_muon import Muon
        if cfg.separate_biases:
            biases = []
            non_biases = []
            for name, param in model.named_parameters():
                if len(param.shape) == 1 or 'bias' in name:
                    biases.append(param)
                else:
                    non_biases.append(param)
            
            muon_optimizer = Muon(
                non_biases, 
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                ns_steps=cfg.ns_steps,
                adjust_lr=cfg.adjust_lr
            )
            adamw_optimizer = torch.optim.AdamW(
                biases,
                lr=cfg.lr,
                weight_decay=0.0,
                betas = (cfg.beta1, cfg.beta2),
                eps=cfg.eps
            )
            return muon_optimizer, adamw_optimizer
        else:
            muon_optimizer = Muon(
                model.parameters(), 
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                ns_steps=cfg.ns_steps,
                orthogonalize=cfg.orthogonalize,
                weight_decay=cfg.weight_decay,
                adjust_lr=cfg.adjust_lr
            )
            return muon_optimizer


        
    elif cfg.optimizer == 'sign_sgd':
        from optim.sign_sgd import signSGD
        sign_sgd_optimizer = signSGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        return sign_sgd_optimizer
    
    elif cfg.optimizer == "sgd":
        sgd_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov
        )
        return sgd_optimizer
    
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
