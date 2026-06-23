import sys, torch
from types import SimpleNamespace

_HERE = "/content/lipschitz_constants"
sys.path.insert(0, _HERE)

from data import cifar10_5k_make_loaders
from engine import Engine
from models import build_model

SEED   = 42
INITS  = ["kaiming", "kaiming_orthog", "orthogonal", "perturbed"]

base = SimpleNamespace(
    dataset="cifar10_5k", data_root=None,
    num_workers=0, batch_size="full", seed=SEED,
    model="mlp_ortho_v2", hidden_dim=1.5, output_dim=5,
    activation="relu", use_bias=False, seperate_biases=False,
    init_mode=None, init_gain=0.02, init_nonlinearity="relu",
    perturb_radius=1.0, perturb_w_star_mode="kaiming",
    optimizer="sgd", lr=1e-3,
    momentum=0.0, nesterov=False, weight_decay=0.0,
    dual_decay=True, adjust_lr=False,
    orthogonalize=True, ns_steps=5,
    beta1=0.9, beta2=0.999, eps=1e-8,
    line_search_bracket=4.0, loss="cross_entropy",
    track_lipschitz=False, track_hessian=False,
    scheduler=None, wandb_project_name=None, iters=1,
)

torch.manual_seed(SEED)
cfg = SimpleNamespace(**vars(base))
cfg.init_mode = INITS[0]
train_loader, val_loader, _ = cifar10_5k_make_loaders(cfg)

print(f"{'init':<20}  train_loss   val_loss")
print("-" * 45)
for init in INITS:
    cfg = SimpleNamespace(**vars(base))
    cfg.init_mode = init
    torch.manual_seed(SEED)
    model = build_model(cfg)
    engine = Engine(model=model, cfg=cfg)
    t = engine.eval(train_loader)
    v = engine.eval(val_loader)
    print(f"{init:<20}  {t['loss']:.4f}       {v['loss']:.4f}")
