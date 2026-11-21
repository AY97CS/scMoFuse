"""
DeepMLP: a plain feed-forward predictor with no bottleneck and no shared
latent space, included as the member of the ensemble that assumes least about
the data. Two branches per modality pair, one per direction, sharing no
parameters but optimised jointly under the sum of their losses.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from contextlib import nullcontext

class _DeepMLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_layers, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepMLP(nn.Module):
    """Generic A↔B predictor. For Multiome use A=ATAC, B=GEX. For CITE-seq use A=ADT, B=GEX."""
    def __init__(self, a_dim, b_dim, hidden=512, n_layers=5, dropout=0.2):
        super().__init__()
        self.a2b = _DeepMLPBranch(a_dim, b_dim, hidden, n_layers, dropout)
        self.b2a = _DeepMLPBranch(b_dim, a_dim, hidden, n_layers, dropout)
        self.a2g = self.a2b
        self.g2a = self.b2a

    @torch.no_grad()
    def predict_a2b(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_b2a(self, x): return self.b2a(x)

    @torch.no_grad()
    def predict_atac2gex(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_gex2atac(self, x): return self.b2a(x)

class _ResBlock(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.fc1   = nn.Linear(hidden, hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(self.norm1(x))
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(self.norm2(h))
        return x + h

class _ResMLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_blocks, dropout):
        super().__init__()
        self.proj_in  = nn.Linear(in_dim, hidden)
        self.blocks   = nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = F.gelu(self.proj_in(x))
        for blk in self.blocks: h = blk(h)
        return self.proj_out(h)

class ResidualMLP(nn.Module):
    def __init__(self, a_dim, b_dim, hidden=384, n_blocks=3, dropout=0.1):
        super().__init__()
        self.a2b = _ResMLPBranch(a_dim, b_dim, hidden, n_blocks, dropout)
        self.b2a = _ResMLPBranch(b_dim, a_dim, hidden, n_blocks, dropout)
        self.a2g = self.a2b; self.g2a = self.b2a

    @torch.no_grad()
    def predict_a2b(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_b2a(self, x): return self.b2a(x)
    @torch.no_grad()
    def predict_atac2gex(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_gex2atac(self, x): return self.b2a(x)

def train_mlp(
    model, model_name, atac, gex, train_idx, val_idx,
    DEVICE, cfg, ckpt_name, early_abort_check_fn=None,
):
    """Train an MLP base model. atac (N, A_dim), gex (N, G_dim). Returns trained model."""
    print(f"\n=== Training {model_name} ===")
    print(f"  ATAC: {atac.shape}  GEX: {gex.shape}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(DEVICE)

    lr        = cfg[f"{model_name.upper()}_LR"]
    epochs    = cfg[f"{model_name.upper()}_EPOCHS"]
    batch     = cfg[f"{model_name.upper()}_BATCH"]
    patience_lim = cfg[f"{model_name.upper()}_PATIENCE"]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05) if cfg.get("USE_LR_SCHEDULER", False) else None

    use_fp16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "float16"
    use_bf16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "bfloat16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    def amp_ctx():
        if use_fp16: return torch.amp.autocast("cuda", dtype=torch.float16)
        if use_bf16: return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    atac_t = torch.from_numpy(atac).float()
    gex_t  = torch.from_numpy(gex).float()
    train_loader = DataLoader(TensorDataset(atac_t[train_idx], gex_t[train_idx]),
                                batch_size=batch, shuffle=True,
                                num_workers=cfg["NUM_WORKERS"], pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(TensorDataset(atac_t[val_idx], gex_t[val_idx]),
                                batch_size=batch, shuffle=False,
                                num_workers=cfg["NUM_WORKERS"])

    best_val, patience = float("inf"), 0
    ckpt_path = Path(cfg["CKPT_DIR"]) / ckpt_name
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for ab, gb in train_loader:
            ab = ab.to(DEVICE, non_blocking=True); gb = gb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with amp_ctx():
                pred_g = model.a2g(ab)
                pred_a = model.g2a(gb)
                loss = F.mse_loss(pred_g, gb) + F.mse_loss(pred_a, ab)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())

        model.eval(); v_losses = []
        with torch.no_grad():
            for ab, gb in val_loader:
                ab = ab.to(DEVICE); gb = gb.to(DEVICE)
                v_losses.append(F.mse_loss(model.predict_atac2gex(ab), gb).item() +
                                F.mse_loss(model.predict_gex2atac(gb), ab).item())
        if sched is not None: sched.step()
        v = float(np.mean(v_losses)); tr = float(np.mean(losses))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  ep{ep:3d}/{epochs}  train={tr:.4f}  val={v:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)")

        if early_abort_check_fn is not None:
            early_abort_check_fn(model_name, ep, model, gex, atac, val_idx, cfg)

        if v < best_val:
            best_val = v; patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= patience_lim:
                print(f"  → early stop @ epoch {ep}"); break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(DEVICE)
    return model

@torch.no_grad()
def mlp_predict(model, atac, gex, indices, DEVICE, batch=256):
    """Predict both directions on given cell indices. Returns (pred_a2g, pred_g2a)."""
    model.eval()
    pred_a2g, pred_g2a = [], []
    for i in range(0, len(indices), batch):
        sl = indices[i:i+batch]
        ab = torch.from_numpy(atac[sl]).float().to(DEVICE)
        gb = torch.from_numpy(gex [sl]).float().to(DEVICE)
        pred_a2g.append(model.predict_atac2gex(ab).cpu().numpy())
        pred_g2a.append(model.predict_gex2atac(gb).cpu().numpy())
    return (np.concatenate(pred_a2g).astype(np.float32),
            np.concatenate(pred_g2a).astype(np.float32))
