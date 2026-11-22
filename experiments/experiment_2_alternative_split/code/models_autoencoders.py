"""
CycleAE and ModDropAE.

CycleAE uses one encoder and one decoder per modality over a shared latent
space, trained on the cross-modal error, a self-reconstruction term and a
cycle-consistency term taken within each modality.

ModDropAE uses a single encoder over both modalities concatenated with two
presence flags, and withholds one modality at random during training. The
divergence term is dropped, so the mapping is deterministic at inference.
"""
import time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from contextlib import nullcontext

def _amp_ctx(cfg, DEVICE):
    if not (cfg["FP16"] and DEVICE.type == "cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if cfg["AMP_DTYPE"] == "bfloat16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)

def _grad_scaler(cfg, DEVICE):
    if cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "float16":
        return torch.amp.GradScaler("cuda")
    return None

class _CycleAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden, latent, n_layers, dropout):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, latent)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class _CycleAEDecoder(nn.Module):
    def __init__(self, latent, hidden, out_dim, n_layers, dropout):
        super().__init__()
        layers, d = [], latent
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class CycleAE(nn.Module):
    """A=source modality, B=target modality. For Multiome A=ATAC, B=GEX.
    For CITE-seq we instantiate as A=ADT (134), B=GEX (~14k).
    The bidirectional API uses A→B (.predict_a2b) and B→A (.predict_b2a)."""
    def __init__(self, a_dim, b_dim, hidden, latent, n_layers, dropout):
        super().__init__()
        self.enc_a = _CycleAEEncoder(a_dim, hidden, latent, n_layers, dropout)
        self.enc_b = _CycleAEEncoder(b_dim, hidden, latent, n_layers, dropout)
        self.dec_a = _CycleAEDecoder(latent, hidden, a_dim, n_layers, dropout)
        self.dec_b = _CycleAEDecoder(latent, hidden, b_dim, n_layers, dropout)

    def forward(self, a, b):
        z_a, z_b = self.enc_a(a), self.enc_b(b)
        return {
            "rec_a": self.dec_a(z_a),    "rec_b": self.dec_b(z_b),
            "pred_b_from_a": self.dec_b(z_a),
            "pred_a_from_b": self.dec_a(z_b),
        }

    @torch.no_grad()
    def predict_a2b(self, a): return self.dec_b(self.enc_a(a))
    @torch.no_grad()
    def predict_b2a(self, b): return self.dec_a(self.enc_b(b))

def train_cycleae(a, b, train_idx, val_idx, DEVICE, cfg, ckpt_path,
                modality_pair_label="multiome"):
    """Train CycleAE on (a_arr, b_arr) using indices. Returns trained model on DEVICE."""
    print(f"\n=== Training CycleAE ({modality_pair_label}) ===")
    print(f"  A: {a.shape}  B: {b.shape}")
    model = CycleAE(a.shape[1], b.shape[1],
                    hidden=cfg["CYCLEAE_HIDDEN"], latent=cfg["CYCLEAE_LATENT"],
                    n_layers=cfg["CYCLEAE_LAYERS"], dropout=cfg["CYCLEAE_DROPOUT"]).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["CYCLEAE_LR"], weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=cfg["CYCLEAE_EPOCHS"], eta_min=cfg["CYCLEAE_LR"]*0.05) if cfg.get("USE_LR_SCHEDULER", False) else None
    scaler = _grad_scaler(cfg, DEVICE)

    a_t = torch.from_numpy(a).float(); b_t = torch.from_numpy(b).float()
    train_loader = DataLoader(TensorDataset(a_t[train_idx], b_t[train_idx]),
                                batch_size=cfg["CYCLEAE_BATCH"], shuffle=True,
                                num_workers=cfg["NUM_WORKERS"], pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(TensorDataset(a_t[val_idx],  b_t[val_idx]),
                                batch_size=cfg["CYCLEAE_BATCH"], shuffle=False,
                                num_workers=cfg["NUM_WORKERS"])

    best_val, patience = float("inf"), 0
    for ep in range(1, cfg["CYCLEAE_EPOCHS"] + 1):
        model.train(); t0 = time.time(); losses = []
        for ab, bb in train_loader:
            ab = ab.to(DEVICE, non_blocking=True); bb = bb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with _amp_ctx(cfg, DEVICE):
                out = model(ab, bb)
                loss_cross = F.mse_loss(out["pred_b_from_a"], bb) + F.mse_loss(out["pred_a_from_b"], ab)
                loss_recon = F.mse_loss(out["rec_a"], ab) + F.mse_loss(out["rec_b"], bb)
                z_a = model.enc_a(ab); cyc_a = model.enc_a(model.dec_a(z_a))
                z_b = model.enc_b(bb); cyc_b = model.enc_b(model.dec_b(z_b))
                loss_cycle = F.mse_loss(cyc_a, z_a) + F.mse_loss(cyc_b, z_b)
                loss = loss_cross + cfg["CYCLEAE_LAMBDA_RECON"]*loss_recon + cfg["CYCLEAE_LAMBDA_CYCLE"]*loss_cycle
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())

        model.eval(); val_losses = []
        with torch.no_grad():
            for ab, bb in val_loader:
                ab = ab.to(DEVICE); bb = bb.to(DEVICE)
                val_losses.append(F.mse_loss(model.predict_a2b(ab), bb).item() +
                                  F.mse_loss(model.predict_b2a(bb), ab).item())
        if sched is not None: sched.step()
        v = float(np.mean(val_losses)); tr = float(np.mean(losses))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  ep{ep:3d}/{cfg['CYCLEAE_EPOCHS']}  train={tr:.4f}  val={v:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)")
        if v < best_val:
            best_val, patience = v, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg["CYCLEAE_PATIENCE"]:
                print(f"  → early stop @ ep{ep}"); break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.to(DEVICE)

@torch.no_grad()
def cycleae_predict(model, a, b, indices, DEVICE, batch=256):
    """Returns (pred_a2b, pred_b2a) on the given indices."""
    model.eval()
    pa2b, pb2a = [], []
    for i in range(0, len(indices), batch):
        sl = indices[i:i+batch]
        ab = torch.from_numpy(a[sl]).float().to(DEVICE)
        bb = torch.from_numpy(b[sl]).float().to(DEVICE)
        pa2b.append(model.predict_a2b(ab).cpu().numpy())
        pb2a.append(model.predict_b2a(bb).cpu().numpy())
    return (np.concatenate(pa2b).astype(np.float32),
            np.concatenate(pb2a).astype(np.float32))

class _ModDropAEEncoder(nn.Module):
    def __init__(self, a_dim, b_dim, hidden, latent, n_layers, dropout):
        super().__init__()
        layers, d = [], a_dim + b_dim + 2
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.net   = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_lv = nn.Linear(hidden, latent)
    def forward(self, a, b, ap, bp):
        h = self.net(torch.cat([a, b, ap, bp], dim=-1))
        return self.fc_mu(h), self.fc_lv(h)

class _ModDropAEDecoder(nn.Module):
    def __init__(self, latent, hidden, out_dim, n_layers, dropout):
        super().__init__()
        layers, d = [], latent
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class ModDropAE(nn.Module):
    """Joint-encoder autoencoder with modality dropout. The divergence term is
    omitted, so the model behaves as a deterministic autoencoder and the
    posterior cannot collapse. a_dim/b_dim are set per modality pair."""
    def __init__(self, a_dim, b_dim, hidden, latent, n_layers, dropout, modality_dropout_p):
        super().__init__()
        self.a_dim, self.b_dim = a_dim, b_dim
        self.modality_dropout_p = float(modality_dropout_p)
        self.encoder = _ModDropAEEncoder(a_dim, b_dim, hidden, latent, n_layers, dropout)
        self.dec_a   = _ModDropAEDecoder(latent, hidden, a_dim, n_layers, dropout)
        self.dec_b   = _ModDropAEDecoder(latent, hidden, b_dim, n_layers, dropout)

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var.clamp(-10, 10))
        return mu + torch.randn_like(std) * std

    def _mask(self, n, device):
        p = self.modality_dropout_p
        u = torch.rand(n, device=device)
        zero_a = (u < p).float().view(-1, 1)
        zero_b = ((u >= p) & (u < 2.0 * p)).float().view(-1, 1)
        return 1.0 - zero_a, 1.0 - zero_b

    def forward(self, a, b):
        ap, bp = self._mask(a.size(0), a.device)
        mu, lv = self.encoder(a*ap, b*bp, ap, bp)
        z = self.reparameterize(mu, lv)
        return {"mu": mu, "log_var": lv, "z": z,
                "rec_a": self.dec_a(z), "rec_b": self.dec_b(z),
                "a_present": ap, "b_present": bp}

    @torch.no_grad()
    def predict_a2b(self, a):
        n = a.size(0)
        z_b = torch.zeros(n, self.b_dim, device=a.device)
        ap  = torch.ones (n, 1, device=a.device)
        bp  = torch.zeros(n, 1, device=a.device)
        mu, _ = self.encoder(a, z_b, ap, bp)
        return self.dec_b(mu)

    @torch.no_grad()
    def predict_b2a(self, b):
        n = b.size(0)
        z_a = torch.zeros(n, self.a_dim, device=b.device)
        ap  = torch.zeros(n, 1, device=b.device)
        bp  = torch.ones (n, 1, device=b.device)
        mu, _ = self.encoder(z_a, b, ap, bp)
        return self.dec_a(mu)

def train_moddropae(a, b, train_idx, val_idx, DEVICE, cfg, ckpt_path,
                    modality_pair_label="multiome"):
    print(f"\n=== Training ModDropAE(KL=0) ({modality_pair_label}) ===")
    print(f"  A: {a.shape}  B: {b.shape}")
    model = ModDropAE(a.shape[1], b.shape[1],
                      hidden=cfg["MODDROPAE_HIDDEN"], latent=cfg["MODDROPAE_LATENT"],
                      n_layers=cfg["MODDROPAE_LAYERS"], dropout=0.1,
                      modality_dropout_p=cfg["MODDROPAE_DROPOUT_P"]).to(DEVICE)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  modality_dropout={cfg['MODDROPAE_DROPOUT_P']}  cross_w={cfg['MODDROPAE_CROSS_WEIGHT']}  KL=0")

    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["MODDROPAE_LR"], weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=cfg["MODDROPAE_EPOCHS"], eta_min=cfg["MODDROPAE_LR"]*0.05) if cfg.get("USE_LR_SCHEDULER", False) else None
    scaler = _grad_scaler(cfg, DEVICE)

    a_t = torch.from_numpy(a).float(); b_t = torch.from_numpy(b).float()
    train_loader = DataLoader(TensorDataset(a_t[train_idx], b_t[train_idx]),
                                batch_size=cfg["MODDROPAE_BATCH"], shuffle=True,
                                num_workers=cfg["NUM_WORKERS"], pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(TensorDataset(a_t[val_idx], b_t[val_idx]),
                                batch_size=cfg["MODDROPAE_BATCH"], shuffle=False,
                                num_workers=cfg["NUM_WORKERS"])

    cross_w = float(cfg["MODDROPAE_CROSS_WEIGHT"])
    best_val, patience = float("inf"), 0
    for ep in range(1, cfg["MODDROPAE_EPOCHS"] + 1):
        model.train(); t0 = time.time(); losses = []
        for ab, bb in train_loader:
            ab = ab.to(DEVICE, non_blocking=True); bb = bb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with _amp_ctx(cfg, DEVICE):
                out = model(ab, bb)
                w_a = (1.0 + cross_w * (1.0 - out["a_present"]))
                w_b = (1.0 + cross_w * (1.0 - out["b_present"]))
                rec_a = (((out["rec_a"] - ab) ** 2) * w_a).mean()
                rec_b = (((out["rec_b"] - bb) ** 2) * w_b).mean()
                loss = rec_a + rec_b
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())

        model.eval(); val_losses = []
        with torch.no_grad():
            for ab, bb in val_loader:
                ab = ab.to(DEVICE); bb = bb.to(DEVICE)
                val_losses.append(F.mse_loss(model.predict_a2b(ab), bb).item() +
                                  F.mse_loss(model.predict_b2a(bb), ab).item())
        if sched is not None: sched.step()
        v = float(np.mean(val_losses)); tr = float(np.mean(losses))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"  ep{ep:3d}/{cfg['MODDROPAE_EPOCHS']}  train={tr:.4f}  val={v:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)")
        if v < best_val:
            best_val, patience = v, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg["MODDROPAE_PATIENCE"]:
                print(f"  → early stop @ ep{ep}"); break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.to(DEVICE)

@torch.no_grad()
def moddropae_predict(model, a, b, indices, DEVICE, batch=256):
    model.eval()
    pa2b, pb2a = [], []
    for i in range(0, len(indices), batch):
        sl = indices[i:i+batch]
        ab = torch.from_numpy(a[sl]).float().to(DEVICE)
        bb = torch.from_numpy(b[sl]).float().to(DEVICE)
        pa2b.append(model.predict_a2b(ab).cpu().numpy())
        pb2a.append(model.predict_b2a(bb).cpu().numpy())
    return (np.concatenate(pa2b).astype(np.float32),
            np.concatenate(pb2a).astype(np.float32))
