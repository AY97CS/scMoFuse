"""
TransformerMLP.

A single-cell profile is not a sequence, so one is constructed: a feed-forward
stem projects the profile into a grid of tokens, pre-normalisation transformer
encoder layers let the tokens attend to one another, and a second feed-forward
block projects the flattened result onto the target. Tokens are arbitrary
groups of learned features and do not correspond to individual genes or peaks.
"""
import time, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from contextlib import nullcontext

class _TFMBranch(nn.Module):
    """One direction of the Transformer base (A → B)."""
    def __init__(self, in_dim, out_dim, n_tokens=32, d_model=64,
                  n_heads=4, n_layers=3, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, n_tokens * d_model),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ffn_mult, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(n_tokens * d_model, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        b = x.shape[0]
        h = self.input_proj(x).view(b, self.n_tokens, self.d_model)
        h = h + self.pos_emb
        h = self.transformer(h)
        h = h.flatten(1)
        return self.output_proj(h)

class TransformerMLP(nn.Module):
    """Bidirectional transformer cross-modal predictor. Same A↔B interface
    as CycleAE/ModDropAE/DeepMLP so the orchestrator can swap it in."""
    def __init__(self, a_dim, b_dim, n_tokens=32, d_model=64, n_heads=4,
                  n_layers=3, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.a2b = _TFMBranch(a_dim, b_dim, n_tokens, d_model, n_heads, n_layers, ffn_mult, dropout)
        self.b2a = _TFMBranch(b_dim, a_dim, n_tokens, d_model, n_heads, n_layers, ffn_mult, dropout)

        self.a2g = self.a2b; self.g2a = self.b2a

    @torch.no_grad()
    def predict_a2b(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_b2a(self, x): return self.b2a(x)
    @torch.no_grad()
    def predict_atac2gex(self, x): return self.a2b(x)
    @torch.no_grad()
    def predict_gex2atac(self, x): return self.b2a(x)

def train_transformer(model, model_name, a, b, train_idx, val_idx,
                        DEVICE, cfg, ckpt_path):
    """Train TransformerMLP. Cosine LR schedule."""
    print(f"\n=== Training {model_name} ===")
    print(f"  A: {a.shape}  B: {b.shape}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(DEVICE)

    lr        = cfg["TRANSFORMER_LR"]
    epochs    = cfg["TRANSFORMER_EPOCHS"]
    batch     = cfg["TRANSFORMER_BATCH"]
    patience_lim = cfg["TRANSFORMER_PATIENCE"]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05) if cfg.get("USE_LR_SCHEDULER", True) else None

    use_fp16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "float16"
    use_bf16 = cfg["FP16"] and DEVICE.type == "cuda" and cfg["AMP_DTYPE"] == "bfloat16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    def amp_ctx():
        if use_fp16: return torch.amp.autocast("cuda", dtype=torch.float16)
        if use_bf16: return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    a_t = torch.from_numpy(a).float(); b_t = torch.from_numpy(b).float()
    train_loader = DataLoader(TensorDataset(a_t[train_idx], b_t[train_idx]),
                                batch_size=batch, shuffle=True,
                                num_workers=cfg["NUM_WORKERS"], pin_memory=(DEVICE.type=="cuda"))
    val_loader   = DataLoader(TensorDataset(a_t[val_idx], b_t[val_idx]),
                                batch_size=batch, shuffle=False,
                                num_workers=cfg["NUM_WORKERS"])

    best_val, patience = float("inf"), 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for ab, bb in train_loader:
            ab = ab.to(DEVICE, non_blocking=True); bb = bb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with amp_ctx():
                pred_b = model.a2b(ab)
                pred_a = model.b2a(bb)
                loss = F.mse_loss(pred_b, bb) + F.mse_loss(pred_a, ab)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
        if sched is not None: sched.step()

        model.eval(); v_losses = []
        with torch.no_grad():
            for ab, bb in val_loader:
                ab = ab.to(DEVICE); bb = bb.to(DEVICE)
                v_losses.append(F.mse_loss(model.predict_a2b(ab), bb).item() +
                                F.mse_loss(model.predict_b2a(bb), ab).item())
        v = float(np.mean(v_losses)); tr = float(np.mean(losses))
        cur_lr = opt.param_groups[0]['lr']
        print(f"  ep{ep:3d}/{epochs}  train={tr:.4f}  val={v:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)")

        if v < best_val:
            best_val, patience = v, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= patience_lim:
                print(f"  → early stop @ ep{ep}"); break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.to(DEVICE)

@torch.no_grad()
def transformer_predict(model, a, b, indices, DEVICE, batch=256):
    """Predict both directions on the given indices."""
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
